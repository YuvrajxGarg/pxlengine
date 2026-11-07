//! Turns `pxl_scene::Scene`s into GPU data and path traces them, frame by frame.

pub mod film;
mod library;
mod program;

pub use program::ProgramSet;

use anyhow::{bail, Context as _, Result};
use glam::{Mat3, Vec3};
use library::{material_emission, Inst, Library};
use pxl_gpu::{vk, Binding, BlasInput, Buffer, ComputePipeline, Context, MemoryLocation, Resource, TlasInstance};
use pxl_scene::{LightShape, Node, Scene};
use std::path::Path;
use std::time::Instant;

const NONE: u32 = u32::MAX;

/// Quality presets. `--set samples=N` / `bounces=N` (or scene render.samples) override.
#[derive(Debug, Clone, Copy)]
pub struct Profile {
    pub name: &'static str,
    pub samples: u32,
    pub bounces: u32,
    pub denoise: bool,
}

pub const PROFILES: [Profile; 4] = [
    Profile { name: "std/previs", samples: 4, bounces: 1, denoise: true },
    Profile { name: "std/draft", samples: 16, bounces: 4, denoise: true },
    Profile { name: "std/final", samples: 64, bounces: 8, denoise: true },
    Profile { name: "std/path", samples: 1024, bounces: 16, denoise: false },
];

pub fn profile(name: &str) -> Option<Profile> {
    let n = name.trim_start_matches("std/");
    PROFILES.iter().copied().find(|p| p.name.trim_start_matches("std/") == n)
}

#[derive(Debug, Clone)]
pub struct RenderOptions {
    pub samples: u32,
    pub bounces: u32,
    pub seed: u32,
    /// Max radiance of indirect contributions (firefly control).
    pub clamp: f32,
    pub denoise: bool,
}

/// A rendered frame: linear HDR beauty plus first-hit passes (all averaged over samples).
pub struct Image {
    pub width: u32,
    pub height: u32,
    /// Beauty, linear RGB.
    pub rgb: Vec<f32>,
    /// First-hit base colour, linear RGB (background = environment colour, clamped).
    pub albedo: Vec<f32>,
    /// First-hit shading normal, world space, unit length (0 for background).
    pub normal: Vec<f32>,
    /// Distance along the camera axis to the first hit, metres (0 for background).
    pub depth: Vec<f32>,
    /// Geometry coverage 0..1.
    pub alpha: Vec<f32>,
    /// Scene node index of the first hit (u32::MAX for background), from the first sample.
    pub object: Vec<u32>,
    pub denoised: bool,
}

#[derive(Debug, Default, Clone)]
pub struct FrameStats {
    pub triangles: usize,
    pub instances: usize,
    pub lights: usize,
    pub emissive_triangles: usize,
    pub build_ms: f64,
    pub render_ms: f64,
    pub denoise_ms: f64,
    pub subframes: usize,
}

fn bytes<T: Copy>(v: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

pub struct Renderer {
    pub ctx: Context,
    pipe: ComputePipeline,
    denoiser: Option<std::result::Result<pxl_denoise::Denoiser, String>>,
    programs: ProgramSet,
    film_pipe: Option<ComputePipeline>,
}

/// Static (whole-render) GPU data.
struct Shared {
    vbuf: Buffer,
    ibuf: Buffer,
    tex_info: Buffer,
    texels: Buffer,
    env_pix: Buffer,
    env_marg: Buffer,
    env_cond: Buffer,
    env: Option<(u32, u32, f32)>,
}

impl Renderer {
    /// Compile the path tracer with `programs` spliced in (use `ProgramSet::none()` for none).
    pub fn new(programs: ProgramSet) -> Result<Renderer> {
        let mut ctx = Context::new()?;
        let what = if programs.camera || programs.propagate || programs.film || !programs.material_ids.is_empty() {
            "compiling path tracer with scene programs"
        } else {
            "compiling path tracer"
        };
        let spirv = pxl_gpu::slang::compile_compute(&programs.source, "main").context(what)?;
        let mut bindings = vec![(0, Binding::Accel)];
        bindings.extend((1..=15).map(|i| (i, Binding::Storage)));
        bindings.push((17, Binding::Storage));
        let pipe = ctx.create_compute(&spirv, "main", &bindings, 16)?;
        let film_pipe = if programs.film {
            let spirv = pxl_gpu::slang::compile_compute(&programs.source, "film_main").context("compiling film program")?;
            Some(ctx.create_compute(&spirv, "main" /* slangc names the compiled entry point main */, &[(16, Binding::Storage), (17, Binding::Storage), (18, Binding::Storage)], 16)?)
        } else {
            None
        };
        Ok(Renderer { ctx, pipe, denoiser: None, programs, film_pipe })
    }

    /// Name of the denoiser device, or why it's unavailable.
    pub fn denoiser_status(&mut self) -> String {
        if self.denoiser.is_none() {
            self.denoiser = Some(pxl_denoise::Denoiser::new().map_err(|e| format!("{e:#}")));
        }
        match self.denoiser.as_ref().unwrap() {
            Ok(d) => format!("OIDN on {} ({})", d.device_name, d.path.display()),
            Err(e) => e.clone(),
        }
    }

    /// Render every scene in `scenes` (typically consecutive animation frames), calling `sink`
    /// with each finished image. Geometry and textures are loaded and uploaded once.
    /// `frames[i]` holds the scene evaluated at each shutter subframe of frame i (one entry
    /// without motion blur). Calls `sink` with each finished image. Geometry and textures are
    /// loaded and uploaded once for the whole sequence.
    pub fn render_sequence(
        &mut self,
        frames: Vec<Vec<Scene>>,
        base_dir: &Path,
        opts: &RenderOptions,
        mut sink: impl FnMut(usize, Image, &FrameStats) -> Result<()>,
    ) -> Result<Vec<String>> {
        let mut lib = Library::new(base_dir, self.programs.material_ids.clone());
        let mut prepared: Vec<Vec<(Scene, Vec<Inst>)>> = Vec::with_capacity(frames.len());
        for subframes in frames {
            let mut list = Vec::with_capacity(subframes.len());
            for s in subframes {
                let insts = lib.instances(&s)?;
                lib.load_env(&s.world)?;
                list.push((s, insts));
            }
            prepared.push(list);
        }
        let shared = self.upload_shared(&mut lib)?;
        let mut warnings = std::mem::take(&mut lib.warnings);
        for (i, subframes) in prepared.iter().enumerate() {
            let (img, stats) = self.render_frame(&lib, &shared, subframes, opts, &mut warnings)?;
            sink(i, img, &stats)?;
        }
        Ok(warnings)
    }

    fn upload_shared(&mut self, lib: &mut Library) -> Result<Shared> {
        let ctx = &mut self.ctx;
        let storage = vk::BufferUsageFlags::STORAGE_BUFFER;
        let as_input = vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR;

        // Geometry: (position, u), (normal, v) per vertex.
        let mut verts: Vec<[f32; 4]> = Vec::new();
        let mut indices: Vec<u32> = Vec::new();
        for m in &mut lib.meshes {
            m.vertex_offset = (verts.len() / 2) as u32;
            m.index_offset = indices.len() as u32;
            let d = &m.data;
            for i in 0..d.positions.len() {
                let (p, n) = (d.positions[i], d.normals[i]);
                let uv = d.uvs.get(i).copied().unwrap_or([0.0, 0.0]);
                verts.push([p[0], p[1], p[2], uv[0]]);
                verts.push([n[0], n[1], n[2], uv[1]]);
            }
            indices.extend_from_slice(&d.indices);
        }
        if verts.is_empty() {
            verts.push([0.0; 4]);
            indices.push(0);
        }
        let vbuf = ctx.create_buffer_init(bytes(&verts), storage | as_input, "vertices")?;
        let ibuf = ctx.create_buffer_init(bytes(&indices), storage | as_input, "indices")?;
        for m in &mut lib.meshes {
            let blas = ctx.build_blas(&BlasInput {
                vertex_address: vbuf.address + m.vertex_offset as u64 * 32,
                vertex_stride: 32,
                vertex_count: m.data.positions.len() as u32,
                index_address: ibuf.address + m.index_offset as u64 * 4,
                triangle_count: m.data.triangle_count() as u32,
                opaque: m.opaque,
            })?;
            m.blas = Some(blas);
        }

        // Textures: RGBA8 texels packed into one buffer.
        let mut info: Vec<[u32; 4]> = Vec::new();
        let mut texels: Vec<u8> = Vec::new();
        for t in &lib.textures {
            info.push([(texels.len() / 4) as u32, t.width, t.height, 0]);
            texels.extend_from_slice(&t.pixels);
        }
        info.push([0; 4]);
        texels.extend_from_slice(&[0; 4]);
        if texels.len() as u64 > 4 << 30 {
            bail!("textures total {} MB; the current limit is 4096 MB", texels.len() >> 20);
        }
        let tex_info = ctx.create_buffer_init(bytes(&info), storage, "texture info")?;
        let texels = ctx.create_buffer_init(&texels, storage, "texels")?;

        // Environment map + importance sampling tables.
        let (env_pix, env_marg, env_cond, env) = match &lib.env {
            Some((_, img)) => {
                let (marg, cond, total) = library::env_cdf(img);
                (
                    ctx.create_buffer_init(bytes(&img.rgb), storage, "env pixels")?,
                    ctx.create_buffer_init(bytes(&marg), storage, "env marginal")?,
                    ctx.create_buffer_init(bytes(&cond), storage, "env conditional")?,
                    Some((img.width, img.height, total)),
                )
            }
            None => {
                let z = [0f32; 4];
                (
                    ctx.create_buffer_init(bytes(&z), storage, "env pixels")?,
                    ctx.create_buffer_init(bytes(&z), storage, "env marginal")?,
                    ctx.create_buffer_init(bytes(&z), storage, "env conditional")?,
                    None,
                )
            }
        };
        Ok(Shared { vbuf, ibuf, tex_info, texels, env_pix, env_marg, env_cond, env })
    }

    /// Render one frame: subframes (shutter samples for motion blur) accumulate into the same
    /// targets, then the result is read back, split into passes, and optionally denoised.
    fn render_frame(
        &mut self,
        lib: &Library,
        sh: &Shared,
        subframes: &[(Scene, Vec<Inst>)],
        opts: &RenderOptions,
        warnings: &mut Vec<String>,
    ) -> Result<(Image, FrameStats)> {
        let [w, h] = subframes[0].0.render.resolution;
        let pixels = w as u64 * h as u64;
        let storage = vk::BufferUsageFlags::STORAGE_BUFFER;
        let src = vk::BufferUsageFlags::TRANSFER_SRC;
        let ctx = &mut self.ctx;
        let accum = ctx.create_buffer(pixels * 16, storage | src, MemoryLocation::GpuOnly, "accum")?;
        let aux = ctx.create_buffer(pixels * 48, storage | src, MemoryLocation::GpuOnly, "aux")?;
        let readback = ctx.create_buffer(pixels * 64, vk::BufferUsageFlags::TRANSFER_DST, MemoryLocation::GpuToCpu, "readback")?;

        let mut stats = FrameStats::default();
        let k = subframes.len() as u32;
        let mut next = 0;
        for (i, (scene, insts)) in subframes.iter().enumerate() {
            // Spread samples evenly over the shutter; every subframe gets at least one.
            let n = (opts.samples / k + u32::from((i as u32) < opts.samples % k)).max(1);
            self.render_subframe(lib, sh, scene, insts, opts, (accum, aux), next..next + n, &mut stats)?;
            next += n;
        }

        let t1 = Instant::now();
        let c = &self.ctx;
        c.submit(|cmd| unsafe {
            c.device.cmd_copy_buffer(cmd, accum.handle, readback.handle, &[vk::BufferCopy { src_offset: 0, dst_offset: 0, size: pixels * 16 }]);
            c.device.cmd_copy_buffer(cmd, aux.handle, readback.handle, &[vk::BufferCopy { src_offset: 0, dst_offset: pixels * 16, size: pixels * 48 }]);
        })?;
        let raw = c.read(&readback)?;
        let data: &[f32] = unsafe { std::slice::from_raw_parts(raw.as_ptr() as *const f32, raw.len() / 4) };
        let np = pixels as usize;
        let mut img = Image {
            width: w,
            height: h,
            rgb: Vec::with_capacity(np * 3),
            albedo: Vec::with_capacity(np * 3),
            normal: Vec::with_capacity(np * 3),
            depth: Vec::with_capacity(np),
            alpha: Vec::with_capacity(np),
            object: Vec::with_capacity(np),
            denoised: false,
        };
        let (beauty, aovs) = data.split_at(np * 4);
        for (px, a) in beauty.chunks_exact(4).zip(aovs.chunks_exact(12)) {
            let inv = 1.0 / px[3].max(1.0);
            img.rgb.extend_from_slice(&[px[0] * inv, px[1] * inv, px[2] * inv]);
            img.albedo.extend_from_slice(&[a[0] * inv, a[1] * inv, a[2] * inv]);
            img.alpha.push(a[3] * inv);
            // Depth averages only over samples that hit something.
            img.depth.push(if a[3] > 0.0 { a[7] / a[3] } else { 0.0 });
            let n = Vec3::new(a[4], a[5], a[6]).normalize_or_zero();
            img.normal.extend_from_slice(&n.to_array());
            img.object.push(a[8].to_bits());
        }
        let ctx = &mut self.ctx;
        for b in [accum, aux, readback] {
            ctx.destroy_buffer(b);
        }

        if opts.denoise {
            if self.denoiser.is_none() {
                self.denoiser = Some(pxl_denoise::Denoiser::new().map_err(|e| format!("{e:#}")));
            }
            match self.denoiser.as_ref().unwrap() {
                Ok(d) => {
                    let q = if opts.samples >= 32 { pxl_denoise::Quality::High } else { pxl_denoise::Quality::Balanced };
                    d.denoise(&mut img.rgb, Some(&img.albedo), Some(&img.normal), w, h, q)?;
                    img.denoised = true;
                }
                Err(e) => {
                    let msg = format!("denoiser unavailable, output is noisy: {e}");
                    if !warnings.contains(&msg) {
                        warnings.push(msg);
                    }
                }
            }
        }
        if let Some(film) = self.film_pipe {
            // Film program: one GPU pass over the finished (denoised) frame.
            let scene = &subframes[subframes.len() / 2].0;
            let mut input: Vec<[f32; 4]> = Vec::with_capacity(np * 3);
            for i in 0..np {
                let (c, n, a) = (&img.rgb[i * 3..i * 3 + 3], &img.normal[i * 3..i * 3 + 3], &img.albedo[i * 3..i * 3 + 3]);
                input.push([c[0], c[1], c[2], img.alpha[i]]);
                input.push([n[0], n[1], n[2], img.depth[i]]);
                input.push([a[0], a[1], a[2], 0.0]);
            }
            let ctx = &mut self.ctx;
            let fin = ctx.create_buffer_init(bytes(&input), storage, "film in")?;
            let par = ctx.create_buffer_init(bytes(&self.programs.param_data(scene)), storage, "film params")?;
            let fout = ctx.create_buffer(pixels * 16, storage, MemoryLocation::GpuToCpu, "film out")?;
            ctx.bind(&film, &[(16, Resource::Buffer(fin)), (17, Resource::Buffer(par)), (18, Resource::Buffer(fout))]);
            let time = (scene.frame / scene.render.fps as f64) as f32;
            let push = [w.to_le_bytes(), h.to_le_bytes(), time.to_le_bytes(), (scene.frame as f32).to_le_bytes()].concat();
            let c = &*ctx;
            c.submit(|cmd| c.dispatch(cmd, &film, &push, [w.div_ceil(8), h.div_ceil(8), 1]))?;
            let raw = c.read(&fout)?;
            let out: &[f32] = unsafe { std::slice::from_raw_parts(raw.as_ptr() as *const f32, raw.len() / 4) };
            for (i, px) in out.chunks_exact(4).take(np).enumerate() {
                img.rgb[i * 3..i * 3 + 3].copy_from_slice(&px[..3]);
            }
            for b in [fin, par, fout] {
                ctx.destroy_buffer(b);
            }
        }
        stats.denoise_ms = t1.elapsed().as_secs_f64() * 1e3;
        stats.subframes = subframes.len();
        Ok((img, stats))
    }

    #[allow(clippy::too_many_arguments)]
    fn render_subframe(
        &mut self,
        lib: &Library,
        sh: &Shared,
        scene: &Scene,
        insts: &[Inst],
        opts: &RenderOptions,
        (accum, aux): (Buffer, Buffer),
        samples: std::ops::Range<u32>,
        stats: &mut FrameStats,
    ) -> Result<()> {
        let t0 = Instant::now();
        stats.instances = insts.len();
        stats.triangles = 0;
        let storage = vk::BufferUsageFlags::STORAGE_BUFFER;

        // ---- instances, materials, emissive triangles
        let mut inst_rec: Vec<[u32; 16]> = Vec::with_capacity(insts.len() + 1);
        let mut mats: Vec<[f32; 24]> = Vec::with_capacity(insts.len() + 1);
        let mut emissive: Vec<[f32; 4]> = Vec::new();
        let mut weights: Vec<f64> = Vec::new();
        let mut tlas_inst = Vec::with_capacity(insts.len());
        for (i, inst) in insts.iter().enumerate() {
            let mesh = &lib.meshes[inst.mesh];
            stats.triangles += mesh.data.triangle_count();
            let m3 = Mat3::from_mat4(inst.xform);
            let nm = if m3.determinant().abs() > 1e-24 { m3.inverse().transpose() } else { Mat3::IDENTITY };
            let e = material_emission(&inst.material);
            let e_lum = 0.2126 * e[0] + 0.7152 * e[1] + 0.0722 * e[2];
            let emissive_offset = if e_lum > 0.0 {
                let off = weights.len() as u32;
                let d = &mesh.data;
                let etex = library::material_texture(&inst.material, 3).and_then(|t| lib.textures.get(t));
                let uv_scale = library::material_uv_scale(&inst.material);
                for t in d.indices.chunks_exact(3) {
                    let p = [t[0], t[1], t[2]].map(|k| inst.xform.transform_point3(Vec3::from(d.positions[k as usize])));
                    let uv = [t[0], t[1], t[2]].map(|k| d.uvs.get(k as usize).copied().unwrap_or([0.0, 0.0]));
                    let area = 0.5 * (p[1] - p[0]).cross(p[2] - p[0]).length();
                    // Weight by the emission texture's brightness over the triangle, so mostly-black
                    // emission maps don't waste light samples. Zero-weight triangles are still lit by
                    // BSDF rays (MIS gives them full weight there), so this stays unbiased.
                    let tex_lum = etex.map_or(1.0, |img| library::triangle_texture_luminance(img, uv, uv_scale));
                    weights.push((area * e_lum * tex_lum) as f64);
                    emissive.push([p[0].x, p[0].y, p[0].z, uv[0][0]]);
                    emissive.push([p[1].x, p[1].y, p[1].z, uv[0][1]]);
                    emissive.push([p[2].x, p[2].y, p[2].z, uv[1][0]]);
                    emissive.push([uv[1][1], uv[2][0], uv[2][1], f32::from_bits(i as u32)]);
                }
                off
            } else {
                NONE
            };
            let mut r = [0u32; 16];
            r[0] = mesh.vertex_offset;
            r[1] = mesh.index_offset;
            r[2] = i as u32;
            r[3] = emissive_offset;
            r[7] = inst.object;
            for row in 0..3 {
                let v = nm.row(row);
                r[4 + row * 4] = v.x.to_bits();
                r[5 + row * 4] = v.y.to_bits();
                r[6 + row * 4] = v.z.to_bits();
            }
            inst_rec.push(r);
            mats.push(inst.material);
            let m = inst.xform.transpose().to_cols_array();
            let mut transform = [0f32; 12];
            transform.copy_from_slice(&m[..12]);
            tlas_inst.push(TlasInstance {
                transform,
                custom_index: i as u32,
                mask: 0xff,
                blas_address: mesh.blas.expect("uploaded").address,
            });
        }
        // GPU light table size (all triangles of emissive instances, so `offset + prim` indexing
        // works); the stat reports only triangles that can actually be sampled.
        let mut emissive_count = weights.len() as u32;
        stats.emissive_triangles = weights.iter().filter(|w| **w > 0.0).count();
        let total: f64 = weights.iter().sum();
        let mut cdf: Vec<f32> = Vec::with_capacity(weights.len() + 1);
        let mut acc = 0f64;
        cdf.push(0.0);
        for w in &weights {
            acc += w;
            cdf.push(if total > 0.0 { (acc / total) as f32 } else { 0.0 });
        }
        if let Some(last) = cdf.last_mut() {
            *last = 1.0;
        }
        if total <= 0.0 {
            emissive.clear();
            emissive_count = 0;
            stats.emissive_triangles = 0;
        }

        // ---- analytic lights
        let mut lights: Vec<[f32; 16]> = Vec::new();
        for node in &scene.nodes {
            let Node::Light(l) = node else { continue };
            let pos = Vec3::from(l.translation);
            let color = Vec3::from(l.color);
            let dir = (Vec3::from(l.target) - pos).try_normalize().unwrap_or(Vec3::NEG_Y);
            let mut rec = [0f32; 16];
            rec[0..3].copy_from_slice(&l.translation);
            match l.shape {
                LightShape::Point => {
                    rec[3] = 0.0;
                    rec[4..7].copy_from_slice(&(color * l.energy / (4.0 * std::f32::consts::PI)).to_array());
                }
                LightShape::Rectangle => {
                    rec[3] = 1.0;
                    let [w, h] = l.size.xy();
                    let up = if dir.y.abs() > 0.99 { Vec3::Z } else { Vec3::Y };
                    let u = up.cross(dir).normalize(); // u x v points along `dir`
                    let v = dir.cross(u).normalize();
                    let radiance = color * l.energy / (std::f32::consts::PI * w * h).max(1e-6);
                    rec[4..7].copy_from_slice(&radiance.to_array());
                    rec[8..11].copy_from_slice(&(u * w * 0.5).to_array());
                    rec[12..15].copy_from_slice(&(v * h * 0.5).to_array());
                }
                LightShape::Directional => {
                    rec[3] = 2.0;
                    rec[4..7].copy_from_slice(&(color * l.energy).to_array());
                    rec[8..11].copy_from_slice(&dir.to_array());
                }
            }
            lights.push(rec);
        }
        stats.lights = lights.len();

        // ---- light category probabilities
        // With a propagation program rays bend, so straight-line light sampling of glowing
        // surfaces and the environment would be wrong: those are then found by path tracing
        // alone (MIS weights fall back to 1). Analytic lights keep straight shadow rays.
        let curved = self.programs.propagate;
        let has = [!lights.is_empty(), emissive_count > 0 && !curved, sh.env.is_some() && !curved];
        let n = has.iter().filter(|h| **h).count().max(1) as f32;
        let p = has.map(|h| if h { 1.0 / n } else { 0.0 });

        // ---- camera + frame constants
        let [w, h] = scene.render.resolution;
        let cam = scene.camera();
        let eye = Vec3::from(cam.translation);
        let fwd = (Vec3::from(cam.target) - eye).try_normalize().unwrap_or(Vec3::NEG_Z);
        let world_up = if fwd.y.abs() > 0.999 { Vec3::Z } else { Vec3::Y };
        let right = fwd.cross(world_up).normalize();
        let up = right.cross(fwd);
        let th = (cam.fov.to_radians() * 0.5).tan();
        let aspect = w as f32 / h as f32;
        let world = Vec3::from(scene.world.color) * scene.world.strength;
        let (env_w, env_h, env_total) = sh.env.unwrap_or((0, 0, 0.0));
        let mut frame: Vec<u8> = Vec::with_capacity(144);
        for v in [
            eye.extend(0.0),
            fwd.extend(0.0),
            (right * th * aspect).extend(0.0),
            (up * th).extend(0.0),
            world.extend(scene.world.strength),
        ] {
            frame.extend_from_slice(bytes(&v.to_array()));
        }
        frame.extend_from_slice(bytes(&[w, h, lights.len() as u32, opts.bounces]));
        frame.extend_from_slice(bytes(&[opts.seed, emissive_count, env_w, env_h]));
        frame.extend_from_slice(bytes(&[scene.world.rotation / 360.0, env_total, p[0], p[1]]));
        frame.extend_from_slice(bytes(&[p[2], opts.clamp, 0.0, 0.0]));
        let (aperture, focus) = cam.lens();
        let time = (scene.frame / scene.render.fps as f64) as f32;
        frame.extend_from_slice(bytes(&[aperture, focus, time, scene.frame as f32]));

        // ---- upload per-frame data
        let ctx = &mut self.ctx;
        inst_rec.push([0; 16]);
        mats.push([0.0; 24]);
        lights.push([0.0; 16]);
        emissive.push([0.0; 4]);
        let inst_buf = ctx.create_buffer_init(bytes(&inst_rec), storage, "instances")?;
        let mat_buf = ctx.create_buffer_init(bytes(&mats), storage, "materials")?;
        let light_buf = ctx.create_buffer_init(bytes(&lights), storage, "lights")?;
        let em_buf = ctx.create_buffer_init(bytes(&emissive), storage, "emissive")?;
        let cdf_buf = ctx.create_buffer_init(bytes(&cdf), storage, "emissive cdf")?;
        let frame_buf = ctx.create_buffer_init(&frame, storage, "frame")?;
        let par_buf = ctx.create_buffer_init(bytes(&self.programs.param_data(scene)), storage, "params")?;
        let tlas = ctx.build_tlas(&tlas_inst)?;
        let pixels = w as u64 * h as u64;
        let resources = [
            Resource::Accel(tlas.handle),
            Resource::Buffer(sh.vbuf),
            Resource::Buffer(sh.ibuf),
            Resource::Buffer(inst_buf),
            Resource::Buffer(mat_buf),
            Resource::Buffer(light_buf),
            Resource::Buffer(accum),
            Resource::Buffer(sh.tex_info),
            Resource::Buffer(sh.texels),
            Resource::Buffer(em_buf),
            Resource::Buffer(cdf_buf),
            Resource::Buffer(sh.env_pix),
            Resource::Buffer(sh.env_marg),
            Resource::Buffer(sh.env_cond),
            Resource::Buffer(frame_buf),
            Resource::Buffer(aux),
        ];
        let mut bound: Vec<(u32, Resource)> = resources.into_iter().enumerate().map(|(i, r)| (i as u32, r)).collect();
        bound.push((17, Resource::Buffer(par_buf)));
        ctx.bind(&self.pipe, &bound);
        stats.build_ms += t0.elapsed().as_secs_f64() * 1e3;

        // ---- progressive samples, batched to stay well under the OS GPU watchdog
        let t1 = Instant::now();
        let groups = [w.div_ceil(8), h.div_ceil(8), 1];
        let batch = (4_000_000 / pixels.max(1)).clamp(1, 32) as u32;
        let pipe = self.pipe;
        let mut s = samples.start;
        while s < samples.end {
            let end = (s + batch).min(samples.end);
            let c = &*ctx;
            c.submit(|cmd| {
                for i in s..end {
                    c.dispatch(cmd, &pipe, bytes(&[i, 0u32, 0, 0]), groups);
                    c.barrier(cmd, vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::COMPUTE_SHADER);
                }
            })?;
            s = end;
        }
        stats.render_ms += t1.elapsed().as_secs_f64() * 1e3;
        for b in [inst_buf, mat_buf, light_buf, em_buf, cdf_buf, frame_buf, par_buf] {
            ctx.destroy_buffer(b);
        }
        ctx.destroy_accel(tlas);
        Ok(())
    }
}
