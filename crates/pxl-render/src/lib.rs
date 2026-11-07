//! Turns a `pxl_scene::Scene` into GPU data and path traces it.

use anyhow::{Context as _, Result};
use glam::{Mat3, Vec3};
use pxl_gpu::{vk, Binding, BlasInput, Context, MemoryLocation, Resource, TlasInstance};
use pxl_scene::{geometry, LightShape, Node, Scene};
use std::time::Instant;

const PATHTRACE_SRC: &str = include_str!("../../../shaders/pathtrace.slang");

/// Quality presets. `--set samples=N` / `bounces=N` (or scene render.samples) override.
#[derive(Debug, Clone, Copy)]
pub struct Profile {
    pub name: &'static str,
    pub samples: u32,
    pub bounces: u32,
}

pub const PROFILES: [Profile; 3] = [
    Profile { name: "std/previs", samples: 8, bounces: 1 },
    Profile { name: "std/draft", samples: 64, bounces: 4 },
    Profile { name: "std/final", samples: 256, bounces: 8 },
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
}

/// Linear HDR image, RGB f32.
pub struct Image {
    pub width: u32,
    pub height: u32,
    pub rgb: Vec<f32>,
}

#[derive(Debug, Default)]
pub struct Stats {
    pub triangles: usize,
    pub instances: usize,
    pub lights: usize,
    pub setup_ms: f64,
    pub render_ms: f64,
}

fn bytes<T: Copy>(v: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

pub fn render(ctx: &mut Context, scene: &Scene, opts: &RenderOptions) -> Result<(Image, Stats)> {
    let t0 = Instant::now();
    let mut stats = Stats::default();

    // ---- gather geometry, materials, instances
    let mut verts: Vec<[f32; 4]> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();
    let mut materials: Vec<[f32; 8]> = Vec::new();
    let mut instances: Vec<[f32; 16]> = Vec::new();
    struct Pending { vertex_offset: u32, vertex_count: u32, index_offset: u32, tri_count: u32, transform: glam::Mat4 }
    let mut pending = Vec::new();

    for node in &scene.nodes {
        let Node::Mesh(m) = node else { continue };
        let mesh = geometry::build(&m.geometry);
        let vertex_offset = (verts.len() / 2) as u32;
        let index_offset = indices.len() as u32;
        for (p, n) in mesh.positions.iter().zip(&mesh.normals) {
            verts.push([p[0], p[1], p[2], 1.0]);
            verts.push([n[0], n[1], n[2], 0.0]);
        }
        indices.extend_from_slice(&mesh.indices);
        let inputs = m.materials.first().map(|m| m.inputs.clone()).unwrap_or_default();
        let e = inputs.emission_luminance;
        materials.push([
            inputs.base_color[0], inputs.base_color[1], inputs.base_color[2], inputs.specular_roughness,
            inputs.emission_color[0] * e, inputs.emission_color[1] * e, inputs.emission_color[2] * e, inputs.base_metalness,
        ]);
        let xf = m.transform();
        let nm = Mat3::from_mat4(xf).inverse().transpose();
        let mat_index = (materials.len() - 1) as u32;
        let mut inst = [0f32; 16];
        inst[0] = f32::from_bits(vertex_offset);
        inst[1] = f32::from_bits(index_offset);
        inst[2] = f32::from_bits(mat_index);
        for r in 0..3 {
            let row = nm.row(r);
            inst[4 + r * 4..7 + r * 4].copy_from_slice(&row.to_array());
        }
        instances.push(inst);
        stats.triangles += mesh.triangle_count();
        pending.push(Pending {
            vertex_offset,
            vertex_count: mesh.positions.len() as u32,
            index_offset,
            tri_count: mesh.triangle_count() as u32,
            transform: xf,
        });
    }
    stats.instances = instances.len();

    // ---- lights
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
                let i = color * l.energy / (4.0 * std::f32::consts::PI); // W/sr
                rec[4..7].copy_from_slice(&i.to_array());
            }
            LightShape::Rectangle => {
                rec[3] = 1.0;
                let [w, h] = l.size.xy();
                let up = if dir.y.abs() > 0.99 { Vec3::Z } else { Vec3::Y };
                // u x v must point along `dir` (the emitting side).
                let u = up.cross(dir).normalize();
                let v = dir.cross(u).normalize();
                let (u, v) = (u * w * 0.5, v * h * 0.5);
                let radiance = color * l.energy / (std::f32::consts::PI * w * h).max(1e-6);
                rec[4..7].copy_from_slice(&radiance.to_array());
                rec[8..11].copy_from_slice(&u.to_array());
                rec[12..15].copy_from_slice(&v.to_array());
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

    // ---- upload + acceleration structures
    let storage = vk::BufferUsageFlags::STORAGE_BUFFER;
    let as_input = vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR;
    if verts.is_empty() {
        verts.push([0.0; 4]);
        indices.push(0);
    }
    let vbuf = ctx.create_buffer_init(bytes(&verts), storage | as_input, "vertices")?;
    let ibuf = ctx.create_buffer_init(bytes(&indices), storage | as_input, "indices")?;
    let inst_buf = ctx.create_buffer_init(bytes(&instances.iter().copied().chain([[0f32; 16]]).collect::<Vec<_>>()), storage, "instances")?;
    let mat_buf = ctx.create_buffer_init(bytes(&materials.iter().copied().chain([[0f32; 8]]).collect::<Vec<_>>()), storage, "materials")?;
    let light_buf = ctx.create_buffer_init(bytes(&lights.iter().copied().chain([[0f32; 16]]).collect::<Vec<_>>()), storage, "lights")?;

    let mut tlas_instances = Vec::new();
    for (i, p) in pending.iter().enumerate() {
        let (_, addr) = ctx.build_blas(&BlasInput {
            vertex_address: vbuf.address + p.vertex_offset as u64 * 32,
            vertex_stride: 32,
            vertex_count: p.vertex_count,
            index_address: ibuf.address + p.index_offset as u64 * 4,
            triangle_count: p.tri_count,
        })?;
        let m = p.transform.transpose().to_cols_array(); // rows of the original
        let mut t = [0f32; 12];
        t.copy_from_slice(&m[..12]);
        tlas_instances.push(TlasInstance { transform: t, custom_index: i as u32, mask: 0xff, blas_address: addr });
    }
    let tlas = ctx.build_tlas(&tlas_instances)?;

    // ---- pipeline
    let spirv = pxl_gpu::slang::compile_compute(PATHTRACE_SRC, "main").context("compiling path tracer")?;
    let pipe = ctx.create_compute(
        &spirv,
        "main",
        &[Binding::Accel, Binding::Storage, Binding::Storage, Binding::Storage, Binding::Storage, Binding::Storage, Binding::Storage],
        112,
    )?;
    let [w, h] = scene.render.resolution;
    let pixels = w as u64 * h as u64;
    let accum = ctx.create_buffer(pixels * 16, storage | vk::BufferUsageFlags::TRANSFER_SRC, MemoryLocation::GpuOnly, "accum")?;
    let readback = ctx.create_buffer(pixels * 16, vk::BufferUsageFlags::TRANSFER_DST, MemoryLocation::GpuToCpu, "readback")?;
    ctx.bind(
        &pipe,
        &[
            Resource::Accel(tlas),
            Resource::Buffer(vbuf),
            Resource::Buffer(ibuf),
            Resource::Buffer(inst_buf),
            Resource::Buffer(mat_buf),
            Resource::Buffer(light_buf),
            Resource::Buffer(accum),
        ],
    );

    // ---- camera
    let cam = scene.camera();
    let eye = Vec3::from(cam.translation);
    let fwd = (Vec3::from(cam.target) - eye).try_normalize().unwrap_or(Vec3::NEG_Z);
    let world_up = if fwd.y.abs() > 0.999 { Vec3::Z } else { Vec3::Y };
    let right = fwd.cross(world_up).normalize();
    let up = right.cross(fwd);
    let th = (cam.fov.to_radians() * 0.5).tan();
    let aspect = w as f32 / h as f32;
    let world = Vec3::from(scene.world.color) * scene.world.strength;

    stats.setup_ms = t0.elapsed().as_secs_f64() * 1e3;
    let t1 = Instant::now();

    // ---- progressive samples, batched to stay well under the OS GPU watchdog
    let groups = [w.div_ceil(8), h.div_ceil(8), 1];
    let batch = (8_000_000 / pixels.max(1)).clamp(1, 64) as u32;
    let mut s = 0;
    while s < opts.samples {
        let end = (s + batch).min(opts.samples);
        let ctx_ref = &*ctx;
        ctx_ref.submit(|cmd| {
            for i in s..end {
                let mut push = Vec::with_capacity(112);
                for v in [eye.extend(0.0), fwd.extend(0.0), (right * th * aspect).extend(0.0), (up * th).extend(0.0), world.extend(0.0)] {
                    push.extend_from_slice(bytes(&v.to_array()));
                }
                push.extend_from_slice(bytes(&[w, h, i, lights.len() as u32, opts.seed, opts.bounces, 0, 0]));
                ctx_ref.dispatch(cmd, &pipe, &push, groups);
                ctx_ref.barrier(cmd, vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::COMPUTE_SHADER);
            }
        })?;
        s = end;
    }
    ctx.submit(|cmd| unsafe {
        ctx.device.cmd_copy_buffer(cmd, accum.handle, readback.handle, &[vk::BufferCopy { src_offset: 0, dst_offset: 0, size: pixels * 16 }]);
    })?;
    stats.render_ms = t1.elapsed().as_secs_f64() * 1e3;

    let raw = ctx.read(&readback)?;
    let data: &[f32] = unsafe { std::slice::from_raw_parts(raw.as_ptr() as *const f32, raw.len() / 4) };
    let mut rgb = Vec::with_capacity(pixels as usize * 3);
    for px in data.chunks_exact(4) {
        let n = px[3].max(1.0);
        rgb.extend_from_slice(&[px[0] / n, px[1] / n, px[2] / n]);
    }
    Ok((Image { width: w, height: h, rgb }, stats))
}
