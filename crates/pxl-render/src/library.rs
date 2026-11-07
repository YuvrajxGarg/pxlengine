//! CPU-side scene library: unique meshes, textures, loaded glTF assets and the HDRI,
//! shared across all frames of a render.

use anyhow::{bail, Context as _, Result};
use glam::Mat4;
use pxl_assets::{GltfAsset, HdrImage, RgbaImage};
use pxl_gpu::Accel;
use pxl_scene::{geometry, MaterialInputs, MeshData, Node, Scene, World};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// 96 bytes, mirrors `Material` in pathtrace.slang. Integer slots hold i32 bit patterns.
pub type GpuMaterial = [f32; 24];

pub struct MeshEntry {
    pub data: MeshData,
    pub opaque: bool,
    pub vertex_offset: u32,
    pub index_offset: u32,
    pub blas: Option<Accel>,
}

#[derive(Clone)]
pub struct Inst {
    /// Index of the scene node this instance came from (object id pass).
    pub object: u32,
    pub mesh: usize,
    pub material: GpuMaterial,
    pub xform: Mat4,
}

pub struct Library {
    base_dir: PathBuf,
    pub meshes: Vec<MeshEntry>,
    mesh_keys: HashMap<String, usize>,
    pub textures: Vec<RgbaImage>,
    tex_keys: HashMap<String, i32>,
    assets: HashMap<PathBuf, Vec<(usize, GpuMaterial, Mat4)>>,
    pub env: Option<(PathBuf, HdrImage)>,
    pub warnings: Vec<String>,
    material_programs: std::collections::BTreeMap<String, u32>,
}

pub struct MatDesc {
    pub base: [f32; 3],
    pub base_alpha: f32,
    pub rough: f32,
    pub metal: f32,
    pub emission: [f32; 3],
    /// base colour, roughness, metalness, emission
    pub tex: [i32; 4],
    pub normal_tex: i32,
    pub rough_channel: i32,
    pub metal_channel: i32,
    pub alpha_mask: bool,
    pub normal_scale: f32,
    pub alpha_cutoff: f32,
    pub uv_scale: [f32; 2],
    /// Scene material program id (0 = none).
    pub program: u32,
}

pub fn encode_material(m: &MatDesc) -> GpuMaterial {
    let i = |v: i32| f32::from_bits(v as u32);
    [
        m.base[0], m.base[1], m.base[2], m.rough,
        m.emission[0], m.emission[1], m.emission[2], m.metal,
        i(m.tex[0]), i(m.tex[1]), i(m.tex[2]), i(m.tex[3]),
        i(m.normal_tex), i(m.rough_channel), i(m.metal_channel), i(m.alpha_mask as i32),
        m.normal_scale, m.alpha_cutoff, m.uv_scale[0], m.uv_scale[1],
        m.base_alpha, f32::from_bits(m.program), 0.0, 0.0,
    ]
}

pub fn material_emission(m: &GpuMaterial) -> [f32; 3] {
    [m[4], m[5], m[6]]
}

/// Texture index in slot 0..4 (base colour, roughness, metalness, emission).
pub fn material_texture(m: &GpuMaterial, slot: usize) -> Option<usize> {
    let i = m[8 + slot].to_bits() as i32;
    (i >= 0).then_some(i as usize)
}

pub fn material_uv_scale(m: &GpuMaterial) -> [f32; 2] {
    [m[18], m[19]]
}

fn srgb_to_linear(c: u8) -> f32 {
    let c = c as f32 / 255.0;
    if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) }
}

/// Mean linear luminance of an sRGB texture over a triangle, from 10 sample points
/// (vertices, edge midpoints, interior), nearest-texel.
pub fn triangle_texture_luminance(img: &RgbaImage, uv: [[f32; 2]; 3], scale: [f32; 2]) -> f32 {
    const B: [[f32; 3]; 10] = [
        [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
        [0.5, 0.5, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5],
        [0.333, 0.333, 0.334], [0.66, 0.17, 0.17], [0.17, 0.66, 0.17], [0.17, 0.17, 0.66],
    ];
    let (w, h) = (img.width as i64, img.height as i64);
    let mut sum = 0.0;
    for b in B {
        let u = (b[0] * uv[0][0] + b[1] * uv[1][0] + b[2] * uv[2][0]) * scale[0];
        let v = (b[0] * uv[0][1] + b[1] * uv[1][1] + b[2] * uv[2][1]) * scale[1];
        let x = ((u * w as f32).floor() as i64).rem_euclid(w) as usize;
        let y = ((v * h as f32).floor() as i64).rem_euclid(h) as usize;
        let p = &img.pixels[(y * w as usize + x) * 4..][..3];
        sum += 0.2126 * srgb_to_linear(p[0]) + 0.7152 * srgb_to_linear(p[1]) + 0.0722 * srgb_to_linear(p[2]);
    }
    sum / B.len() as f32
}

impl Library {
    pub fn new(base_dir: &Path, material_programs: std::collections::BTreeMap<String, u32>) -> Library {
        Library {
            base_dir: base_dir.to_path_buf(),
            meshes: Vec::new(),
            mesh_keys: HashMap::new(),
            textures: Vec::new(),
            tex_keys: HashMap::new(),
            assets: HashMap::new(),
            env: None,
            warnings: Vec::new(),
            material_programs,
        }
    }

    fn resolve(&self, file: &str) -> PathBuf {
        let p = Path::new(file);
        if p.is_absolute() { p.to_path_buf() } else { self.base_dir.join(p) }
    }

    fn add_mesh(&mut self, key: String, data: MeshData, opaque: bool) -> usize {
        if let Some(&i) = self.mesh_keys.get(&key) {
            return i;
        }
        self.meshes.push(MeshEntry { data, opaque, vertex_offset: 0, index_offset: 0, blas: None });
        self.mesh_keys.insert(key, self.meshes.len() - 1);
        self.meshes.len() - 1
    }

    fn texture_file(&mut self, file: &str) -> Result<i32> {
        let path = self.resolve(file);
        let key = path.to_string_lossy().into_owned();
        if let Some(&i) = self.tex_keys.get(&key) {
            return Ok(i);
        }
        let img = pxl_assets::load_image(&path)?;
        self.textures.push(img);
        let i = (self.textures.len() - 1) as i32;
        self.tex_keys.insert(key, i);
        Ok(i)
    }

    fn material_from_inputs(&mut self, m: &MaterialInputs, program: Option<&str>) -> Result<GpuMaterial> {
        let program = match program {
            Some(name) => *self
                .material_programs
                .get(name)
                .with_context(|| format!("material program '{name}' is not known to the compiled programs"))?,
            None => 0,
        };
        let mut tex = |f: &Option<String>| -> Result<i32> {
            match f {
                Some(f) => self.texture_file(f),
                None => Ok(-1),
            }
        };
        let t = [
            tex(&m.base_color_texture)?,
            tex(&m.specular_roughness_texture)?,
            tex(&m.base_metalness_texture)?,
            tex(&m.emission_color_texture)?,
        ];
        let normal_tex = tex(&m.normal_texture)?;
        let e = m.emission_luminance;
        Ok(encode_material(&MatDesc {
            base: m.base_color,
            base_alpha: 1.0,
            rough: m.specular_roughness,
            metal: m.base_metalness,
            emission: m.emission_color.map(|c| c * e),
            tex: t,
            normal_tex,
            rough_channel: 0,
            metal_channel: 0,
            alpha_mask: false,
            normal_scale: m.normal_scale,
            alpha_cutoff: 0.5,
            uv_scale: m.uv_scale,
            program,
        }))
    }

    fn asset(&mut self, file: &str) -> Result<Vec<(usize, GpuMaterial, Mat4)>> {
        let path = self.resolve(file);
        let path = path.canonicalize().with_context(|| format!("asset file {} not found", path.display()))?;
        if let Some(parts) = self.assets.get(&path) {
            return Ok(parts.clone());
        }
        let GltfAsset { primitives, images, instances, warnings } = pxl_assets::load_gltf(&path)?;
        let name = path.file_name().unwrap_or_default().to_string_lossy().into_owned();
        self.warnings.extend(warnings.into_iter().map(|w| format!("{name}: {w}")));
        // Asset image -> global texture index, assigned on first use.
        let mut image_map: HashMap<usize, i32> = HashMap::new();
        let mut images: Vec<Option<RgbaImage>> = images.into_iter().map(Some).collect();
        let mut tex = |slot: Option<usize>, lib: &mut Library| -> i32 {
            let Some(i) = slot else { return -1 };
            if let Some(&g) = image_map.get(&i) {
                return g;
            }
            let Some(img) = images.get_mut(i).and_then(Option::take) else { return -1 };
            lib.textures.push(img);
            let g = (lib.textures.len() - 1) as i32;
            image_map.insert(i, g);
            g
        };
        let mut prim_info = Vec::new();
        for (pi, prim) in primitives.into_iter().enumerate() {
            let m = &prim.material;
            let mr = tex(m.metallic_roughness_tex, self);
            let desc = MatDesc {
                base: [m.base_color[0], m.base_color[1], m.base_color[2]],
                base_alpha: m.base_color[3],
                rough: m.roughness,
                metal: m.metallic,
                emission: m.emissive.map(|c| c * m.emissive_strength),
                tex: [tex(m.base_color_tex, self), mr, mr, tex(m.emissive_tex, self)],
                normal_tex: tex(m.normal_tex, self),
                rough_channel: 1,
                metal_channel: 2,
                alpha_mask: m.alpha_mask,
                normal_scale: m.normal_scale,
                alpha_cutoff: m.alpha_cutoff,
                uv_scale: [1.0, 1.0],
                program: 0,
            };
            let opaque = !m.alpha_mask;
            let mesh = self.add_mesh(format!("{}#p{pi}", path.display()), prim.mesh, opaque);
            prim_info.push((mesh, encode_material(&desc)));
        }
        let parts: Vec<_> = instances.into_iter().map(|(p, xf)| (prim_info[p].0, prim_info[p].1, xf)).collect();
        self.assets.insert(path, parts.clone());
        Ok(parts)
    }

    /// Mesh + material drawn for every agent of a simulation.
    pub fn sim_mesh(&mut self, sim: &pxl_scene::Simulation) -> Result<(usize, GpuMaterial)> {
        if let Some(g) = &sim.geometry {
            let key = format!("{g:?}");
            let mesh = match self.mesh_keys.get(&key) {
                Some(&i) => i,
                None => self.add_mesh(key, geometry::build(g), true),
            };
            let first = sim.materials.first().cloned().unwrap_or_default();
            let material = self.material_from_inputs(&first.inputs, first.program.as_deref())?;
            return Ok((mesh, material));
        }
        let file = sim.asset.as_deref().expect("validated: geometry or asset");
        let parts = self.asset(file)?;
        let (mesh, material, _) = parts.first().copied().context("asset has no meshes")?;
        Ok((mesh, material))
    }

    /// Resolve every renderable node of `scene` into instances, loading what is new.
    pub fn instances(&mut self, scene: &Scene) -> Result<Vec<Inst>> {
        let mut out = Vec::new();
        for (object, node) in scene.nodes.iter().enumerate() {
            let object = object as u32;
            match node {
                Node::Mesh(m) => {
                    let key = format!("{:?}", m.geometry);
                    let mesh = match self.mesh_keys.get(&key) {
                        Some(&i) => i,
                        None => self.add_mesh(key, geometry::build(&m.geometry), true),
                    };
                    let first = m.materials.first().cloned().unwrap_or_default();
                    let material = self.material_from_inputs(&first.inputs, first.program.as_deref()).with_context(|| {
                        format!("material of mesh '{}'", m.name.as_deref().unwrap_or("unnamed"))
                    })?;
                    out.push(Inst { object, mesh, material, xform: m.transform() });
                }
                Node::Asset(a) => {
                    let base = a.transform();
                    for (mesh, material, local) in self.asset(&a.file)? {
                        out.push(Inst { object, mesh, material, xform: base * local });
                    }
                }
                Node::Light(_) | Node::Camera(_) => {}
            }
        }
        Ok(out)
    }

    pub fn load_env(&mut self, world: &World) -> Result<()> {
        let Some(file) = &world.hdri else { return Ok(()) };
        let path = self.resolve(file);
        match &self.env {
            Some((p, _)) if *p == path => Ok(()),
            Some((p, _)) => bail!("world.hdri cannot change during a render ({} vs {})", p.display(), path.display()),
            None => {
                let img = pxl_assets::load_hdr(&path)?;
                self.env = Some((path, img));
                Ok(())
            }
        }
    }
}

/// Importance-sampling tables for an equirectangular map: (marginal, conditional, total weight).
pub fn env_cdf(img: &HdrImage) -> (Vec<f32>, Vec<f32>, f32) {
    let (w, h) = (img.width as usize, img.height as usize);
    let mut cond = vec![0f32; h * (w + 1)];
    let mut marg = vec![0f32; h + 1];
    for y in 0..h {
        let s = (std::f32::consts::PI * (y as f32 + 0.5) / h as f32).sin();
        let row = &mut cond[y * (w + 1)..(y + 1) * (w + 1)];
        let mut acc = 0f64;
        for x in 0..w {
            let p = &img.rgb[(y * w + x) * 3..(y * w + x) * 3 + 3];
            let l = (0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2]).max(0.0) * s;
            acc += l as f64;
            row[x + 1] = acc as f32;
        }
        for x in 0..=w {
            row[x] = if acc > 0.0 { (row[x] as f64 / acc) as f32 } else { x as f32 / w as f32 };
        }
        marg[y + 1] = marg[y] + acc as f32;
    }
    let total = marg[h];
    for v in &mut marg {
        *v = if total > 0.0 { *v / total } else { 0.0 };
    }
    marg[h] = 1.0;
    (marg, cond, total)
}
