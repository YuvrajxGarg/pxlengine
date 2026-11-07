//! Asset loading: glTF 2.0 scenes, LDR textures (RGBA8), HDR environments (RGB f32).

use anyhow::{bail, Context, Result};
use glam::Mat4;
use pxl_scene::MeshData;
use std::path::Path;

/// 8-bit RGBA image, row-major, top row first.
#[derive(Debug, Clone)]
pub struct RgbaImage {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u8>,
}

/// Linear RGB float image (equirectangular environments).
#[derive(Debug, Clone)]
pub struct HdrImage {
    pub width: u32,
    pub height: u32,
    pub rgb: Vec<f32>,
}

pub fn load_image(path: &Path) -> Result<RgbaImage> {
    let img = image::open(path).with_context(|| format!("loading image {}", path.display()))?.into_rgba8();
    Ok(RgbaImage { width: img.width(), height: img.height(), pixels: img.into_raw() })
}

pub fn load_hdr(path: &Path) -> Result<HdrImage> {
    let img = image::open(path).with_context(|| format!("loading HDRI {}", path.display()))?.into_rgb32f();
    Ok(HdrImage { width: img.width(), height: img.height(), rgb: img.into_raw() })
}

/// glTF metallic-roughness material, with texture references into `GltfAsset::images`.
#[derive(Debug, Clone)]
pub struct PbrMaterial {
    pub base_color: [f32; 4],
    pub base_color_tex: Option<usize>,
    pub metallic: f32,
    pub roughness: f32,
    /// glTF packs roughness in G and metalness in B.
    pub metallic_roughness_tex: Option<usize>,
    pub emissive: [f32; 3],
    pub emissive_strength: f32,
    pub emissive_tex: Option<usize>,
    pub normal_tex: Option<usize>,
    pub normal_scale: f32,
    pub alpha_mask: bool,
    pub alpha_cutoff: f32,
    /// KHR_materials_transmission / ior / volume.
    pub transmission: f32,
    pub ior: f32,
    pub attenuation_color: [f32; 3],
    pub attenuation_distance: f32,
}

#[derive(Debug, Clone)]
pub struct GltfPrimitive {
    pub mesh: MeshData,
    pub material: PbrMaterial,
}

#[derive(Debug, Clone)]
pub struct GltfAsset {
    pub primitives: Vec<GltfPrimitive>,
    pub images: Vec<RgbaImage>,
    /// (primitive index, asset-local transform) for every placed primitive.
    pub instances: Vec<(usize, Mat4)>,
    pub warnings: Vec<String>,
}

pub fn load_gltf(path: &Path) -> Result<GltfAsset> {
    let (doc, buffers, images) =
        gltf::import(path).with_context(|| format!("loading glTF {}", path.display()))?;
    let mut warnings = Vec::new();

    let images: Vec<RgbaImage> = images.iter().map(convert_image).collect::<Result<_>>()?;
    let tex_image = |t: gltf::Texture| t.source().index();

    let mut primitives = Vec::new();
    // mesh index -> list of primitive indices into `primitives`
    let mut mesh_prims: Vec<Vec<usize>> = Vec::new();
    for mesh in doc.meshes() {
        let mut list = Vec::new();
        for prim in mesh.primitives() {
            if prim.mode() != gltf::mesh::Mode::Triangles {
                warnings.push(format!("mesh {:?}: skipped non-triangle primitive ({:?})", mesh.name(), prim.mode()));
                continue;
            }
            let reader = prim.reader(|b| Some(&buffers[b.index()]));
            let Some(pos) = reader.read_positions() else {
                warnings.push(format!("mesh {:?}: primitive without positions", mesh.name()));
                continue;
            };
            let positions: Vec<[f32; 3]> = pos.collect();
            let n = positions.len();
            let indices: Vec<u32> = match reader.read_indices() {
                Some(i) => i.into_u32().collect(),
                None => (0..n as u32).collect(),
            };
            let uvs: Vec<[f32; 2]> = match reader.read_tex_coords(0) {
                Some(t) => t.into_f32().collect(),
                None => vec![[0.0, 0.0]; n],
            };
            let mut data = MeshData { positions, normals: Vec::new(), uvs, indices };
            match reader.read_normals() {
                Some(nr) => data.normals = nr.collect(),
                None => data.compute_normals(),
            }
            if data.indices.is_empty() {
                continue;
            }

            let m = prim.material();
            let pbr = m.pbr_metallic_roughness();
            let alpha_mask = !matches!(m.alpha_mode(), gltf::material::AlphaMode::Opaque);
            let material = PbrMaterial {
                base_color: pbr.base_color_factor(),
                base_color_tex: pbr.base_color_texture().map(|i| tex_image(i.texture())),
                metallic: pbr.metallic_factor(),
                roughness: pbr.roughness_factor(),
                metallic_roughness_tex: pbr.metallic_roughness_texture().map(|i| tex_image(i.texture())),
                emissive: m.emissive_factor(),
                emissive_strength: m.emissive_strength().unwrap_or(1.0),
                emissive_tex: m.emissive_texture().map(|i| tex_image(i.texture())),
                normal_tex: m.normal_texture().map(|i| tex_image(i.texture())),
                normal_scale: m.normal_texture().map(|i| i.scale()).unwrap_or(1.0),
                alpha_mask,
                alpha_cutoff: m.alpha_cutoff().unwrap_or(0.5),
                transmission: m.transmission().map(|t| t.transmission_factor()).unwrap_or(0.0),
                ior: m.ior().unwrap_or(1.5),
                attenuation_color: m.volume().map(|v| v.attenuation_color()).unwrap_or([1.0; 3]),
                attenuation_distance: m
                    .volume()
                    .map(|v| v.attenuation_distance())
                    .filter(|d| d.is_finite())
                    .unwrap_or(0.0),
            };
            if matches!(m.alpha_mode(), gltf::material::AlphaMode::Blend) {
                warnings.push(format!("material {:?}: alpha BLEND rendered as MASK", m.name()));
            }
            list.push(primitives.len());
            primitives.push(GltfPrimitive { mesh: data, material });
        }
        mesh_prims.push(list);
    }

    let mut instances = Vec::new();
    fn walk(node: gltf::Node, parent: Mat4, mesh_prims: &[Vec<usize>], out: &mut Vec<(usize, Mat4)>) {
        let m = parent * Mat4::from_cols_array_2d(&node.transform().matrix());
        if let Some(mesh) = node.mesh() {
            for &p in &mesh_prims[mesh.index()] {
                out.push((p, m));
            }
        }
        for c in node.children() {
            walk(c, m, mesh_prims, out);
        }
    }
    let Some(scene) = doc.default_scene().or_else(|| doc.scenes().next()) else {
        bail!("{} has no scenes", path.display());
    };
    for node in scene.nodes() {
        walk(node, Mat4::IDENTITY, &mesh_prims, &mut instances);
    }
    if instances.is_empty() {
        bail!("{} contains no renderable triangle meshes", path.display());
    }
    Ok(GltfAsset { primitives, images, instances, warnings })
}

fn convert_image(img: &gltf::image::Data) -> Result<RgbaImage> {
    use gltf::image::Format as F;
    let (channels, bytes_per) = match img.format {
        F::R8 => (1, 1),
        F::R8G8 => (2, 1),
        F::R8G8B8 => (3, 1),
        F::R8G8B8A8 => (4, 1),
        F::R16 => (1, 2),
        F::R16G16 => (2, 2),
        F::R16G16B16 => (3, 2),
        F::R16G16B16A16 => (4, 2),
        F::R32G32B32FLOAT => (3, 4),
        F::R32G32B32A32FLOAT => (4, 4),
    };
    let px = (img.width * img.height) as usize;
    let mut out = Vec::with_capacity(px * 4);
    let stride = channels * bytes_per;
    for i in 0..px {
        let texel = &img.pixels[i * stride..(i + 1) * stride];
        let ch = |c: usize| -> u8 {
            let b = &texel[c * bytes_per..(c + 1) * bytes_per];
            match bytes_per {
                1 => b[0],
                2 => b[1], // little-endian u16: high byte
                _ => (f32::from_le_bytes([b[0], b[1], b[2], b[3]]).clamp(0.0, 1.0) * 255.0 + 0.5) as u8,
            }
        };
        let rgba = match channels {
            1 => [ch(0), ch(0), ch(0), 255],
            2 => [ch(0), ch(1), 0, 255],
            3 => [ch(0), ch(1), ch(2), 255],
            _ => [ch(0), ch(1), ch(2), ch(3)],
        };
        out.extend_from_slice(&rgba);
    }
    Ok(RgbaImage { width: img.width, height: img.height, pixels: out })
}
