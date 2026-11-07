//! Scene description for pxlengine: strict JSON (unknown fields are errors, so agents
//! learn about typos immediately) plus built-in ("std") geometry generators.

pub mod geometry;

use anyhow::{bail, Context, Result};
use glam::{EulerRot, Mat4, Quat, Vec3};
use serde::Deserialize;

pub use geometry::MeshData;

pub const SCENE_VERSION: u32 = 1;

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Scene {
    pub version: u32,
    #[serde(default)]
    pub nodes: Vec<Node>,
    #[serde(default)]
    pub world: World,
    #[serde(default)]
    pub render: RenderSettings,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Node {
    Mesh(MeshNode),
    Light(LightNode),
    Camera(CameraNode),
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MeshNode {
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub translation: [f32; 3],
    /// Euler XYZ, degrees.
    #[serde(default)]
    pub rotation: [f32; 3],
    #[serde(default = "one3")]
    pub scale: [f32; 3],
    pub geometry: Geometry,
    #[serde(default)]
    pub materials: Vec<Material>,
}

impl MeshNode {
    pub fn transform(&self) -> Mat4 {
        let r = self.rotation.map(f32::to_radians);
        Mat4::from_scale_rotation_translation(
            Vec3::from(self.scale),
            Quat::from_euler(EulerRot::XYZ, r[0], r[1], r[2]),
            Vec3::from(self.translation),
        )
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "std", rename_all = "lowercase", deny_unknown_fields)]
pub enum Geometry {
    Sphere {
        #[serde(default = "one")]
        radius: f32,
        #[serde(default = "seg64")]
        segments: u32,
    },
    #[serde(alias = "plane")]
    Grid {
        #[serde(default = "one2")]
        size: [f32; 2],
    },
    Box {
        #[serde(default = "one3")]
        size: [f32; 3],
    },
    Cylinder {
        #[serde(default = "one")]
        radius: f32,
        #[serde(default = "two")]
        height: f32,
        #[serde(default = "seg64")]
        segments: u32,
    },
    Torus {
        #[serde(default = "one")]
        radius: f32,
        #[serde(default = "quarter")]
        tube: f32,
        #[serde(default = "seg64")]
        segments: u32,
    },
}

/// OpenPBR-style input names (the same vocabulary MaterialX uses).
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Material {
    #[serde(default)]
    pub inputs: MaterialInputs,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MaterialInputs {
    #[serde(default = "grey")]
    pub base_color: [f32; 3],
    #[serde(default = "half")]
    pub specular_roughness: f32,
    #[serde(default)]
    pub base_metalness: f32,
    #[serde(default = "one3")]
    pub emission_color: [f32; 3],
    /// Emitted radiance scale (W/sr/m²); 0 = not emissive.
    #[serde(default)]
    pub emission_luminance: f32,
}

impl Default for MaterialInputs {
    fn default() -> Self {
        serde_json::from_str("{}").unwrap()
    }
}

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum LightShape {
    Point,
    Rectangle,
    Directional,
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(untagged)]
pub enum Size2 {
    Uniform(f32),
    Xy([f32; 2]),
}

impl Size2 {
    pub fn xy(self) -> [f32; 2] {
        match self {
            Size2::Uniform(s) => [s, s],
            Size2::Xy(v) => v,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LightNode {
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default = "point_shape")]
    pub shape: LightShape,
    #[serde(default)]
    pub translation: [f32; 3],
    /// Point the light (rectangle / directional) at this world position.
    #[serde(default)]
    pub target: [f32; 3],
    /// Watts for point and rectangle lights; irradiance (W/m²) for directional.
    #[serde(default = "hundred")]
    pub energy: f32,
    #[serde(default = "one3")]
    pub color: [f32; 3],
    /// Rectangle size in metres: number or [w, h].
    #[serde(default = "size_one")]
    pub size: Size2,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CameraNode {
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default = "cam_pos")]
    pub translation: [f32; 3],
    #[serde(default)]
    pub target: [f32; 3],
    /// Vertical field of view, degrees.
    #[serde(default = "fov")]
    pub fov: f32,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct World {
    /// Uniform environment radiance seen by rays that escape the scene.
    #[serde(default = "world_color")]
    pub color: [f32; 3],
    #[serde(default = "one")]
    pub strength: f32,
}

impl Default for World {
    fn default() -> Self {
        serde_json::from_str("{}").unwrap()
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RenderSettings {
    #[serde(default = "res")]
    pub resolution: [u32; 2],
    #[serde(default)]
    pub samples: Option<u32>,
    #[serde(default)]
    pub bounces: Option<u32>,
    /// Exposure in stops.
    #[serde(default)]
    pub exposure: f32,
    #[serde(default)]
    pub seed: u32,
}

impl Default for RenderSettings {
    fn default() -> Self {
        serde_json::from_str("{}").unwrap()
    }
}

impl Scene {
    pub fn from_json(text: &str) -> Result<Scene> {
        let de = &mut serde_json::Deserializer::from_str(text);
        let scene: Scene = serde_path_to_error(de)?;
        if scene.version != SCENE_VERSION {
            bail!("scene version {} is not supported (expected {SCENE_VERSION})", scene.version);
        }
        let cams = scene.nodes.iter().filter(|n| matches!(n, Node::Camera(_))).count();
        if cams == 0 {
            bail!("scene has no camera node; add {{\"type\":\"camera\",\"translation\":[x,y,z],\"target\":[0,0,0]}}");
        }
        let [w, h] = scene.render.resolution;
        if w == 0 || h == 0 || w > 16384 || h > 16384 {
            bail!("render.resolution must be within 1..16384, got [{w},{h}]");
        }
        Ok(scene)
    }

    pub fn load(path: &std::path::Path) -> Result<Scene> {
        let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        Scene::from_json(&text).with_context(|| format!("in {}", path.display()))
    }

    pub fn camera(&self) -> &CameraNode {
        self.nodes
            .iter()
            .find_map(|n| match n {
                Node::Camera(c) => Some(c),
                _ => None,
            })
            .expect("validated")
    }
}

/// serde_json errors already carry line/column; this keeps the message agent-readable.
fn serde_path_to_error(de: &mut serde_json::Deserializer<serde_json::de::StrRead>) -> Result<Scene> {
    Scene::deserialize(de).map_err(|e| anyhow::anyhow!("invalid scene: {e}"))
}

fn one() -> f32 { 1.0 }
fn two() -> f32 { 2.0 }
fn half() -> f32 { 0.5 }
fn quarter() -> f32 { 0.25 }
fn hundred() -> f32 { 100.0 }
fn fov() -> f32 { 40.0 }
fn seg64() -> u32 { 64 }
fn one2() -> [f32; 2] { [1.0, 1.0] }
fn one3() -> [f32; 3] { [1.0, 1.0, 1.0] }
fn grey() -> [f32; 3] { [0.8, 0.8, 0.8] }
fn cam_pos() -> [f32; 3] { [0.0, 1.0, 5.0] }
fn world_color() -> [f32; 3] { [0.05, 0.055, 0.065] }
fn res() -> [u32; 2] { [640, 480] }
fn point_shape() -> LightShape { LightShape::Point }
fn size_one() -> Size2 { Size2::Uniform(1.0) }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_first_scene() {
        let s = Scene::from_json(include_str!("../../../examples/first-scene.json")).unwrap();
        assert_eq!(s.nodes.len(), 4);
    }

    #[test]
    fn rejects_typos() {
        let e = Scene::from_json(r#"{"version":1,"nodes":[{"type":"camera","fvo":30}]}"#).unwrap_err();
        assert!(format!("{e:#}").contains("fvo"), "{e:#}");
    }
}
