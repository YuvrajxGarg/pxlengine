//! Scene description for pxlengine: strict JSON (unknown fields are errors, so agents
//! learn about typos immediately), std geometry generators, and animation tracks.

pub mod anim;
pub mod geometry;

use anyhow::{bail, Context, Result};
use glam::{EulerRot, Mat4, Quat, Vec3};
use serde::Deserialize;
use serde_json::Value;

pub use anim::{Animation, Interp, Orbit, Track};
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
    #[serde(default)]
    pub animation: Option<Animation>,
    #[serde(default)]
    pub programs: Option<Programs>,
    #[serde(default)]
    pub simulations: Vec<Simulation>,
    /// The JSON this scene was parsed from; animation tracks are applied to it per frame.
    #[serde(skip)]
    raw: Value,
    /// Frame this scene was evaluated at (0 for a static scene).
    #[serde(skip)]
    pub frame: f64,
}

/// A population of GPU agents stepped by scene programs and drawn as instances of one mesh.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Simulation {
    pub name: String,
    /// Number of agents (1 .. 16M).
    pub count: u32,
    /// `void f(uint id, inout Agent a, inout uint rng)`, run once.
    pub init: String,
    /// `void f(uint id, inout Agent a, inout uint rng)`, run every substep (`SIM.dt`).
    pub step: String,
    /// Neighbour grid cell size (≈ interaction radius). 0 = no neighbour queries.
    #[serde(default)]
    pub cell_size: f32,
    #[serde(default = "one_u32")]
    pub substeps: u32,
    /// Seconds simulated before time 0, so the system has settled when the shot starts.
    #[serde(default)]
    pub warmup: f32,
    /// Mesh drawn for every agent (+Z = forward, +Y = up), or `asset` (first mesh of a glTF).
    #[serde(default)]
    pub geometry: Option<Geometry>,
    #[serde(default)]
    pub asset: Option<String>,
    #[serde(default)]
    pub materials: Vec<Material>,
    /// Mesh scale (multiplied by each agent's `size`).
    #[serde(default = "one")]
    pub scale: f32,
    /// Turn each instance to face along its velocity.
    #[serde(default = "yes")]
    pub orient: bool,
}

fn one_u32() -> u32 {
    1
}
fn yes() -> bool {
    true
}

/// Scene-supplied Slang code hooked into the renderer. See `pxl help programs`.
#[derive(Debug, Clone, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct Programs {
    /// Slang file, relative to the scene file.
    #[serde(default)]
    pub source: Option<String>,
    /// Inline Slang code (appended after `source`).
    #[serde(default)]
    pub code: Option<String>,
    /// `bool f(CameraIn c, inout float3 origin, inout float3 dir)`
    #[serde(default)]
    pub camera: Option<String>,
    /// `Hit f(inout Ray r, inout float3 radiance, inout float3 throughput, inout uint rng)`
    #[serde(default)]
    pub propagate: Option<String>,
    /// `float3 f(FilmIn i)`
    #[serde(default)]
    pub film: Option<String>,
    /// Values exposed to programs as `P.<name>`: number -> float, [a,b] -> float2, ... float4.
    /// Animatable with tracks on path `programs/params/<name>`.
    #[serde(default)]
    pub params: std::collections::BTreeMap<String, Value>,
}

impl Programs {
    /// Number of float components of a param value (1..4), or an error.
    pub fn param_width(name: &str, v: &Value) -> Result<usize> {
        let n = match v {
            Value::Number(_) => 1,
            Value::Array(a) if (2..=4).contains(&a.len()) && a.iter().all(Value::is_number) => a.len(),
            _ => bail!("programs.params.{name} must be a number or an array of 2-4 numbers"),
        };
        Ok(n)
    }

    pub fn param_values(v: &Value) -> [f32; 4] {
        let mut out = [0f32; 4];
        match v {
            Value::Number(n) => out[0] = n.as_f64().unwrap_or(0.0) as f32,
            Value::Array(a) => {
                for (i, x) in a.iter().take(4).enumerate() {
                    out[i] = x.as_f64().unwrap_or(0.0) as f32;
                }
            }
            _ => {}
        }
        out
    }

    fn validate(&self, nodes: &[Node], sims_use_programs: bool) -> Result<()> {
        for (name, v) in &self.params {
            if !name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') || name.starts_with(|c: char| c.is_ascii_digit()) {
                bail!("programs.params: '{name}' must be a valid identifier (letters, digits, _)");
            }
            Programs::param_width(name, v)?;
        }
        for (field, f) in [("camera", &self.camera), ("propagate", &self.propagate), ("film", &self.film)] {
            if let Some(f) = f {
                if !is_ident(f) {
                    bail!("programs.{field}: '{f}' must be the name of a Slang function");
                }
            }
        }
        let uses_materials = nodes.iter().any(|n| matches!(n, Node::Mesh(m) if m.materials.iter().any(|m| m.program.is_some())));
        let uses_materials = uses_materials || sims_use_programs;
        let hooks = self.camera.is_some() || self.propagate.is_some() || self.film.is_some() || uses_materials;
        if hooks && self.source.is_none() && self.code.is_none() {
            bail!("programs: hooks are named but there is no `source` file or inline `code`");
        }
        Ok(())
    }
}

fn is_ident(s: &str) -> bool {
    !s.is_empty() && !s.starts_with(|c: char| c.is_ascii_digit()) && s.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Node {
    Mesh(MeshNode),
    Asset(AssetNode),
    Light(LightNode),
    Camera(CameraNode),
}

pub fn trs(t: [f32; 3], r_deg: [f32; 3], s: [f32; 3]) -> Mat4 {
    let r = r_deg.map(f32::to_radians);
    Mat4::from_scale_rotation_translation(Vec3::from(s), Quat::from_euler(EulerRot::XYZ, r[0], r[1], r[2]), Vec3::from(t))
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
        trs(self.translation, self.rotation, self.scale)
    }
}

/// A glTF 2.0 file (.glb or .gltf) placed in the scene.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AssetNode {
    #[serde(default)]
    pub name: Option<String>,
    /// Path relative to the scene file.
    pub file: String,
    #[serde(default)]
    pub translation: [f32; 3],
    #[serde(default)]
    pub rotation: [f32; 3],
    #[serde(default = "one3")]
    pub scale: [f32; 3],
}

impl AssetNode {
    pub fn transform(&self) -> Mat4 {
        trs(self.translation, self.rotation, self.scale)
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
    /// Low-poly bird (wingspan 1, +Z forward, +Y up), cheap enough to instance millions.
    Bird {},
}

/// OpenPBR-style input names (the same vocabulary MaterialX uses).
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Material {
    #[serde(default)]
    pub inputs: MaterialInputs,
    /// Name of a material function in `programs` that edits the surface after inputs/textures.
    #[serde(default)]
    pub program: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MaterialInputs {
    #[serde(default = "grey")]
    pub base_color: [f32; 3],
    /// sRGB image multiplied with base_color.
    #[serde(default)]
    pub base_color_texture: Option<String>,
    #[serde(default = "half")]
    pub specular_roughness: f32,
    /// Greyscale (red channel) image multiplied with specular_roughness.
    #[serde(default)]
    pub specular_roughness_texture: Option<String>,
    #[serde(default)]
    pub base_metalness: f32,
    #[serde(default)]
    pub base_metalness_texture: Option<String>,
    #[serde(default = "one3")]
    pub emission_color: [f32; 3],
    #[serde(default)]
    pub emission_color_texture: Option<String>,
    /// Emitted radiance scale (W/sr/m²); 0 = not emissive.
    #[serde(default)]
    pub emission_luminance: f32,
    /// Tangent-space normal map (OpenGL / +Y convention, as glTF and Poly Haven "nor_gl").
    #[serde(default)]
    pub normal_texture: Option<String>,
    #[serde(default = "one")]
    pub normal_scale: f32,
    /// Texture tiling.
    #[serde(default = "one2")]
    pub uv_scale: [f32; 2],
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
    /// Lens f-number for depth of field (e.g. 1.4 = very shallow, 16 = deep). Omit for a pinhole.
    #[serde(default)]
    pub fstop: Option<f32>,
    /// Distance to the plane in focus, metres. Defaults to the distance to `focus_target`,
    /// else to `target`.
    #[serde(default)]
    pub focus_distance: Option<f32>,
    #[serde(default)]
    pub focus_target: Option<[f32; 3]>,
}

/// Full-frame sensor height (36x24 mm) used to turn fov + fstop into a physical aperture.
pub const SENSOR_HEIGHT_M: f32 = 0.024;

impl CameraNode {
    /// (aperture radius m, focus distance m); radius 0 = pinhole.
    pub fn lens(&self) -> (f32, f32) {
        let eye = Vec3::from(self.translation);
        let fwd = (Vec3::from(self.target) - eye).try_normalize().unwrap_or(Vec3::NEG_Z);
        let focus = self.focus_distance.unwrap_or_else(|| {
            let t = Vec3::from(self.focus_target.unwrap_or(self.target));
            (t - eye).dot(fwd).max(1e-3)
        });
        match self.fstop {
            Some(n) if n > 0.0 => {
                let focal = SENSOR_HEIGHT_M * 0.5 / (self.fov.to_radians() * 0.5).tan();
                (focal / (2.0 * n), focus)
            }
            _ => (0.0, focus),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct World {
    /// Uniform environment radiance (used when no `hdri`).
    #[serde(default = "world_color")]
    pub color: [f32; 3],
    #[serde(default = "one")]
    pub strength: f32,
    /// Equirectangular .hdr / .exr environment, path relative to the scene file.
    #[serde(default)]
    pub hdri: Option<String>,
    /// Rotate the HDRI around +Y, degrees.
    #[serde(default)]
    pub rotation: f32,
}

impl Default for World {
    fn default() -> Self {
        serde_json::from_str("{}").unwrap()
    }
}

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub enum View {
    /// Blender-style AgX filmic curve (default): soft highlights, no hue skew.
    #[default]
    Agx,
    Aces,
    /// Plain sRGB clip.
    Standard,
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
    pub view: View,
    #[serde(default)]
    pub seed: u32,
    #[serde(default = "fps")]
    pub fps: f32,
    /// Shutter open time as a fraction of a frame (0.5 = 180° shutter). Omit for no motion blur.
    #[serde(default)]
    pub motion_blur: Option<f32>,
    /// Run the AI denoiser (default: per profile).
    #[serde(default)]
    pub denoise: Option<bool>,
    /// Transparent background in PNG output (alpha = geometry coverage).
    #[serde(default)]
    pub transparent: bool,
}

impl Default for RenderSettings {
    fn default() -> Self {
        serde_json::from_str("{}").unwrap()
    }
}

impl Scene {
    pub fn from_json(text: &str) -> Result<Scene> {
        let de = &mut serde_json::Deserializer::from_str(text);
        let mut scene = Scene::deserialize(de).map_err(|e| anyhow::anyhow!("invalid scene: {e}"))?;
        scene.raw = serde_json::from_str(text)?;
        scene.validate()?;
        if let Some(anim) = &scene.animation {
            anim.validate(&scene.raw)?;
            // Evaluate once so bad paths / values fail before any GPU work.
            scene.at_frame(anim.frame_start as f64)?;
        }
        Ok(scene)
    }

    fn validate(&self) -> Result<()> {
        if self.version != SCENE_VERSION {
            bail!("scene version {} is not supported (expected {SCENE_VERSION})", self.version);
        }
        if !self.nodes.iter().any(|n| matches!(n, Node::Camera(_))) {
            bail!("scene has no camera node; add {{\"type\":\"camera\",\"translation\":[x,y,z],\"target\":[0,0,0]}}");
        }
        let [w, h] = self.render.resolution;
        if w == 0 || h == 0 || w > 16384 || h > 16384 {
            bail!("render.resolution must be within 1..16384, got [{w},{h}]");
        }
        if !(self.render.fps > 0.0) {
            bail!("render.fps must be > 0");
        }
        for n in &self.nodes {
            if let Node::Mesh(m) = n {
                for mat in &m.materials {
                    if let Some(p) = &mat.program {
                        if !is_ident(p) {
                            bail!("material program '{p}' must be the name of a Slang function");
                        }
                        if self.programs.is_none() {
                            bail!("a material uses program '{p}' but the scene has no `programs` block");
                        }
                    }
                }
            }
        }
        for (i, sim) in self.simulations.iter().enumerate() {
            let ctx = |m: String| anyhow::anyhow!("simulations[{i}] '{}': {m}", sim.name);
            if !(1..=16_000_000).contains(&sim.count) {
                return Err(ctx(format!("count must be 1..16000000, got {}", sim.count)));
            }
            for (field, f) in [("init", &sim.init), ("step", &sim.step)] {
                if !is_ident(f) {
                    return Err(ctx(format!("{field}: '{f}' must be the name of a Slang function")));
                }
            }
            if sim.geometry.is_some() == sim.asset.is_some() {
                return Err(ctx("give exactly one of `geometry` or `asset` for the agent mesh".into()));
            }
            if !(sim.cell_size >= 0.0) || sim.substeps == 0 || sim.substeps > 64 || !(sim.warmup >= 0.0) {
                return Err(ctx("cell_size >= 0, substeps 1..64, warmup >= 0".into()));
            }
            if self.programs.is_none() {
                return Err(ctx("simulations need a `programs` block with the init/step functions".into()));
            }
            if self.simulations[..i].iter().any(|o| o.name == sim.name) {
                return Err(ctx("duplicate simulation name".into()));
            }
        }
        if let Some(p) = &self.programs {
            p.validate(&self.nodes, !self.simulations.is_empty())?;
        }
        if let Some(m) = self.render.motion_blur {
            if !(0.0..=1.0).contains(&m) {
                bail!("render.motion_blur is the shutter fraction of a frame, 0..1 (0.5 = 180°), got {m}");
            }
        }
        Ok(())
    }

    pub fn load(path: &std::path::Path) -> Result<Scene> {
        let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        Scene::from_json(&text).with_context(|| format!("in {}", path.display()))
    }

    /// The scene with all animation tracks evaluated at `frame`.
    pub fn at_frame(&self, frame: f64) -> Result<Scene> {
        let Some(anim) = &self.animation else {
            let mut s = self.clone();
            s.frame = frame;
            return Ok(s);
        };
        let mut raw = self.raw.clone();
        for (i, track) in anim.tracks.iter().enumerate() {
            let v = track.eval(frame).with_context(|| format!("animation.tracks[{i}]"))?;
            anim::apply(&mut raw, track, v).with_context(|| format!("animation.tracks[{i}]"))?;
        }
        let mut s: Scene = serde_json::from_value(raw.clone())
            .map_err(|e| anyhow::anyhow!("scene invalid at frame {frame} after animation: {e}"))?;
        s.raw = raw;
        s.frame = frame;
        s.validate()?;
        if let (Some(a), Some(b)) = (&self.programs, &s.programs) {
            for (k, v) in &a.params {
                let same = b.params.get(k).map(|w| Programs::param_width(k, w).ok() == Programs::param_width(k, v).ok());
                if same != Some(true) {
                    bail!("programs.params.{k}: animation may change values but not remove a param or its size");
                }
            }
        }
        Ok(s)
    }

    /// Inclusive frame range of the animation, if any.
    pub fn frame_range(&self) -> Option<(i32, i32)> {
        self.animation.as_ref().map(|a| (a.frame_start, a.end()))
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

fn one() -> f32 { 1.0 }
fn two() -> f32 { 2.0 }
fn half() -> f32 { 0.5 }
fn quarter() -> f32 { 0.25 }
fn hundred() -> f32 { 100.0 }
fn fov() -> f32 { 40.0 }
fn fps() -> f32 { 24.0 }
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

    #[test]
    fn animates_fields() {
        let s = Scene::from_json(
            r#"{"version":1,"nodes":[
                {"type":"camera","name":"cam","translation":[0,1,5]},
                {"type":"mesh","name":"ball","geometry":{"std":"sphere"}}],
              "animation":{"frame_end":10,"tracks":[
                {"node":"cam","path":"translation","keys":[[0,[0,1,5]],[10,[10,1,5]]]},
                {"node":"ball","path":"materials/0/inputs/emission_luminance","keys":[[0,0],[10,20]]},
                {"path":"render/exposure","keys":[[0,0],[10,2]],"interp":"step"}]}}"#,
        )
        .unwrap();
        let f = s.at_frame(5.0).unwrap();
        assert!((f.camera().translation[0] - 5.0).abs() < 1e-5);
        let Node::Mesh(m) = &f.nodes[1] else { panic!() };
        assert!((m.materials[0].inputs.emission_luminance - 10.0).abs() < 1e-5);
        assert_eq!(f.render.exposure, 0.0);
        assert_eq!(s.at_frame(10.0).unwrap().render.exposure, 2.0);
        assert_eq!(s.frame_range(), Some((0, 10)));
    }

    #[test]
    fn rejects_unknown_track_node() {
        let e = Scene::from_json(
            r#"{"version":1,"nodes":[{"type":"camera"}],
              "animation":{"tracks":[{"node":"nope","path":"translation","keys":[[0,[0,0,0]]]}]}}"#,
        )
        .unwrap_err();
        assert!(format!("{e:#}").contains("nope"), "{e:#}");
    }

    #[test]
    fn simulations_validate() {
        let ok = r#"{"version":1,"programs":{"code":"//"},"nodes":[{"type":"camera"}],
            "simulations":[{"name":"f","count":10,"init":"i","step":"s","geometry":{"std":"bird"}}]}"#;
        let s = Scene::from_json(ok).unwrap();
        assert_eq!(s.simulations[0].substeps, 1);
        assert!(geometry::build(s.simulations[0].geometry.as_ref().unwrap()).triangle_count() > 8);
        let no_mesh = ok.replace(r#","geometry":{"std":"bird"}"#, "");
        assert!(format!("{:#}", Scene::from_json(&no_mesh).unwrap_err()).contains("geometry"));
        let no_programs = ok.replace(r#""programs":{"code":"//"},"#, "");
        assert!(format!("{:#}", Scene::from_json(&no_programs).unwrap_err()).contains("programs"));
        let too_many = ok.replace(r#""count":10"#, r#""count":20000000"#);
        assert!(Scene::from_json(&too_many).is_err());
    }

    #[test]
    fn orbit_track() {
        let s = Scene::from_json(
            r#"{"version":1,"nodes":[{"type":"camera","name":"cam"}],
              "animation":{"tracks":[{"node":"cam","path":"translation",
                "orbit":{"center":[0,0,0],"radius":4,"height":1,"frames":[0,40]}}]}}"#,
        )
        .unwrap();
        let f = s.at_frame(10.0).unwrap();
        let t = f.camera().translation;
        assert!((t[0] - 4.0).abs() < 1e-4 && (t[1] - 1.0).abs() < 1e-4 && t[2].abs() < 1e-4, "{t:?}");
    }
}
