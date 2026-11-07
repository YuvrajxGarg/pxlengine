//! Scene programs: assembles the path tracer source with scene-supplied Slang code.
//!
//! The core shader declares hook points (`#ifdef PXL_HOOK_*`) and a `//@PXL_USER_CODE`
//! marker. We prepend hook defines, then splice in at the marker:
//!   1. a generated `Params` struct + `P` global + loader (one float4 per param),
//!   2. the scene's code under `#line` directives so compiler errors name the user's file,
//!   3. generated wrappers that call the user's named functions.

use anyhow::{bail, Context as _, Result};
use pxl_scene::{Node, Programs, Scene};
use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::Path;

const CORE: &str = include_str!("../../../shaders/pathtrace.slang");
const MARKER: &str = "//@PXL_USER_CODE";

#[derive(Debug, Clone, Default)]
pub struct ProgramSet {
    /// Complete Slang source for the path tracer (and film pass, if any).
    pub source: String,
    /// Material program name -> id (1-based; 0 = none).
    pub material_ids: BTreeMap<String, u32>,
    /// Param names and component counts, in buffer order.
    pub params: Vec<(String, usize)>,
    pub camera: bool,
    pub propagate: bool,
    pub film: bool,
    /// Simulation names in scene order (index = SIM.index).
    pub simulations: Vec<String>,
}

fn slang_type(n: usize) -> &'static str {
    ["float", "float2", "float3", "float4"][n - 1]
}

fn line_name(s: &str) -> String {
    s.replace('\\', "/").replace('"', "'")
}

impl ProgramSet {
    /// The built-in renderer with no scene programs.
    pub fn none() -> ProgramSet {
        Self::assemble(None, &[], &[], "", &BTreeMap::new()).expect("core shader assembles")
    }

    pub fn from_scene(scene: &Scene, base_dir: &Path) -> Result<ProgramSet> {
        let Some(p) = &scene.programs else { return Ok(Self::none()) };
        let mut user = String::new();
        if let Some(file) = &p.source {
            let path = base_dir.join(file);
            let code = std::fs::read_to_string(&path)
                .with_context(|| format!("reading programs.source {}", path.display()))?;
            let _ = writeln!(user, "#line 1 \"{}\"", line_name(file));
            user.push_str(&code);
            user.push('\n');
        }
        if let Some(code) = &p.code {
            user.push_str("#line 1 \"programs.code\"\n");
            user.push_str(code);
            user.push('\n');
        }
        let mut names: Vec<String> = Vec::new();
        for n in &scene.nodes {
            if let Node::Mesh(m) = n {
                for mat in &m.materials {
                    if let Some(name) = &mat.program {
                        if !names.contains(name) {
                            names.push(name.clone());
                        }
                    }
                }
            }
        }
        let sims: Vec<(String, String, String)> =
            scene.simulations.iter().map(|s| (s.name.clone(), s.init.clone(), s.step.clone())).collect();
        Self::assemble(Some(p), &names, &sims, &user, &p.params)
    }

    fn assemble(
        p: Option<&Programs>,
        material_names: &[String],
        sims: &[(String, String, String)],
        user: &str,
        params: &BTreeMap<String, serde_json::Value>,
    ) -> Result<ProgramSet> {
        let mut set = ProgramSet::default();
        let mut defines = String::new();
        if let Some(p) = p {
            set.camera = p.camera.is_some();
            set.propagate = p.propagate.is_some();
            set.film = p.film.is_some();
        }
        for (on, name) in [(set.camera, "CAMERA"), (set.propagate, "PROPAGATE"), (set.film, "FILM")] {
            if on {
                let _ = writeln!(defines, "#define PXL_HOOK_{name} 1");
            }
        }
        if !sims.is_empty() {
            defines.push_str("#define PXL_SIMULATIONS 1\n");
        }

        // Params: one float4 slot each, exposed as P.<name>.
        let mut gen = String::from("#line 1 \"pxl-generated (params)\"\nstruct Params {\n");
        for (name, v) in params {
            let w = Programs::param_width(name, v)?;
            set.params.push((name.clone(), w));
            let _ = writeln!(gen, "    {} {name};", slang_type(w));
        }
        gen.push_str("    float _pxl_pad;\n};\nstatic Params P;\nvoid pxl_load_params() {\n");
        for (i, (name, w)) in set.params.iter().enumerate() {
            let sw = ["x", "xy", "xyz", "xyzw"][w - 1];
            let _ = writeln!(gen, "    P.{name} = params[{i}].{sw};");
        }
        gen.push_str("}\n");

        // Wrappers calling the scene's functions.
        let mut wrap = String::from("#line 1 \"pxl-generated (hooks)\"\n");
        if let Some(p) = p {
            if let Some(f) = &p.camera {
                let _ = writeln!(wrap, "bool pxl_hook_camera(CameraIn c, inout float3 o, inout float3 d) {{ return {f}(c, o, d); }}");
            }
            if let Some(f) = &p.propagate {
                let _ = writeln!(
                    wrap,
                    "Hit pxl_hook_propagate(inout Ray r, inout float3 radiance, inout float3 throughput, inout uint rng) {{ return {f}(r, radiance, throughput, rng); }}"
                );
            }
            if let Some(f) = &p.film {
                let _ = writeln!(wrap, "float3 pxl_hook_film(FilmIn i) {{ return {f}(i); }}");
            }
        }
        wrap.push_str("void pxl_run_material_program(uint id, inout Surface s, SurfaceIn i) {\n    switch (id) {\n");
        for (i, name) in material_names.iter().enumerate() {
            let id = i as u32 + 1;
            set.material_ids.insert(name.clone(), id);
            let _ = writeln!(wrap, "    case {id}: {name}(s, i); break;");
        }
        wrap.push_str("    default: break;\n    }\n}\n");
        for (kind, pick) in [("init", 1usize), ("step", 2usize)] {
            let _ = writeln!(wrap, "void pxl_sim_{kind}(uint sim, uint id, inout Agent a, inout uint rng) {{\n    switch (sim) {{");
            for (i, s) in sims.iter().enumerate() {
                let f = if pick == 1 { &s.1 } else { &s.2 };
                let _ = writeln!(wrap, "    case {i}: {f}(id, a, rng); break;");
            }
            wrap.push_str("    default: break;\n    }\n}\n");
        }
        set.simulations = sims.iter().map(|s| s.0.clone()).collect();

        let Some((head, tail)) = CORE.split_once(MARKER) else { bail!("core shader is missing {MARKER}") };
        let body_line = head.lines().count() + 1;
        set.source = format!(
            "{defines}#line 1 \"pathtrace.slang\"\n{head}{gen}{user}{wrap}#line {body_line} \"pathtrace.slang\"\n{tail}"
        );
        Ok(set)
    }

    /// Param values of `scene` packed one float4 per param (at least one element).
    pub fn param_data(&self, scene: &Scene) -> Vec<[f32; 4]> {
        let mut out: Vec<[f32; 4]> = self
            .params
            .iter()
            .map(|(name, _)| {
                scene
                    .programs
                    .as_ref()
                    .and_then(|p| p.params.get(name))
                    .map(Programs::param_values)
                    .unwrap_or_default()
            })
            .collect();
        if out.is_empty() {
            out.push([0.0; 4]);
        }
        out
    }
}
