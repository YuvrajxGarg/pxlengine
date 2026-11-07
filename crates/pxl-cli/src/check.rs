//! `pxl check`: validate a scene without rendering — schema, every animated frame, files it
//! references, and scene programs compiled with slangc. Fast feedback for agents and SDKs.

use anyhow::{bail, Context as _, Result};
use pxl_scene::{Node, Scene};
use std::path::Path;

fn exists(base: &Path, file: &str, what: &str, missing: &mut Vec<String>) {
    let p = base.join(file);
    let ok = if file.contains('#') {
        // Sequence: check the first numbered file in the directory matches the pattern.
        let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
        let (pre, post) = (name.split('#').next().unwrap_or(""), name.rsplit('#').next().unwrap_or(""));
        p.parent()
            .and_then(|d| std::fs::read_dir(if d.as_os_str().is_empty() { Path::new(".") } else { d }).ok())
            .map(|rd| {
                rd.flatten().any(|e| {
                    let n = e.file_name().to_string_lossy().into_owned();
                    n.starts_with(pre) && n.ends_with(post) && n.len() > pre.len() + post.len()
                })
            })
            .unwrap_or(false)
    } else {
        p.is_file()
    };
    if !ok {
        missing.push(format!("{what}: {} not found", p.display()));
    }
}

pub fn run(scene_path: &Path) -> Result<()> {
    let scene = Scene::load(scene_path)?;
    let base = scene_path.parent().map(Path::to_path_buf).unwrap_or_default();

    // Every frame of the animation must evaluate to a valid scene.
    let (a, b) = scene.frame_range().unwrap_or((0, 0));
    for f in a..=b {
        scene.at_frame(f as f64).with_context(|| format!("frame {f}"))?;
    }

    let mut missing = Vec::new();
    for n in &scene.nodes {
        match n {
            Node::Asset(x) => exists(&base, &x.file, "asset", &mut missing),
            Node::Mesh(m) => {
                for mat in &m.materials {
                    let i = &mat.inputs;
                    for t in [&i.base_color_texture, &i.specular_roughness_texture, &i.base_metalness_texture, &i.emission_color_texture, &i.normal_texture]
                        .into_iter()
                        .flatten()
                    {
                        exists(&base, t, "texture", &mut missing);
                    }
                }
            }
            Node::Camera(c) => {
                if let Some(t) = &c.track {
                    exists(&base, t, "camera track", &mut missing);
                }
            }
            Node::Light(_) => {}
        }
    }
    if let Some(h) = &scene.world.hdri {
        exists(&base, h, "world.hdri", &mut missing);
    }
    if let Some(p) = &scene.plate {
        exists(&base, &p.file, "plate", &mut missing);
        if let Some(m) = &p.matte {
            exists(&base, m, "plate.matte", &mut missing);
        }
    }
    if let Some(p) = &scene.programs {
        if let Some(s) = &p.source {
            exists(&base, s, "programs.source", &mut missing);
        }
    }
    if !missing.is_empty() {
        bail!("missing files:\n  {}", missing.join("\n  "));
    }

    // Compile scene programs exactly as the renderer will.
    let mut compiled = Vec::new();
    if scene.programs.is_some() {
        let set = pxl_render::ProgramSet::from_scene(&scene, &base)?;
        pxl_gpu::slang::compile_compute(&set.source, "main").context("scene programs")?;
        compiled.push("main");
        if set.film {
            pxl_gpu::slang::compile_compute(&set.source, "film_main").context("film program")?;
            compiled.push("film");
        }
        if !set.simulations.is_empty() {
            pxl_gpu::slang::compile_compute(&set.source, "sim_init").context("simulation init")?;
            pxl_gpu::slang::compile_compute(&set.source, "sim_step").context("simulation step")?;
            compiled.push("simulations");
        }
    }
    println!(
        "ok: {} nodes, frames {a}..{b}, {} simulation(s), {} fluid(s){}",
        scene.nodes.len(),
        scene.simulations.len(),
        scene.fluids.len(),
        if compiled.is_empty() { String::new() } else { format!(", programs compiled ({})", compiled.join(", ")) }
    );
    Ok(())
}
