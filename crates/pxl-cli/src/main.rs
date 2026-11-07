use anyhow::{bail, Context as _, Result};
use clap::{Parser, Subcommand};
use pxl_render::{profile, RenderOptions, Renderer, PROFILES};
use pxl_scene::Scene;
use std::path::{Path, PathBuf};
use std::time::Instant;

mod help;
mod output;

#[derive(Parser)]
#[command(name = "pxl", version, disable_help_subcommand = true, about = "pxlengine: headless 3D + VFX engine for agents. Start with `pxl help`.")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Render a scene JSON to an image, image sequence, or video.
    Render {
        scene: PathBuf,
        /// Output: still.png, frames_####.png (sequence) or clip.mp4.
        #[arg(long, short)]
        out: PathBuf,
        /// Quality profile: std/previs, std/draft, std/final.
        #[arg(long, default_value = "std/draft")]
        profile: String,
        /// Override settings: samples=N, bounces=N, seed=N, exposure=EV, clamp=X.
        #[arg(long = "set", value_name = "KEY=VALUE")]
        set: Vec<String>,
        /// Render a single frame of an animated scene.
        #[arg(long, conflicts_with_all = ["frame_start", "frame_end"])]
        frame: Option<f64>,
        /// First frame (inclusive). Defaults to animation.frame_start for sequences/video.
        #[arg(long)]
        frame_start: Option<i32>,
        /// Last frame (inclusive). Defaults to the animation's last frame.
        #[arg(long)]
        frame_end: Option<i32>,
    },
    /// Print help topics (agent documentation).
    Help {
        /// Topic name, or `all` / `--all` for every topic.
        #[arg(allow_hyphen_values = true)]
        topic: Option<String>,
    },
    /// Show the GPU pxl will use.
    Info,
}

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e:#}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    match Cli::parse().cmd {
        Cmd::Help { topic } => help::print(topic.as_deref()),
        Cmd::Info => {
            let ctx = pxl_gpu::Context::new()?;
            println!("device:  {}", ctx.info.name);
            println!("vulkan:  {}", ctx.info.api_version);
            println!("slangc:  {}", pxl_gpu::slang::find_slangc().map(|p| p.display().to_string()).unwrap_or_else(|e| e.to_string()));
            println!("ffmpeg:  {}", output::ffmpeg_status());
            println!("cache:   {}", pxl_gpu::slang::cache_dir().display());
            Ok(())
        }
        Cmd::Render { scene, out, profile: prof, set, frame, frame_start, frame_end } => {
            render(&scene, &out, &prof, &set, frame, frame_start, frame_end)
        }
    }
}

fn render(
    scene_path: &Path,
    out: &Path,
    prof: &str,
    set: &[String],
    frame: Option<f64>,
    frame_start: Option<i32>,
    frame_end: Option<i32>,
) -> Result<()> {
    let t0 = Instant::now();
    let scene = Scene::load(scene_path)?;
    let p = profile(prof).with_context(|| {
        format!("unknown profile '{prof}'; choose one of {}", PROFILES.map(|p| p.name).join(", "))
    })?;
    let mut opts = RenderOptions {
        samples: scene.render.samples.unwrap_or(p.samples),
        bounces: scene.render.bounces.unwrap_or(p.bounces),
        seed: scene.render.seed,
        clamp: 32.0,
    };
    let mut exposure = None;
    for kv in set {
        let (k, v) = kv.split_once('=').with_context(|| format!("--set expects KEY=VALUE, got '{kv}'"))?;
        let bad = || format!("--set {k}: cannot parse '{v}'");
        match k {
            "samples" => opts.samples = v.parse::<u32>().with_context(bad)?.clamp(1, 65536),
            "bounces" => opts.bounces = v.parse::<u32>().with_context(bad)?.min(64),
            "seed" => opts.seed = v.parse().with_context(bad)?,
            "exposure" => exposure = Some(v.parse::<f32>().with_context(bad)?),
            "clamp" => opts.clamp = v.parse().with_context(bad)?,
            _ => bail!("unknown --set key '{k}' (samples, bounces, seed, exposure, clamp)"),
        }
    }

    let kind = output::Kind::from_path(out)?;
    let frames: Vec<f64> = match (frame, frame_start, frame_end) {
        (Some(f), _, _) => vec![f],
        (None, None, None) if !kind.is_multi() => vec![scene.frame_range().map(|r| r.0).unwrap_or(0) as f64],
        _ => {
            let anim = scene.frame_range();
            let a = frame_start.or(anim.map(|r| r.0)).unwrap_or(0);
            let b = frame_end.or(anim.map(|r| r.1)).with_context(|| {
                "no frame range: add an `animation` block or pass --frame-start/--frame-end"
            })?;
            if b < a {
                bail!("--frame-end {b} is before --frame-start {a}");
            }
            (a..=b).map(|f| f as f64).collect()
        }
    };
    if frames.len() > 1 && !kind.is_multi() {
        bail!(
            "{} frames requested but '{}' is a single image; use frames_####.png or clip.mp4",
            frames.len(),
            out.display()
        );
    }

    let scenes: Vec<Scene> = frames
        .iter()
        .map(|&f| scene.at_frame(f))
        .collect::<Result<_>>()?;
    let base_dir = scene_path.parent().map(Path::to_path_buf).unwrap_or_default();
    let [w, h] = scenes[0].render.resolution;
    if scenes.iter().any(|s| s.render.resolution != [w, h]) {
        bail!("render.resolution cannot be animated");
    }

    let mut renderer = Renderer::new()?;
    let device = renderer.ctx.info.name.clone();
    let mut sink = output::Sink::open(kind, out, w, h, scenes[0].render.fps)?;
    let total = scenes.len();
    let multi = total > 1;
    let mut last_stats = None;
    let warnings = renderer.render_sequence(&scenes, &base_dir, &opts, |i, img, st| {
        let s = &scenes[i];
        let rgb = pxl_render::film::to_rgb8(&img, exposure.unwrap_or(s.render.exposure), s.render.view);
        sink.write(frames[i] as i64, &rgb)?;
        if multi {
            eprintln!("frame {} ({}/{total}) {:.0} ms", frames[i], i + 1, st.build_ms + st.render_ms);
        }
        last_stats = Some(st.clone());
        Ok(())
    })?;
    sink.finish()?;
    for w in &warnings {
        eprintln!("warning: {w}");
    }
    let st = last_stats.unwrap_or_default();
    eprintln!(
        "{} {w}x{h} | {} frame(s) | {} spp, {} bounces | {} tris, {} instances, {} lights, {} emissive tris | {:.2} s total | {device}",
        p.name, total, opts.samples, opts.bounces, st.triangles, st.instances, st.lights, st.emissive_triangles,
        t0.elapsed().as_secs_f64()
    );
    println!("{}", out.display());
    Ok(())
}
