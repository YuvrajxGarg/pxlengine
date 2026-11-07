use anyhow::{bail, Context as _, Result};
use clap::{Parser, Subcommand};
use pxl_render::{profile, RenderOptions, Renderer, PROFILES};
use pxl_scene::Scene;
use std::path::{Path, PathBuf};
use std::time::Instant;

mod help;
mod output;
mod plate;

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
        /// Output: still.png / still.exr, frames_####.png|exr (sequence), clip.mp4 / clip.mov.
        #[arg(long, short)]
        out: PathBuf,
        /// Quality profile: std/previs, std/draft, std/final, std/path.
        #[arg(long, default_value = "std/draft")]
        profile: String,
        /// Extra pass files beside the output: albedo,normal,depth,alpha,object or all.
        #[arg(long)]
        passes: Option<String>,
        /// Override settings: samples=N, bounces=N, seed=N, exposure=EV, clamp=X, denoise=BOOL,
        /// motion_blur=SHUTTER.
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
    /// Write the scene camera, per frame, as a Nuke .chan track (for round trips and other tools).
    Track {
        scene: PathBuf,
        #[arg(long, short)]
        out: PathBuf,
        #[arg(long)]
        frame_start: Option<i32>,
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
            drop(ctx);
            let mut r = Renderer::new(pxl_render::ProgramSet::none())?;
            println!("denoise: {}", r.denoiser_status());
            println!("cache:   {}", pxl_gpu::slang::cache_dir().display());
            Ok(())
        }
        Cmd::Track { scene, out, frame_start, frame_end } => export_track(&scene, &out, frame_start, frame_end),
        Cmd::Render { scene, out, profile: prof, passes, set, frame, frame_start, frame_end } => {
            let passes = passes.as_deref().map(output::Pass::parse_list).transpose()?.unwrap_or_default();
            render(&scene, &out, &prof, passes, &set, frame, frame_start, frame_end)
        }
    }
}

#[allow(clippy::too_many_arguments)]
/// Load the camera's `.chan` track (if any) once, to apply to every evaluated scene.
fn camera_track(scene: &Scene, base_dir: &Path) -> Result<Option<pxl_scene::track::CameraTrack>> {
    match &scene.camera().track {
        Some(f) => Ok(Some(pxl_scene::track::CameraTrack::load(&base_dir.join(f))?)),
        None => Ok(None),
    }
}

fn scene_at(scene: &Scene, track: Option<&pxl_scene::track::CameraTrack>, f: f64) -> Result<Scene> {
    let mut s = scene.at_frame(f)?;
    if let Some(t) = track {
        t.apply(s.camera_mut(), f);
    }
    Ok(s)
}

fn export_track(scene_path: &Path, out: &Path, frame_start: Option<i32>, frame_end: Option<i32>) -> Result<()> {
    let scene = Scene::load(scene_path)?;
    let base_dir = scene_path.parent().map(Path::to_path_buf).unwrap_or_default();
    let track = camera_track(&scene, &base_dir)?;
    let range = scene.frame_range().unwrap_or((0, 0));
    let (a, b) = (frame_start.unwrap_or(range.0), frame_end.unwrap_or(range.1));
    let mut text = String::from("# pxl camera: frame tx ty tz rx ry rz vfov (degrees, rotation order ZXY)\n");
    for f in a..=b {
        let s = scene_at(&scene, track.as_ref(), f as f64)?;
        text.push_str(&pxl_scene::track::CameraTrack::chan_line(s.camera(), f as f64));
        text.push('\n');
    }
    if let Some(dir) = out.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir)?;
    }
    std::fs::write(out, text)?;
    eprintln!("{} frames ({a}..{b}) -> {}", b - a + 1, out.display());
    Ok(())
}

fn render(
    scene_path: &Path,
    out: &Path,
    prof: &str,
    passes: Vec<output::Pass>,
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
        denoise: scene.render.denoise.unwrap_or(p.denoise),
    };
    let mut exposure = None;
    let mut motion_blur = scene.render.motion_blur;
    for kv in set {
        let (k, v) = kv.split_once('=').with_context(|| format!("--set expects KEY=VALUE, got '{kv}'"))?;
        let bad = || format!("--set {k}: cannot parse '{v}'");
        match k {
            "samples" => opts.samples = v.parse::<u32>().with_context(bad)?.clamp(1, 65536),
            "bounces" => opts.bounces = v.parse::<u32>().with_context(bad)?.min(64),
            "seed" => opts.seed = v.parse().with_context(bad)?,
            "exposure" => exposure = Some(v.parse::<f32>().with_context(bad)?),
            "clamp" => opts.clamp = v.parse().with_context(bad)?,
            "denoise" => opts.denoise = v.parse().with_context(bad)?,
            "motion_blur" => motion_blur = Some(v.parse::<f32>().with_context(bad)?.clamp(0.0, 1.0)),
            _ => bail!("unknown --set key '{k}' (samples, bounces, seed, exposure, clamp, denoise, motion_blur)"),
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

    // Motion blur: evaluate the scene at several instants across the open shutter (centred on
    // the frame); samples are spread across them, so moving objects and cameras smear.
    let shutter = motion_blur.filter(|s| *s > 0.0);
    let subframe_count = if shutter.is_some() { opts.samples.clamp(1, 16) } else { 1 };
    let base_dir = scene_path.parent().map(Path::to_path_buf).unwrap_or_default();
    let track = camera_track(&scene, &base_dir)?;
    let mut frame_scenes: Vec<Vec<Scene>> = Vec::with_capacity(frames.len());
    for &f in &frames {
        let mut subs = Vec::with_capacity(subframe_count as usize);
        for k in 0..subframe_count {
            let t = match shutter {
                Some(sh) => f + ((k as f64 + 0.5) / subframe_count as f64 - 0.5) * sh as f64,
                None => f,
            };
            subs.push(scene_at(&scene, track.as_ref(), t)?);
        }
        frame_scenes.push(subs);
    }
    let first = frame_scenes[0][0].clone();
    let [w, h] = first.render.resolution;
    if frame_scenes.iter().flatten().any(|s| s.render.resolution != [w, h]) {
        bail!("render.resolution cannot be animated");
    }
    let mut films: Vec<output::Film> = frame_scenes
        .iter()
        .map(|subs| output::Film {
            exposure: exposure.unwrap_or(subs[0].render.exposure),
            view: subs[0].render.view,
            transparent: subs[0].render.transparent,
            plate: None,
            matte: None,
        })
        .collect();
    let mut plates = match &first.plate {
        Some(p) => Some((
            plate::PlateReader::open(&base_dir.join(&p.file), w, h, false)?,
            p.matte.as_ref().map(|m| plate::PlateReader::open(&base_dir.join(m), w, h, true)).transpose()?,
            p.offset as i64,
        )),
        None => None,
    };

    let programs = pxl_render::ProgramSet::from_scene(&frame_scenes[0][0], &base_dir)?;
    let mut renderer = Renderer::new(programs)?;
    let device = renderer.ctx.info.name.clone();
    let mut sink = output::Sink::open(kind, out, w, h, first.render.fps, passes)?;
    let total = frames.len();
    let multi = total > 1;
    let mut last_stats = None;
    let mut denoised = false;
    let warnings = renderer.render_sequence(frame_scenes, &base_dir, &opts, |i, img, st| {
        if let Some((reader, matte, offset)) = &mut plates {
            let n = frames[i].round() as i64 + *offset;
            films[i].plate = Some(reader.frame(n)?);
            films[i].matte = matte.as_mut().map(|m| m.frame(n)).transpose()?;
        }
        sink.write(frames[i] as i64, &img, &films[i])?;
        films[i].plate = None;
        films[i].matte = None;
        denoised = img.denoised;
        if multi {
            eprintln!(
                "frame {} ({}/{total}) {:.0} ms{}",
                frames[i],
                i + 1,
                st.sim_ms + st.build_ms + st.render_ms + st.denoise_ms,
                if st.sim_ms > 0.5 { format!(" (sim {:.0} ms)", st.sim_ms) } else { String::new() }
            );
        }
        last_stats = Some(st.clone());
        Ok(())
    })?;
    let written = std::mem::take(&mut sink.written);
    sink.finish()?;
    for w in &warnings {
        eprintln!("warning: {w}");
    }
    let st = last_stats.unwrap_or_default();
    let mut extras = Vec::new();
    if denoised {
        extras.push("denoised".to_string());
    }
    if st.sim_ms > 0.5 {
        extras.push(format!("sim {:.0} ms + build {:.0} ms + trace {:.0} ms", st.sim_ms, st.build_ms, st.render_ms));
    }
    if let Some(sh) = shutter {
        extras.push(format!("motion blur {sh} x{subframe_count}"));
    }
    if first.camera().lens().0 > 0.0 {
        extras.push(format!("f/{}", first.camera().fstop.unwrap_or(0.0)));
    }
    eprintln!(
        "{} {w}x{h} | {} frame(s) | {} spp, {} bounces{} | {} tris, {} instances, {} lights, {} emissive tris | {:.2} s total | {device}",
        p.name, total, opts.samples, opts.bounces,
        if extras.is_empty() { String::new() } else { format!(", {}", extras.join(", ")) },
        st.triangles, st.instances, st.lights, st.emissive_triangles,
        t0.elapsed().as_secs_f64()
    );
    if kind.format == output::Format::Video {
        println!("{}", out.display());
    }
    for p in &written {
        println!("{}", p.display());
    }
    Ok(())
}
