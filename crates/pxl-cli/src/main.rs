use anyhow::{bail, Context as _, Result};
use clap::{Parser, Subcommand};
use pxl_render::{profile, Image, RenderOptions, PROFILES};
use pxl_scene::Scene;
use std::path::{Path, PathBuf};

mod help;

#[derive(Parser)]
#[command(name = "pxl", version, disable_help_subcommand = true, about = "pxlengine: headless 3D + VFX engine for agents. Start with `pxl help`.")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Render a scene JSON to an image.
    Render {
        scene: PathBuf,
        /// Output path (.png).
        #[arg(long, short)]
        out: PathBuf,
        /// Quality profile: std/previs, std/draft, std/final.
        #[arg(long, default_value = "std/draft")]
        profile: String,
        /// Override settings: samples=N, bounces=N, seed=N, exposure=EV.
        #[arg(long = "set", value_name = "KEY=VALUE")]
        set: Vec<String>,
    },
    /// Print help topics (agent documentation).
    Help {
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
            println!("cache:   {}", pxl_gpu::slang::cache_dir().display());
            Ok(())
        }
        Cmd::Render { scene, out, profile: prof, set } => render(&scene, &out, &prof, &set),
    }
}

fn render(scene_path: &Path, out: &Path, prof: &str, set: &[String]) -> Result<()> {
    let scene = Scene::load(scene_path)?;
    let p = profile(prof).with_context(|| {
        format!("unknown profile '{prof}'; choose one of {}", PROFILES.map(|p| p.name).join(", "))
    })?;
    let mut opts = RenderOptions {
        samples: scene.render.samples.unwrap_or(p.samples),
        bounces: scene.render.bounces.unwrap_or(p.bounces),
        seed: scene.render.seed,
    };
    let mut exposure = scene.render.exposure;
    for kv in set {
        let (k, v) = kv.split_once('=').with_context(|| format!("--set expects KEY=VALUE, got '{kv}'"))?;
        match k {
            "samples" => opts.samples = v.parse::<u32>()?.clamp(1, 65536),
            "bounces" => opts.bounces = v.parse::<u32>()?.min(64),
            "seed" => opts.seed = v.parse()?,
            "exposure" => exposure = v.parse()?,
            _ => bail!("unknown --set key '{k}' (samples, bounces, seed, exposure)"),
        }
    }
    let ext = out.extension().and_then(|e| e.to_str()).unwrap_or("").to_ascii_lowercase();
    if ext != "png" {
        bail!("unsupported output '{}': only .png in this release", out.display());
    }

    let mut ctx = pxl_gpu::Context::new()?;
    let (img, stats) = pxl_render::render(&mut ctx, &scene, &opts)?;
    write_png(&img, exposure, out)?;
    eprintln!(
        "{} {}x{} | {} spp, {} bounces | {} tris, {} meshes, {} lights | setup {:.0} ms, render {:.0} ms | {}",
        p.name, img.width, img.height, opts.samples, opts.bounces, stats.triangles, stats.instances, stats.lights,
        stats.setup_ms, stats.render_ms, ctx.info.name
    );
    println!("{}", out.display());
    Ok(())
}

/// AgX-like filmic curve is phase 3; for now ACES (Narkowicz fit) + sRGB OETF.
fn tonemap(x: f32) -> f32 {
    let (a, b, c, d, e) = (2.51, 0.03, 2.43, 0.59, 0.14);
    ((x * (a * x + b)) / (x * (c * x + d) + e)).clamp(0.0, 1.0)
}

fn srgb(x: f32) -> f32 {
    if x <= 0.0031308 { 12.92 * x } else { 1.055 * x.powf(1.0 / 2.4) - 0.055 }
}

fn write_png(img: &Image, exposure: f32, path: &Path) -> Result<()> {
    let scale = 2f32.powf(exposure);
    let data: Vec<u8> = img
        .rgb
        .iter()
        .map(|&v| (srgb(tonemap(v * scale)) * 255.0 + 0.5) as u8)
        .collect();
    if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir)?;
    }
    let file = std::fs::File::create(path).with_context(|| format!("creating {}", path.display()))?;
    let mut enc = png::Encoder::new(std::io::BufWriter::new(file), img.width, img.height);
    enc.set_color(png::ColorType::Rgb);
    enc.set_depth(png::BitDepth::Eight);
    enc.write_header()?.write_image_data(&data)?;
    Ok(())
}
