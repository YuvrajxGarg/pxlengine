//! Output sinks: PNG / EXR stills and numbered sequences, video via ffmpeg, plus pass files.

use anyhow::{bail, Context as _, Result};
use pxl_render::{film, Image};
use pxl_scene::View;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    Png,
    Exr,
    Video,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Kind {
    pub format: Format,
    /// Path contains a `#` run that is replaced by the frame number.
    pub sequence: bool,
}

impl Kind {
    pub fn from_path(p: &Path) -> Result<Kind> {
        let ext = p.extension().and_then(|e| e.to_str()).unwrap_or("").to_ascii_lowercase();
        let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
        let sequence = name.contains('#');
        let format = match ext.as_str() {
            "png" => Format::Png,
            "exr" => Format::Exr,
            "mp4" | "mov" if !sequence => Format::Video,
            _ => bail!("unsupported output '{}': use .png, .exr, name_####.png/.exr, .mp4 or .mov", p.display()),
        };
        Ok(Kind { format, sequence })
    }

    pub fn is_multi(self) -> bool {
        self.sequence || self.format == Format::Video
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pass {
    Albedo,
    Normal,
    Depth,
    Alpha,
    Object,
    Shadow,
}

impl Pass {
    pub const ALL: [Pass; 6] = [Pass::Albedo, Pass::Normal, Pass::Depth, Pass::Alpha, Pass::Object, Pass::Shadow];

    pub fn name(self) -> &'static str {
        match self {
            Pass::Albedo => "albedo",
            Pass::Normal => "normal",
            Pass::Depth => "depth",
            Pass::Alpha => "alpha",
            Pass::Object => "object",
            Pass::Shadow => "shadow",
        }
    }

    pub fn parse_list(s: &str) -> Result<Vec<Pass>> {
        let mut out = Vec::new();
        for x in s.split(',').map(str::trim).filter(|x| !x.is_empty()) {
            if x == "all" {
                out.extend(Pass::ALL);
                continue;
            }
            let p = Pass::ALL
                .iter()
                .find(|p| p.name() == x)
                .with_context(|| format!("unknown pass '{x}' (albedo, normal, depth, alpha, object, shadow, all)"))?;
            out.push(*p);
        }
        let mut uniq = Vec::new();
        for p in out {
            if !uniq.contains(&p) {
                uniq.push(p);
            }
        }
        Ok(uniq)
    }

    /// Depth is metric, so it is always EXR; the rest follow the main output (PNG for video).
    fn ext(self, main: Format) -> &'static str {
        match (self, main) {
            (Pass::Depth, _) | (_, Format::Exr) => "exr",
            _ => "png",
        }
    }
}

pub struct Film {
    pub exposure: f32,
    pub view: View,
    pub transparent: bool,
    /// Plate frame (RGB8 at render size) and optional roto matte (one byte per pixel).
    pub plate: Option<Vec<u8>>,
    pub matte: Option<Vec<u8>>,
}

pub struct Sink {
    kind: Kind,
    path: PathBuf,
    passes: Vec<Pass>,
    video: Option<Child>,
    pub written: Vec<PathBuf>,
}

fn ffmpeg_works(args: &[&str]) -> bool {
    let Some(ff) = ffmpeg() else { return false };
    Command::new(ff)
        .args(["-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=black:s=256x256:d=0.1"])
        .args(args)
        .args(["-f", "null", "-"])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// ffmpeg executable: `PXL_FFMPEG`, else the first `ffmpeg(.exe)` on PATH.
pub fn ffmpeg() -> Option<PathBuf> {
    if let Some(p) = std::env::var_os("PXL_FFMPEG") {
        return Some(p.into());
    }
    pxl_gpu::slang::find_on_path(if cfg!(windows) { "ffmpeg.exe" } else { "ffmpeg" })
}

pub fn ffmpeg_status() -> String {
    let Some(ff) = ffmpeg() else { return "not found on PATH (needed for .mp4/.mov output; or set PXL_FFMPEG)".into() };
    match Command::new(&ff).arg("-version").output() {
        Ok(o) if o.status.success() => {
            let v = String::from_utf8_lossy(&o.stdout).lines().next().unwrap_or("").to_string();
            let nv = if ffmpeg_works(&["-c:v", "h264_nvenc"]) { "NVENC available" } else { "software x264" };
            format!("{v} ({nv})")
        }
        Ok(o) => format!("ffmpeg -version failed ({})", o.status),
        Err(e) => format!("{}: {e}", ff.display()),
    }
}

pub(crate) fn numbered(pattern: &Path, frame: i64) -> PathBuf {
    let s = pattern.to_string_lossy();
    let Some(end) = s.rfind('#').map(|e| e + 1) else { return pattern.to_path_buf() };
    let start = s[..end].trim_end_matches('#').len();
    let width = end - start;
    PathBuf::from(format!("{}{:0width$}{}", &s[..start], frame, &s[end..], width = width))
}

/// `shot.png` + depth -> `shot.depth.exr`
fn pass_path(main: &Path, pass: Pass, format: Format) -> PathBuf {
    let stem = main.file_stem().unwrap_or_default().to_string_lossy();
    main.with_file_name(format!("{stem}.{}.{}", pass.name(), pass.ext(format)))
}

fn ensure_parent(path: &Path) -> Result<()> {
    if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir)?;
    }
    Ok(())
}

fn write_png(path: &Path, w: u32, h: u32, data: &[u8], color: png::ColorType) -> Result<()> {
    ensure_parent(path)?;
    let file = std::fs::File::create(path).with_context(|| format!("creating {}", path.display()))?;
    let mut enc = png::Encoder::new(std::io::BufWriter::new(file), w, h);
    enc.set_color(color);
    enc.set_depth(png::BitDepth::Eight);
    enc.write_header()?.write_image_data(data)?;
    Ok(())
}

fn write_exr(path: &Path, w: u32, h: u32, rgba: Vec<f32>) -> Result<()> {
    ensure_parent(path)?;
    let img = image::Rgba32FImage::from_raw(w, h, rgba).context("EXR buffer size")?;
    img.save_with_format(path, image::ImageFormat::OpenExr).with_context(|| format!("writing {}", path.display()))
}

fn srgb(x: f32) -> f32 {
    let x = x.clamp(0.0, 1.0);
    if x <= 0.0031308 { 12.92 * x } else { 1.055 * x.powf(1.0 / 2.4) - 0.055 }
}

/// Stable, well-separated colour per object id (black = background).
fn id_color(id: u32) -> [u8; 3] {
    if id == u32::MAX {
        return [0, 0, 0];
    }
    let mut x = id.wrapping_mul(0x9e3779b9) ^ 0x5bd1e995;
    x ^= x >> 15;
    x = x.wrapping_mul(0x2c1b3c6d);
    x ^= x >> 12;
    [(x & 255) as u8 | 0x30, ((x >> 8) & 255) as u8 | 0x30, ((x >> 16) & 255) as u8 | 0x30]
}

fn rgba(c: &[f32], a: f32) -> [f32; 4] {
    [c[0], c[1], c[2], a]
}

impl Sink {
    pub fn open(kind: Kind, out: &Path, w: u32, h: u32, fps: f32, passes: Vec<Pass>) -> Result<Sink> {
        let mut video = None;
        if kind.format == Format::Video {
            ensure_parent(out)?;
            let ff = ffmpeg().context("ffmpeg not found on PATH (needed for video output; or set PXL_FFMPEG)")?;
            let is_mov = out.extension().is_some_and(|e| e.eq_ignore_ascii_case("mov"));
            let codec: Vec<&str> = if is_mov {
                vec!["-c:v", "prores_ks", "-profile:v", "3", "-pix_fmt", "yuv422p10le"]
            } else if ffmpeg_works(&["-c:v", "h264_nvenc"]) {
                vec!["-c:v", "h264_nvenc", "-preset", "p6", "-tune", "hq", "-rc", "vbr", "-cq", "17", "-b:v", "0", "-pix_fmt", "yuv420p"]
            } else {
                vec!["-c:v", "libx264", "-preset", "medium", "-crf", "16", "-pix_fmt", "yuv420p"]
            };
            let child = Command::new(ff)
                .args(["-y", "-hide_banner", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24"])
                .args(["-s", &format!("{w}x{h}"), "-framerate", &fps.to_string(), "-i", "-"])
                .args(["-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2"])
                .args(&codec)
                .args(if is_mov { &[][..] } else { &["-movflags", "+faststart"][..] })
                .arg(out)
                .stdin(Stdio::piped())
                .spawn()
                .context("starting ffmpeg")?;
            video = Some(child);
        }
        Ok(Sink { kind, path: out.to_path_buf(), passes, video, written: Vec::new() })
    }

    pub fn write(&mut self, frame: i64, img: &Image, f: &Film) -> Result<()> {
        let (w, h) = (img.width, img.height);
        let main = if self.kind.sequence { numbered(&self.path, frame) } else { self.path.clone() };
        // Video pass files go beside the clip as numbered sequences: clip_0001.depth.exr ...
        let pass_base = if self.kind.format == Format::Video {
            let stem = self.path.file_stem().unwrap_or_default().to_string_lossy();
            numbered(&self.path.with_file_name(format!("{stem}_####.png")), frame)
        } else {
            main.clone()
        };
        let k = 2f32.powf(f.exposure);
        match self.kind.format {
            Format::Png => {
                if let Some(plate) = &f.plate {
                    let rgb = film::composite_rgb8(img, f.exposure, f.view, plate, f.matte.as_deref());
                    write_png(&main, w, h, &rgb, png::ColorType::Rgb)?;
                } else if f.transparent {
                    // CG layer only (unpremultiplied colour + coverage); shadows go to the shadow pass.
                    let q = |x: f32| (x.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
                    let data: Vec<u8> = (0..img.alpha.len())
                        .flat_map(|i| {
                            let c = film::cg_display(img, i, f.exposure, f.view);
                            [q(c[0]), q(c[1]), q(c[2]), q(img.alpha[i])]
                        })
                        .collect();
                    write_png(&main, w, h, &data, png::ColorType::Rgba)?;
                } else {
                    let rgb = film::to_rgb8(img, f.exposure, f.view);
                    write_png(&main, w, h, &rgb, png::ColorType::Rgb)?;
                }
                self.written.push(main.clone());
            }
            Format::Exr => {
                // Scene-linear (no view transform), exposure applied, alpha = coverage.
                let data = match &f.plate {
                    Some(plate) => film::composite_linear(img, f.exposure, plate, f.matte.as_deref()),
                    None => img
                        .rgb
                        .chunks_exact(3)
                        .zip(&img.alpha)
                        .flat_map(|(c, a)| [c[0] * k, c[1] * k, c[2] * k, *a])
                        .collect(),
                };
                write_exr(&main, w, h, data)?;
                self.written.push(main.clone());
            }
            Format::Video => {
                let rgb = match &f.plate {
                    Some(plate) => film::composite_rgb8(img, f.exposure, f.view, plate, f.matte.as_deref()),
                    None => film::to_rgb8(img, f.exposure, f.view),
                };
                let child = self.video.as_mut().expect("video sink");
                child.stdin.as_mut().unwrap().write_all(&rgb).context("ffmpeg stopped accepting frames")?;
            }
        }
        let fmt = if self.kind.format == Format::Video { Format::Png } else { self.kind.format };
        for pass in self.passes.clone() {
            let path = pass_path(&pass_base, pass, fmt);
            let exr = pass.ext(fmt) == "exr";
            match pass {
                Pass::Depth => write_exr(&path, w, h, img.depth.iter().flat_map(|&d| [d, d, d, 1.0]).collect())?,
                Pass::Albedo if exr => write_exr(&path, w, h, img.albedo.chunks_exact(3).flat_map(|c| rgba(c, 1.0)).collect())?,
                Pass::Normal if exr => write_exr(&path, w, h, img.normal.chunks_exact(3).flat_map(|c| rgba(c, 1.0)).collect())?,
                Pass::Alpha if exr => write_exr(&path, w, h, img.alpha.iter().flat_map(|&a| [a, a, a, 1.0]).collect())?,
                Pass::Shadow if exr => write_exr(&path, w, h, img.shadow.iter().flat_map(|&a| [a, a, a, 1.0]).collect())?,
                Pass::Shadow => {
                    let d: Vec<u8> = img.shadow.iter().map(|&a| (a.clamp(0.0, 1.0) * 255.0 + 0.5) as u8).collect();
                    write_png(&path, w, h, &d, png::ColorType::Grayscale)?;
                }
                Pass::Albedo => {
                    let d: Vec<u8> = img.albedo.iter().map(|&x| (srgb(x) * 255.0 + 0.5) as u8).collect();
                    write_png(&path, w, h, &d, png::ColorType::Rgb)?;
                }
                Pass::Normal => {
                    let d: Vec<u8> = img.normal.iter().map(|&x| ((x * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0 + 0.5) as u8).collect();
                    write_png(&path, w, h, &d, png::ColorType::Rgb)?;
                }
                Pass::Alpha => {
                    let d: Vec<u8> = img.alpha.iter().map(|&a| (a.clamp(0.0, 1.0) * 255.0 + 0.5) as u8).collect();
                    write_png(&path, w, h, &d, png::ColorType::Grayscale)?;
                }
                Pass::Object => {
                    let d: Vec<u8> = img.object.iter().flat_map(|&id| id_color(id)).collect();
                    if exr {
                        let f: Vec<f32> = d.iter().map(|&b| b as f32 / 255.0).collect();
                        write_exr(&path, w, h, f.chunks_exact(3).flat_map(|c| rgba(c, 1.0)).collect())?;
                    } else {
                        write_png(&path, w, h, &d, png::ColorType::Rgb)?;
                    }
                }
            }
            self.written.push(path);
        }
        Ok(())
    }

    pub fn finish(mut self) -> Result<()> {
        if let Some(mut child) = self.video.take() {
            drop(child.stdin.take());
            let status = child.wait()?;
            if !status.success() {
                bail!("ffmpeg failed ({status})");
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn naming() {
        assert_eq!(numbered(Path::new("out/f_####.png"), 7), PathBuf::from("out/f_0007.png"));
        assert_eq!(numbered(Path::new("x#.png"), 12), PathBuf::from("x12.png"));
        assert_eq!(pass_path(Path::new("out/shot.png"), Pass::Depth, Format::Png), PathBuf::from("out/shot.depth.exr"));
        assert_eq!(pass_path(Path::new("out/f_0003.png"), Pass::Normal, Format::Png), PathBuf::from("out/f_0003.normal.png"));
        assert_eq!(Pass::parse_list("depth, all").unwrap().len(), 6);
        assert!(Pass::parse_list("dpeth").is_err());
    }
}
