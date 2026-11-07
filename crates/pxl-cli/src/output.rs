//! Output sinks: single PNG, numbered PNG sequence, or video via ffmpeg.

use anyhow::{bail, Context as _, Result};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Png,
    PngSequence,
    Video,
}

impl Kind {
    pub fn from_path(p: &Path) -> Result<Kind> {
        let ext = p.extension().and_then(|e| e.to_str()).unwrap_or("").to_ascii_lowercase();
        let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
        match ext.as_str() {
            "png" if name.contains('#') => Ok(Kind::PngSequence),
            "png" => Ok(Kind::Png),
            "mp4" | "mov" => Ok(Kind::Video),
            _ => bail!("unsupported output '{}': use .png, name_####.png, .mp4 or .mov", p.display()),
        }
    }

    pub fn is_multi(self) -> bool {
        self != Kind::Png
    }
}

pub enum Sink {
    Png(PathBuf, u32, u32),
    Sequence(PathBuf, u32, u32),
    Video(Child),
}

/// ffmpeg executable: `PXL_FFMPEG`, else the first `ffmpeg(.exe)` on PATH.
pub fn ffmpeg() -> Option<PathBuf> {
    if let Some(p) = std::env::var_os("PXL_FFMPEG") {
        return Some(p.into());
    }
    let exe = if cfg!(windows) { "ffmpeg.exe" } else { "ffmpeg" };
    pxl_gpu::slang::find_on_path(exe)
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

fn numbered(pattern: &Path, frame: i64) -> PathBuf {
    let s = pattern.to_string_lossy();
    let start = s.rfind('#').map(|end| s[..=end].trim_end_matches('#').len()).unwrap_or(0);
    let end = s.rfind('#').map(|e| e + 1).unwrap_or(0);
    let width = end - start;
    PathBuf::from(format!("{}{:0width$}{}", &s[..start], frame, &s[end..], width = width))
}

pub fn write_png(path: &Path, w: u32, h: u32, rgb: &[u8]) -> Result<()> {
    if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir)?;
    }
    let file = std::fs::File::create(path).with_context(|| format!("creating {}", path.display()))?;
    let mut enc = png::Encoder::new(std::io::BufWriter::new(file), w, h);
    enc.set_color(png::ColorType::Rgb);
    enc.set_depth(png::BitDepth::Eight);
    enc.write_header()?.write_image_data(rgb)?;
    Ok(())
}

impl Sink {
    pub fn open(kind: Kind, out: &Path, w: u32, h: u32, fps: f32) -> Result<Sink> {
        Ok(match kind {
            Kind::Png => Sink::Png(out.to_path_buf(), w, h),
            Kind::PngSequence => Sink::Sequence(out.to_path_buf(), w, h),
            Kind::Video => {
                if let Some(dir) = out.parent().filter(|d| !d.as_os_str().is_empty()) {
                    std::fs::create_dir_all(dir)?;
                }
                let is_mov = out.extension().is_some_and(|e| e.eq_ignore_ascii_case("mov"));
                let codec: Vec<&str> = if is_mov {
                    vec!["-c:v", "prores_ks", "-profile:v", "3", "-pix_fmt", "yuv422p10le"]
                } else if ffmpeg_works(&["-c:v", "h264_nvenc"]) {
                    vec!["-c:v", "h264_nvenc", "-preset", "p6", "-tune", "hq", "-rc", "vbr", "-cq", "17", "-b:v", "0", "-pix_fmt", "yuv420p"]
                } else {
                    vec!["-c:v", "libx264", "-preset", "medium", "-crf", "16", "-pix_fmt", "yuv420p"]
                };
                let ff = ffmpeg().context("ffmpeg not found on PATH (needed for video output; or set PXL_FFMPEG)")?;
                let child = Command::new(ff)
                    .args(["-y", "-hide_banner", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24"])
                    .args(["-s", &format!("{w}x{h}"), "-framerate", &fps.to_string(), "-i", "-"])
                    .args(["-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2"])
                    .args(&codec)
                    .args(if is_mov { &[][..] } else { &["-movflags", "+faststart"][..] })
                    .arg(out)
                    .stdin(Stdio::piped())
                    .spawn()
                    .context("starting ffmpeg (install it and put it on PATH for video output)")?;
                Sink::Video(child)
            }
        })
    }

    pub fn write(&mut self, frame: i64, rgb: &[u8]) -> Result<()> {
        match self {
            Sink::Png(p, w, h) => write_png(p, *w, *h, rgb),
            Sink::Sequence(p, w, h) => write_png(&numbered(p, frame), *w, *h, rgb),
            Sink::Video(child) => child
                .stdin
                .as_mut()
                .unwrap()
                .write_all(rgb)
                .context("ffmpeg stopped accepting frames"),
        }
    }

    pub fn finish(self) -> Result<()> {
        if let Sink::Video(mut child) = self {
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
    fn numbering() {
        assert_eq!(numbered(Path::new("out/f_####.png"), 7), PathBuf::from("out/f_0007.png"));
        assert_eq!(numbered(Path::new("x#.png"), 12), PathBuf::from("x12.png"));
    }
}
