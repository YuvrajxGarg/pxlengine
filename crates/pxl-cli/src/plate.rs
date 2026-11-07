//! Reading plate footage at render resolution: video (streamed through ffmpeg), numbered
//! image sequences (`plate_####.png`), or a single still.

use anyhow::{bail, Context as _, Result};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};

enum Source {
    Still(Vec<u8>),
    Sequence(PathBuf),
    Video { path: PathBuf, child: Option<Child>, next: i64 },
}

pub struct PlateReader {
    src: Source,
    w: u32,
    h: u32,
    gray: bool,
}

fn load_resized(path: &Path, w: u32, h: u32) -> Result<Vec<u8>> {
    let img = image::open(path).with_context(|| format!("reading plate image {}", path.display()))?.into_rgb8();
    let img = if img.dimensions() != (w, h) {
        image::imageops::resize(&img, w, h, image::imageops::FilterType::Lanczos3)
    } else {
        img
    };
    Ok(img.into_raw())
}

impl PlateReader {
    /// `gray`: return one byte per pixel (red channel), for mattes.
    pub fn open(path: &Path, w: u32, h: u32, gray: bool) -> Result<PlateReader> {
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("").to_ascii_lowercase();
        let video = ["mp4", "mov", "mkv", "webm", "avi", "m4v", "mxf"].contains(&ext.as_str());
        let src = if name.contains('#') {
            Source::Sequence(path.to_path_buf())
        } else if video {
            if !path.is_file() {
                bail!("plate video {} not found", path.display());
            }
            Source::Video { path: path.to_path_buf(), child: None, next: 0 }
        } else {
            Source::Still(load_resized(path, w, h)?)
        };
        Ok(PlateReader { src, w, h, gray })
    }

    fn to_output(&self, rgb: Vec<u8>) -> Vec<u8> {
        if self.gray { rgb.chunks_exact(3).map(|p| p[0]).collect() } else { rgb }
    }

    /// Plate frame `n` (0-based) as RGB8 (or gray) at render resolution.
    pub fn frame(&mut self, n: i64) -> Result<Vec<u8>> {
        let (w, h) = (self.w, self.h);
        let rgb = match &mut self.src {
            Source::Still(px) => px.clone(),
            Source::Sequence(pattern) => {
                let p = crate::output::numbered(pattern, n);
                load_resized(&p, w, h).with_context(|| format!("plate frame {n}"))?
            }
            Source::Video { path, child, next } => {
                if n < 0 {
                    bail!("plate frame {n} is before the start of the video");
                }
                if child.is_none() || n < *next {
                    if let Some(mut c) = child.take() {
                        let _ = c.kill();
                    }
                    let ff = crate::output::ffmpeg().context("ffmpeg is needed to read video plates")?;
                    let c = Command::new(ff)
                        .args(["-v", "error", "-i"])
                        .arg(&*path)
                        .args(["-vf", &format!("select=gte(n\\,{n}),scale={w}:{h}:flags=lanczos"), "-fps_mode", "passthrough"])
                        .args(["-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
                        .stdout(Stdio::piped())
                        .stderr(Stdio::null())
                        .spawn()
                        .context("starting ffmpeg to decode the plate")?;
                    *child = Some(c);
                    *next = n;
                }
                let out = child.as_mut().unwrap().stdout.as_mut().unwrap();
                let mut buf = vec![0u8; (w * h * 3) as usize];
                while *next <= n {
                    out.read_exact(&mut buf)
                        .with_context(|| format!("plate {} has no frame {n} (it ended at frame {})", path.display(), *next))?;
                    *next += 1;
                }
                buf
            }
        };
        Ok(self.to_output(rgb))
    }
}

impl Drop for PlateReader {
    fn drop(&mut self) {
        if let Source::Video { child: Some(c), .. } = &mut self.src {
            let _ = c.kill();
        }
    }
}
