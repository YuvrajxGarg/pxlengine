//! Slang -> SPIR-V via `slangc`, cached on disk by content hash.
//!
//! slangc lookup order: `PXL_SLANGC`, `$VULKAN_SDK/Bin/slangc`, newest `C:\VulkanSDK\*\Bin`, `PATH`.

use anyhow::{bail, Context, Result};
use std::path::{Path, PathBuf};
use std::process::Command;

pub fn cache_dir() -> PathBuf {
    if let Some(d) = std::env::var_os("PXL_CACHE_DIR") {
        return d.into();
    }
    let base = std::env::var_os("LOCALAPPDATA")
        .or_else(|| std::env::var_os("XDG_CACHE_HOME"))
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))
        .unwrap_or_else(std::env::temp_dir);
    base.join("pxl").join("cache")
}

/// Find an executable on PATH. Splits on the separator and strips quotes per entry, so a
/// stray `"` in PATH (which makes `std::env::split_paths` and `Command` swallow the rest of
/// PATH into one entry) doesn't hide tools.
pub fn find_on_path(exe: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    let sep = if cfg!(windows) { ';' } else { ':' };
    path.to_string_lossy()
        .split(sep)
        .map(|d| d.trim().trim_matches('"'))
        .filter(|d| !d.is_empty())
        .map(|d| Path::new(d).join(exe))
        .find(|p| p.is_file())
}

pub fn find_slangc() -> Result<PathBuf> {
    let exe = if cfg!(windows) { "slangc.exe" } else { "slangc" };
    if let Some(p) = std::env::var_os("PXL_SLANGC") {
        return Ok(p.into());
    }
    if let Some(sdk) = std::env::var_os("VULKAN_SDK") {
        for sub in ["Bin", "bin"] {
            let p = Path::new(&sdk).join(sub).join(exe);
            if p.exists() {
                return Ok(p);
            }
        }
    }
    if cfg!(windows) {
        if let Ok(rd) = std::fs::read_dir("C:\\VulkanSDK") {
            let mut dirs: Vec<_> = rd.flatten().map(|e| e.path().join("Bin").join(exe)).filter(|p| p.exists()).collect();
            dirs.sort();
            if let Some(p) = dirs.pop() {
                return Ok(p);
            }
        }
    }
    if let Some(p) = find_on_path(exe) {
        return Ok(p);
    }
    bail!("slangc not found: install the Vulkan SDK or set PXL_SLANGC")
}

fn fnv1a(parts: &[&[u8]]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for p in parts {
        for b in *p {
            h ^= *b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        h ^= 0xff;
    }
    h
}

/// Compile one compute entry point. Returns SPIR-V bytes.
pub fn compile_compute(source: &str, entry: &str) -> Result<Vec<u8>> {
    let dir = cache_dir();
    std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
    let key = fnv1a(&[source.as_bytes(), entry.as_bytes(), env!("CARGO_PKG_VERSION").as_bytes()]);
    let spv = dir.join(format!("{key:016x}.spv"));
    if let Ok(bytes) = std::fs::read(&spv) {
        return Ok(bytes);
    }
    let src = dir.join(format!("{key:016x}.slang"));
    std::fs::write(&src, source)?;
    let slangc = find_slangc()?;
    let out = Command::new(&slangc)
        .arg(&src)
        .args(["-target", "spirv", "-profile", "spirv_1_6", "-capability", "spvRayQueryKHR", "-stage", "compute", "-entry", entry, "-O2", "-o"])
        .arg(&spv)
        .output()
        .with_context(|| format!("running {}", slangc.display()))?;
    if !out.status.success() {
        let _ = std::fs::remove_file(&spv);
        bail!(
            "Slang compilation failed:\n{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
    }
    std::fs::read(&spv).context("reading compiled SPIR-V")
}
