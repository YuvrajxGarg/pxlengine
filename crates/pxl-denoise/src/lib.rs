//! Intel Open Image Denoise, loaded at runtime (OIDN ships beside the binary).
//!
//! Search order: `PXL_OIDN_DIR`, `<exe dir>/oidn/bin`, `<exe dir>/../../vendor/oidn/bin`
//! (cargo target dir), `./vendor/oidn/bin`.

use anyhow::{anyhow, bail, Context, Result};
use libloading::Library;
use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::path::{Path, PathBuf};

type Handle = *mut c_void;
const FORMAT_FLOAT3: c_int = 3;
const QUALITY_BALANCED: c_int = 5;
const QUALITY_HIGH: c_int = 6;

#[cfg(windows)]
const LIB_NAME: &str = "OpenImageDenoise.dll";
#[cfg(target_os = "macos")]
const LIB_NAME: &str = "libOpenImageDenoise.2.dylib";
#[cfg(all(unix, not(target_os = "macos")))]
const LIB_NAME: &str = "libOpenImageDenoise.so.2";

struct Api {
    new_device: unsafe extern "C" fn(c_int) -> Handle,
    commit_device: unsafe extern "C" fn(Handle),
    release_device: unsafe extern "C" fn(Handle),
    get_device_error: unsafe extern "C" fn(Handle, *mut *const c_char) -> c_int,
    get_device_int: unsafe extern "C" fn(Handle, *const c_char) -> c_int,
    new_buffer: unsafe extern "C" fn(Handle, usize) -> Handle,
    write_buffer: unsafe extern "C" fn(Handle, usize, usize, *const c_void),
    read_buffer: unsafe extern "C" fn(Handle, usize, usize, *mut c_void),
    release_buffer: unsafe extern "C" fn(Handle),
    new_filter: unsafe extern "C" fn(Handle, *const c_char) -> Handle,
    set_filter_image: unsafe extern "C" fn(Handle, *const c_char, Handle, c_int, usize, usize, usize, usize, usize),
    set_filter_bool: unsafe extern "C" fn(Handle, *const c_char, bool),
    set_filter_int: unsafe extern "C" fn(Handle, *const c_char, c_int),
    commit_filter: unsafe extern "C" fn(Handle),
    execute_filter: unsafe extern "C" fn(Handle),
    release_filter: unsafe extern "C" fn(Handle),
}

pub struct Denoiser {
    api: Api,
    device: Handle,
    pub device_name: String,
    pub path: PathBuf,
    _lib: Library,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Quality {
    Balanced,
    High,
}

fn candidates() -> Vec<PathBuf> {
    let mut v = Vec::new();
    if let Some(d) = std::env::var_os("PXL_OIDN_DIR") {
        let d = PathBuf::from(d);
        v.push(d.join("bin"));
        v.push(d.join("lib"));
        v.push(d);
    }
    let sub = if cfg!(windows) { "bin" } else { "lib" };
    if let Some(exe_dir) = std::env::current_exe().ok().and_then(|p| p.parent().map(Path::to_path_buf)) {
        v.push(exe_dir.join("oidn").join(sub));
        v.push(exe_dir.join("..").join("..").join("vendor").join("oidn").join(sub));
    }
    v.push(PathBuf::from("vendor").join("oidn").join(sub));
    v
}

unsafe fn load_library(path: &Path) -> Result<Library> {
    #[cfg(windows)]
    {
        // Resolve OIDN's own dependencies (core, device modules, TBB) from its folder.
        const LOAD_WITH_ALTERED_SEARCH_PATH: u32 = 0x8;
        let l = libloading::os::windows::Library::load_with_flags(path, LOAD_WITH_ALTERED_SEARCH_PATH)?;
        Ok(l.into())
    }
    #[cfg(not(windows))]
    {
        Ok(Library::new(path)?)
    }
}

impl Denoiser {
    /// Find and initialise OIDN on the best available device (CUDA on NVIDIA, else CPU).
    pub fn new() -> Result<Denoiser> {
        let path = candidates()
            .into_iter()
            .map(|d| d.join(LIB_NAME))
            .find(|p| p.is_file())
            .ok_or_else(|| anyhow!("Open Image Denoise not found (run scripts/fetch-oidn.sh or set PXL_OIDN_DIR)"))?;
        unsafe {
            let lib = load_library(&path).with_context(|| format!("loading {}", path.display()))?;
            macro_rules! sym {
                ($name:literal) => {
                    *lib.get(concat!($name, "\0").as_bytes()).with_context(|| format!("OIDN symbol {}", $name))?
                };
            }
            let api = Api {
                new_device: sym!("oidnNewDevice"),
                commit_device: sym!("oidnCommitDevice"),
                release_device: sym!("oidnReleaseDevice"),
                get_device_error: sym!("oidnGetDeviceError"),
                get_device_int: sym!("oidnGetDeviceInt"),
                new_buffer: sym!("oidnNewBuffer"),
                write_buffer: sym!("oidnWriteBuffer"),
                read_buffer: sym!("oidnReadBuffer"),
                release_buffer: sym!("oidnReleaseBuffer"),
                new_filter: sym!("oidnNewFilter"),
                set_filter_image: sym!("oidnSetFilterImage"),
                set_filter_bool: sym!("oidnSetFilterBool"),
                set_filter_int: sym!("oidnSetFilterInt"),
                commit_filter: sym!("oidnCommitFilter"),
                execute_filter: sym!("oidnExecuteFilter"),
                release_filter: sym!("oidnReleaseFilter"),
            };
            // Ask for device types explicitly: OIDN's automatic choice can give up entirely when an
            // unrelated backend (e.g. Intel Level Zero on a machine without it) fails to initialise.
            // `PXL_OIDN_DEVICE=cpu|cuda|hip|metal` forces one.
            let order: Vec<c_int> = match std::env::var("PXL_OIDN_DEVICE").ok().as_deref() {
                Some("cpu") => vec![1],
                Some("cuda") => vec![3],
                Some("hip") => vec![4],
                Some("metal") => vec![5],
                _ => vec![3, 4, 5, 1],
            };
            let mut errors = Vec::new();
            let mut device = std::ptr::null_mut();
            for ty in order {
                let dev = (api.new_device)(ty);
                let mut msg: *const c_char = std::ptr::null();
                if dev.is_null() {
                    (api.get_device_error)(std::ptr::null_mut(), &mut msg); // clear per-thread error
                    continue;
                }
                (api.commit_device)(dev);
                if (api.get_device_error)(dev, &mut msg) == 0 {
                    device = dev;
                    break;
                }
                if !msg.is_null() {
                    errors.push(CStr::from_ptr(msg).to_string_lossy().into_owned());
                }
                (api.release_device)(dev);
            }
            if device.is_null() {
                bail!("OIDN found no usable device{}", if errors.is_empty() { String::new() } else { format!(": {}", errors.join("; ")) });
            }
            let mut d = Denoiser { api, device, device_name: String::new(), path, _lib: lib };
            d.check()?;
            let ty = (d.api.get_device_int)(device, c"type".as_ptr());
            d.device_name = match ty {
                1 => "CPU",
                2 => "SYCL",
                3 => "CUDA",
                4 => "HIP",
                5 => "Metal",
                _ => "unknown",
            }
            .to_string();
            Ok(d)
        }
    }

    fn check(&self) -> Result<()> {
        let mut msg: *const c_char = std::ptr::null();
        let code = unsafe { (self.api.get_device_error)(self.device, &mut msg) };
        if code != 0 {
            let m = if msg.is_null() { "unknown error".into() } else { unsafe { CStr::from_ptr(msg) }.to_string_lossy().into_owned() };
            bail!("OIDN error {code}: {m}");
        }
        Ok(())
    }

    /// Denoise linear HDR `color` (RGB f32) in place. `albedo` and `normal` (RGB f32, same size)
    /// preserve texture and geometric detail.
    pub fn denoise(
        &self,
        color: &mut [f32],
        albedo: Option<&[f32]>,
        normal: Option<&[f32]>,
        width: u32,
        height: u32,
        quality: Quality,
    ) -> Result<()> {
        let (w, h) = (width as usize, height as usize);
        let bytes = w * h * 12;
        if color.len() * 4 != bytes {
            bail!("denoise: color buffer size mismatch");
        }
        let a = &self.api;
        unsafe {
            let mut buffers = Vec::new();
            let mut img = |name: &str, data: &[f32], filter: Handle| -> Result<Handle> {
                let b = (a.new_buffer)(self.device, bytes);
                if b.is_null() {
                    self.check()?;
                    bail!("OIDN buffer allocation failed");
                }
                (a.write_buffer)(b, 0, bytes, data.as_ptr() as *const c_void);
                let n = CString::new(name)?;
                (a.set_filter_image)(filter, n.as_ptr(), b, FORMAT_FLOAT3, w, h, 0, 0, 0);
                buffers.push(b);
                Ok(b)
            };
            let filter = (a.new_filter)(self.device, c"RT".as_ptr());
            let result = (|| -> Result<()> {
                let color_buf = img("color", color, filter)?;
                if let Some(al) = albedo {
                    img("albedo", al, filter)?;
                    if let Some(n) = normal {
                        img("normal", n, filter)?;
                    }
                }
                (a.set_filter_image)(filter, c"output".as_ptr(), color_buf, FORMAT_FLOAT3, w, h, 0, 0, 0);
                (a.set_filter_bool)(filter, c"hdr".as_ptr(), true);
                (a.set_filter_int)(
                    filter,
                    c"quality".as_ptr(),
                    if quality == Quality::High { QUALITY_HIGH } else { QUALITY_BALANCED },
                );
                (a.commit_filter)(filter);
                (a.execute_filter)(filter);
                self.check()?;
                (a.read_buffer)(color_buf, 0, bytes, color.as_mut_ptr() as *mut c_void);
                self.check()
            })();
            (a.release_filter)(filter);
            for b in buffers {
                (a.release_buffer)(b);
            }
            result
        }
    }
}

impl Drop for Denoiser {
    fn drop(&mut self) {
        unsafe { (self.api.release_device)(self.device) };
    }
}
