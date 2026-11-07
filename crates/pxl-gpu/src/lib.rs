//! Vulkan core for pxlengine: device with inline ray queries, buffers, acceleration
//! structures, compute pipelines, and a Slang -> SPIR-V compiler with an on-disk cache.

mod accel;
mod compute;
pub mod slang;

pub use accel::{BlasInput, TlasInstance};
pub use ash::vk;
pub use compute::{Binding, ComputePipeline, Resource};
pub use gpu_allocator::MemoryLocation;

use anyhow::{anyhow, bail, Context as _, Result};
use ash::{Device, Entry, Instance};
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme, Allocator, AllocatorCreateDesc};
use std::ffi::CStr;

const REQUIRED_EXTENSIONS: [&CStr; 3] = [
    ash::khr::acceleration_structure::NAME,
    ash::khr::ray_query::NAME,
    ash::khr::deferred_host_operations::NAME,
];

/// Lightweight handle; the allocation lives in the owning [`Context`].
#[derive(Debug, Clone, Copy)]
pub struct Buffer {
    pub handle: vk::Buffer,
    pub size: u64,
    pub address: u64,
    id: usize,
}

#[derive(Debug, Clone)]
pub struct DeviceInfo {
    pub name: String,
    pub api_version: String,
    pub driver_version: u32,
    pub discrete: bool,
}

pub struct Context {
    _entry: Entry,
    pub instance: Instance,
    pub physical_device: vk::PhysicalDevice,
    pub device: Device,
    pub queue: vk::Queue,
    pub queue_family: u32,
    pub info: DeviceInfo,
    pub(crate) as_loader: ash::khr::acceleration_structure::Device,
    pub(crate) scratch_align: u64,
    cmd_pool: vk::CommandPool,
    allocator: Option<Allocator>,
    buffers: Vec<Option<(vk::Buffer, Allocation)>>,
    pub(crate) accels: Vec<vk::AccelerationStructureKHR>,
    pub(crate) pipelines: Vec<ComputePipeline>,
    debug: Option<(ash::ext::debug_utils::Instance, vk::DebugUtilsMessengerEXT)>,
}

unsafe extern "system" fn debug_cb(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    _ty: vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    _user: *mut std::ffi::c_void,
) -> vk::Bool32 {
    let msg = unsafe { CStr::from_ptr((*data).p_message) }.to_string_lossy();
    eprintln!("[vulkan {severity:?}] {msg}");
    vk::FALSE
}

impl Context {
    /// Create a context on the best ray-query capable GPU.
    /// `PXL_DEVICE=<substring>` selects a device by name; `PXL_VALIDATION=1` enables validation.
    pub fn new() -> Result<Context> {
        let entry = unsafe { Entry::load() }.context("loading the Vulkan loader (is a GPU driver installed?)")?;
        let validation = std::env::var_os("PXL_VALIDATION").is_some();

        let app = vk::ApplicationInfo::default()
            .application_name(c"pxlengine")
            .engine_name(c"pxl")
            .api_version(vk::API_VERSION_1_3);
        let layers: Vec<*const i8> = if validation { vec![c"VK_LAYER_KHRONOS_validation".as_ptr()] } else { vec![] };
        let exts: Vec<*const i8> = if validation { vec![ash::ext::debug_utils::NAME.as_ptr()] } else { vec![] };
        let instance = unsafe {
            entry.create_instance(
                &vk::InstanceCreateInfo::default()
                    .application_info(&app)
                    .enabled_layer_names(&layers)
                    .enabled_extension_names(&exts),
                None,
            )
        }
        .context("creating Vulkan instance")?;

        let debug = if validation {
            let du = ash::ext::debug_utils::Instance::new(&entry, &instance);
            let info = vk::DebugUtilsMessengerCreateInfoEXT::default()
                .message_severity(
                    vk::DebugUtilsMessageSeverityFlagsEXT::WARNING | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
                )
                .message_type(
                    vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                        | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                        | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
                )
                .pfn_user_callback(Some(debug_cb));
            let m = unsafe { du.create_debug_utils_messenger(&info, None)? };
            Some((du, m))
        } else {
            None
        };

        let (physical_device, queue_family, info) = pick_device(&instance)?;

        let mut f12 = vk::PhysicalDeviceVulkan12Features::default()
            .buffer_device_address(true)
            .descriptor_indexing(true)
            .scalar_block_layout(true);
        let mut f13 = vk::PhysicalDeviceVulkan13Features::default().synchronization2(true);
        let mut f_as = vk::PhysicalDeviceAccelerationStructureFeaturesKHR::default().acceleration_structure(true);
        let mut f_rq = vk::PhysicalDeviceRayQueryFeaturesKHR::default().ray_query(true);
        let prio = [1.0f32];
        let queues = [vk::DeviceQueueCreateInfo::default().queue_family_index(queue_family).queue_priorities(&prio)];
        let ext_ptrs: Vec<*const i8> = REQUIRED_EXTENSIONS.iter().map(|e| e.as_ptr()).collect();
        let device = unsafe {
            instance.create_device(
                physical_device,
                &vk::DeviceCreateInfo::default()
                    .queue_create_infos(&queues)
                    .enabled_extension_names(&ext_ptrs)
                    .push_next(&mut f12)
                    .push_next(&mut f13)
                    .push_next(&mut f_as)
                    .push_next(&mut f_rq),
                None,
            )
        }
        .context("creating Vulkan device")?;
        let queue = unsafe { device.get_device_queue(queue_family, 0) };

        let scratch_align = {
            let mut asp = vk::PhysicalDeviceAccelerationStructurePropertiesKHR::default();
            {
                let mut p2 = vk::PhysicalDeviceProperties2::default().push_next(&mut asp);
                unsafe { instance.get_physical_device_properties2(physical_device, &mut p2) };
            }
            (asp.min_acceleration_structure_scratch_offset_alignment as u64).max(1)
        };

        let as_loader = ash::khr::acceleration_structure::Device::new(&instance, &device);
        let cmd_pool = unsafe {
            device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(queue_family)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )?
        };
        let allocator = Allocator::new(&AllocatorCreateDesc {
            instance: instance.clone(),
            device: device.clone(),
            physical_device,
            debug_settings: Default::default(),
            buffer_device_address: true,
            allocation_sizes: Default::default(),
        })?;

        Ok(Context {
            _entry: entry,
            instance,
            physical_device,
            device,
            queue,
            queue_family,
            info,
            as_loader,
            scratch_align,
            cmd_pool,
            allocator: Some(allocator),
            buffers: Vec::new(),
            accels: Vec::new(),
            pipelines: Vec::new(),
            debug,
        })
    }

    pub fn create_buffer(&mut self, size: u64, usage: vk::BufferUsageFlags, location: MemoryLocation, name: &str) -> Result<Buffer> {
        let size = size.max(16);
        let handle = unsafe {
            self.device.create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(size)
                    .usage(usage | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS),
                None,
            )?
        };
        let requirements = unsafe { self.device.get_buffer_memory_requirements(handle) };
        let alloc = self
            .allocator
            .as_mut()
            .unwrap()
            .allocate(&AllocationCreateDesc {
                name,
                requirements,
                location,
                linear: true,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            })
            .with_context(|| format!("allocating {size} bytes for '{name}'"))?;
        unsafe { self.device.bind_buffer_memory(handle, alloc.memory(), alloc.offset())? };
        let address = unsafe { self.device.get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(handle)) };
        self.buffers.push(Some((handle, alloc)));
        Ok(Buffer { handle, size, address, id: self.buffers.len() - 1 })
    }

    /// Host-visible buffer initialised with `data`.
    pub fn create_buffer_init(&mut self, data: &[u8], usage: vk::BufferUsageFlags, name: &str) -> Result<Buffer> {
        let b = self.create_buffer(data.len() as u64, usage, MemoryLocation::CpuToGpu, name)?;
        self.write(&b, 0, data)?;
        Ok(b)
    }

    pub fn write(&mut self, b: &Buffer, offset: usize, data: &[u8]) -> Result<()> {
        let (_, alloc) = self.buffers[b.id].as_mut().ok_or_else(|| anyhow!("buffer destroyed"))?;
        let slice = alloc.mapped_slice_mut().ok_or_else(|| anyhow!("buffer is not host visible"))?;
        slice[offset..offset + data.len()].copy_from_slice(data);
        Ok(())
    }

    pub fn read(&self, b: &Buffer) -> Result<Vec<u8>> {
        let (_, alloc) = self.buffers[b.id].as_ref().ok_or_else(|| anyhow!("buffer destroyed"))?;
        let slice = alloc.mapped_slice().ok_or_else(|| anyhow!("buffer is not host visible"))?;
        Ok(slice[..b.size as usize].to_vec())
    }

    pub fn destroy_buffer(&mut self, b: Buffer) {
        if let Some((handle, alloc)) = self.buffers[b.id].take() {
            unsafe { self.device.destroy_buffer(handle, None) };
            let _ = self.allocator.as_mut().unwrap().free(alloc);
        }
    }

    /// Record commands into a one-shot command buffer, submit, and wait.
    pub fn submit(&self, record: impl FnOnce(vk::CommandBuffer)) -> Result<()> {
        unsafe {
            let cmd = self.device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(self.cmd_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )?[0];
            self.device.begin_command_buffer(
                cmd,
                &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
            record(cmd);
            self.device.end_command_buffer(cmd)?;
            let fence = self.device.create_fence(&vk::FenceCreateInfo::default(), None)?;
            let cmds = [cmd];
            let result = self
                .device
                .queue_submit(self.queue, &[vk::SubmitInfo::default().command_buffers(&cmds)], fence)
                .and_then(|_| self.device.wait_for_fences(&[fence], true, u64::MAX));
            self.device.destroy_fence(fence, None);
            self.device.free_command_buffers(self.cmd_pool, &cmds);
            result.context("GPU submission failed (device lost?)")?;
        }
        Ok(())
    }

    /// Full memory barrier between two command groups.
    pub fn barrier(&self, cmd: vk::CommandBuffer, src: vk::PipelineStageFlags, dst: vk::PipelineStageFlags) {
        let mb = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::MEMORY_WRITE)
            .dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE);
        unsafe {
            self.device
                .cmd_pipeline_barrier(cmd, src, dst, vk::DependencyFlags::empty(), &[mb], &[], &[])
        };
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            for p in self.pipelines.drain(..) {
                p.destroy(&self.device);
            }
            for a in self.accels.drain(..) {
                self.as_loader.destroy_acceleration_structure(a, None);
            }
            for (handle, alloc) in self.buffers.drain(..).flatten() {
                self.device.destroy_buffer(handle, None);
                let _ = self.allocator.as_mut().unwrap().free(alloc);
            }
            self.allocator = None;
            self.device.destroy_command_pool(self.cmd_pool, None);
            self.device.destroy_device(None);
            if let Some((du, m)) = self.debug.take() {
                du.destroy_debug_utils_messenger(m, None);
            }
            self.instance.destroy_instance(None);
        }
    }
}

fn pick_device(instance: &Instance) -> Result<(vk::PhysicalDevice, u32, DeviceInfo)> {
    let wanted = std::env::var("PXL_DEVICE").ok().map(|s| s.to_lowercase());
    let mut best: Option<(i32, vk::PhysicalDevice, u32, DeviceInfo)> = None;
    let mut rejected = Vec::new();
    for pd in unsafe { instance.enumerate_physical_devices()? } {
        let props = unsafe { instance.get_physical_device_properties(pd) };
        let name = props.device_name_as_c_str().map(|c| c.to_string_lossy().into_owned()).unwrap_or_default();
        let exts = unsafe { instance.enumerate_device_extension_properties(pd)? };
        let missing: Vec<_> = REQUIRED_EXTENSIONS
            .iter()
            .filter(|r| !exts.iter().any(|e| e.extension_name_as_c_str().ok() == Some(**r)))
            .map(|r| r.to_string_lossy().into_owned())
            .collect();
        if !missing.is_empty() || vk::api_version_minor(props.api_version) < 3 {
            rejected.push(format!("{name}: missing {}", missing.join(", ")));
            continue;
        }
        if let Some(w) = &wanted {
            if !name.to_lowercase().contains(w) {
                continue;
            }
        }
        let qf = unsafe { instance.get_physical_device_queue_family_properties(pd) }
            .iter()
            .position(|q| q.queue_flags.contains(vk::QueueFlags::COMPUTE))
            .ok_or_else(|| anyhow!("{name} has no compute queue"))? as u32;
        let discrete = props.device_type == vk::PhysicalDeviceType::DISCRETE_GPU;
        let score = if discrete { 10 } else { 1 };
        let info = DeviceInfo {
            name,
            api_version: format!(
                "{}.{}.{}",
                vk::api_version_major(props.api_version),
                vk::api_version_minor(props.api_version),
                vk::api_version_patch(props.api_version)
            ),
            driver_version: props.driver_version,
            discrete,
        };
        if best.as_ref().map_or(true, |b| score > b.0) {
            best = Some((score, pd, qf, info));
        }
    }
    match best {
        Some((_, pd, qf, info)) => Ok((pd, qf, info)),
        None => bail!(
            "no GPU with Vulkan 1.3 + hardware ray queries found{}. Rejected: {}",
            wanted.map(|w| format!(" matching PXL_DEVICE={w}")).unwrap_or_default(),
            if rejected.is_empty() { "none".into() } else { rejected.join("; ") }
        ),
    }
}
