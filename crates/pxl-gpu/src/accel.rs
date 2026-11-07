//! Bottom/top-level acceleration structures for inline ray queries.

use crate::{Buffer, Context, MemoryLocation};
use anyhow::Result;
use ash::vk;

/// Triangle geometry already resident in GPU buffers (addresses include offsets).
#[derive(Debug, Clone, Copy)]
pub struct BlasInput {
    pub vertex_address: u64,
    pub vertex_stride: u64,
    pub vertex_count: u32,
    pub index_address: u64,
    pub triangle_count: u32,
    /// false for alpha-masked geometry: ray queries then report candidates for alpha testing.
    pub opaque: bool,
}

/// An acceleration structure plus its backing storage.
#[derive(Debug, Clone, Copy)]
pub struct Accel {
    pub handle: vk::AccelerationStructureKHR,
    pub address: u64,
    storage: Buffer,
    id: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct TlasInstance {
    /// Row-major 3x4 object-to-world transform.
    pub transform: [f32; 12],
    pub custom_index: u32,
    pub mask: u8,
    pub blas_address: u64,
}

impl Context {
    fn build_accel(
        &mut self,
        ty: vk::AccelerationStructureTypeKHR,
        geometry: vk::AccelerationStructureGeometryKHR<'_>,
        primitive_count: u32,
    ) -> Result<Accel> {
        let geoms = [geometry];
        let mut build = vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .ty(ty)
            .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .geometries(&geoms);
        let mut sizes = vk::AccelerationStructureBuildSizesInfoKHR::default();
        unsafe {
            self.as_loader.get_acceleration_structure_build_sizes(
                vk::AccelerationStructureBuildTypeKHR::DEVICE,
                &build,
                &[primitive_count],
                &mut sizes,
            )
        };
        let storage = self.create_buffer(
            sizes.acceleration_structure_size,
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR,
            MemoryLocation::GpuOnly,
            "accel storage",
        )?;
        let accel = unsafe {
            self.as_loader.create_acceleration_structure(
                &vk::AccelerationStructureCreateInfoKHR::default()
                    .buffer(storage.handle)
                    .size(sizes.acceleration_structure_size)
                    .ty(ty),
                None,
            )?
        };
        self.accels.push(Some(accel));
        let id = self.accels.len() - 1;
        let align = self.scratch_align;
        let scratch = self.create_buffer(
            sizes.build_scratch_size + align,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            MemoryLocation::GpuOnly,
            "accel scratch",
        )?;
        let scratch_addr = (scratch.address + align - 1) & !(align - 1);
        build = build
            .dst_acceleration_structure(accel)
            .scratch_data(vk::DeviceOrHostAddressKHR { device_address: scratch_addr });
        let ranges = [vk::AccelerationStructureBuildRangeInfoKHR::default().primitive_count(primitive_count)];
        let loader = &self.as_loader;
        self.submit(|cmd| unsafe {
            loader.cmd_build_acceleration_structures(cmd, std::slice::from_ref(&build), &[&ranges[..]])
        })?;
        self.destroy_buffer(scratch);
        let address = unsafe {
            self.as_loader.get_acceleration_structure_device_address(
                &vk::AccelerationStructureDeviceAddressInfoKHR::default().acceleration_structure(accel),
            )
        };
        Ok(Accel { handle: accel, address, storage, id })
    }

    pub fn destroy_accel(&mut self, a: Accel) {
        if let Some(h) = self.accels[a.id].take() {
            unsafe { self.as_loader.destroy_acceleration_structure(h, None) };
            self.destroy_buffer(a.storage);
        }
    }

    pub fn build_blas(&mut self, input: &BlasInput) -> Result<Accel> {
        let tri = vk::AccelerationStructureGeometryTrianglesDataKHR::default()
            .vertex_format(vk::Format::R32G32B32_SFLOAT)
            .vertex_data(vk::DeviceOrHostAddressConstKHR { device_address: input.vertex_address })
            .vertex_stride(input.vertex_stride)
            .max_vertex(input.vertex_count.saturating_sub(1))
            .index_type(vk::IndexType::UINT32)
            .index_data(vk::DeviceOrHostAddressConstKHR { device_address: input.index_address });
        let geom = vk::AccelerationStructureGeometryKHR::default()
            .geometry_type(vk::GeometryTypeKHR::TRIANGLES)
            .geometry(vk::AccelerationStructureGeometryDataKHR { triangles: tri })
            .flags(if input.opaque { vk::GeometryFlagsKHR::OPAQUE } else { vk::GeometryFlagsKHR::NO_DUPLICATE_ANY_HIT_INVOCATION });
        self.build_accel(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL, geom, input.triangle_count)
    }

    pub fn build_tlas(&mut self, instances: &[TlasInstance]) -> Result<Accel> {
        let raw: Vec<vk::AccelerationStructureInstanceKHR> = instances
            .iter()
            .map(|i| vk::AccelerationStructureInstanceKHR {
                transform: vk::TransformMatrixKHR { matrix: i.transform },
                instance_custom_index_and_mask: vk::Packed24_8::new(i.custom_index, i.mask),
                instance_shader_binding_table_record_offset_and_flags: vk::Packed24_8::new(
                    0,
                    vk::GeometryInstanceFlagsKHR::TRIANGLE_FACING_CULL_DISABLE.as_raw() as u8,
                ),
                acceleration_structure_reference: vk::AccelerationStructureReferenceKHR {
                    device_handle: i.blas_address,
                },
            })
            .collect();
        let bytes = unsafe {
            std::slice::from_raw_parts(raw.as_ptr() as *const u8, std::mem::size_of_val(raw.as_slice()))
        };
        let buf = self.create_buffer(
            bytes.len().max(64) as u64,
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
            MemoryLocation::CpuToGpu,
            "tlas instances",
        )?;
        self.write(&buf, 0, bytes)?;
        let geom = vk::AccelerationStructureGeometryKHR::default()
            .geometry_type(vk::GeometryTypeKHR::INSTANCES)
            .geometry(vk::AccelerationStructureGeometryDataKHR {
                instances: vk::AccelerationStructureGeometryInstancesDataKHR::default()
                    .array_of_pointers(false)
                    .data(vk::DeviceOrHostAddressConstKHR { device_address: buf.address }),
            });
        let tlas = self.build_accel(vk::AccelerationStructureTypeKHR::TOP_LEVEL, geom, instances.len() as u32);
        self.destroy_buffer(buf);
        tlas
    }
}
