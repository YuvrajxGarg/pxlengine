//! Compute pipelines with a single descriptor set and a push-constant block.

use crate::{Buffer, Context};
use anyhow::{Context as _, Result};
use ash::vk;

#[derive(Debug, Clone, Copy)]
pub enum Binding {
    Accel,
    Storage,
}

#[derive(Debug, Clone, Copy)]
pub enum Resource {
    Accel(vk::AccelerationStructureKHR),
    Buffer(Buffer),
}

#[derive(Debug, Clone, Copy)]
pub struct ComputePipeline {
    pub pipeline: vk::Pipeline,
    pub layout: vk::PipelineLayout,
    pub set_layout: vk::DescriptorSetLayout,
    pub pool: vk::DescriptorPool,
    pub set: vk::DescriptorSet,
    pub push_size: u32,
}

impl ComputePipeline {
    pub(crate) fn destroy(&self, d: &ash::Device) {
        unsafe {
            d.destroy_pipeline(self.pipeline, None);
            d.destroy_pipeline_layout(self.layout, None);
            d.destroy_descriptor_pool(self.pool, None);
            d.destroy_descriptor_set_layout(self.set_layout, None);
        }
    }
}

impl Context {
    /// `bindings` are (binding number, kind) in descriptor set 0.
    pub fn create_compute(&mut self, spirv: &[u8], entry: &str, bindings: &[(u32, Binding)], push_size: u32) -> Result<ComputePipeline> {
        let d = &self.device;
        let words = ash::util::read_spv(&mut std::io::Cursor::new(spirv)).context("invalid SPIR-V")?;
        let layout_bindings: Vec<_> = bindings
            .iter()
            .map(|(i, b)| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(*i)
                    .descriptor_type(match b {
                        Binding::Accel => vk::DescriptorType::ACCELERATION_STRUCTURE_KHR,
                        Binding::Storage => vk::DescriptorType::STORAGE_BUFFER,
                    })
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
            })
            .collect();
        let entry_c = std::ffi::CString::new(entry)?;
        unsafe {
            let set_layout = d.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&layout_bindings),
                None,
            )?;
            let ranges = [vk::PushConstantRange::default()
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .size(push_size.max(4))];
            let set_layouts = [set_layout];
            let layout = d.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&set_layouts)
                    .push_constant_ranges(if push_size > 0 { &ranges[..] } else { &[] }),
                None,
            )?;
            let module = d.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&words), None)?;
            let stage = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(module)
                .name(&entry_c);
            let pipeline = d
                .create_compute_pipelines(
                    vk::PipelineCache::null(),
                    &[vk::ComputePipelineCreateInfo::default().stage(stage).layout(layout)],
                    None,
                )
                .map_err(|(_, e)| e)
                .context("creating compute pipeline")?[0];
            d.destroy_shader_module(module, None);

            let sizes = [
                vk::DescriptorPoolSize { ty: vk::DescriptorType::ACCELERATION_STRUCTURE_KHR, descriptor_count: 4 },
                vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_BUFFER, descriptor_count: 32 },
            ];
            let pool = d.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default().max_sets(1).pool_sizes(&sizes),
                None,
            )?;
            let set = d.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(&set_layouts),
            )?[0];
            let p = ComputePipeline { pipeline, layout, set_layout, pool, set, push_size };
            self.pipelines.push(p);
            Ok(p)
        }
    }

    /// Bind (binding number, resource) pairs in the pipeline's set.
    pub fn bind(&self, p: &ComputePipeline, resources: &[(u32, Resource)]) {
        for &(i, r) in resources {
            match r {
                Resource::Accel(a) => {
                    let accels = [a];
                    let mut w_as = vk::WriteDescriptorSetAccelerationStructureKHR::default().acceleration_structures(&accels);
                    let w = vk::WriteDescriptorSet::default()
                        .dst_set(p.set)
                        .dst_binding(i)
                        .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
                        .descriptor_count(1)
                        .push_next(&mut w_as);
                    unsafe { self.device.update_descriptor_sets(&[w], &[]) };
                }
                Resource::Buffer(b) => {
                    let info = [vk::DescriptorBufferInfo::default().buffer(b.handle).range(vk::WHOLE_SIZE)];
                    let w = vk::WriteDescriptorSet::default()
                        .dst_set(p.set)
                        .dst_binding(i)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(&info);
                    unsafe { self.device.update_descriptor_sets(&[w], &[]) };
                }
            }
        }
    }

    pub fn dispatch(&self, cmd: vk::CommandBuffer, p: &ComputePipeline, push: &[u8], groups: [u32; 3]) {
        unsafe {
            self.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, p.pipeline);
            self.device
                .cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, p.layout, 0, &[p.set], &[]);
            if !push.is_empty() {
                self.device
                    .cmd_push_constants(cmd, p.layout, vk::ShaderStageFlags::COMPUTE, 0, push);
            }
            self.device.cmd_dispatch(cmd, groups[0], groups[1], groups[2]);
        }
    }
}
