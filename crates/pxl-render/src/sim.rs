//! GPU simulations: agent buffers, the per-step neighbour grid, scene init/step programs,
//! and emission of every agent as a ray-tracing instance.

use crate::library::{GpuMaterial, Library};
use crate::{bytes, Renderer};
use anyhow::{Context as _, Result};
use pxl_gpu::{vk, Binding, Buffer, ComputePipeline, Context, MemoryLocation, Resource};
use pxl_scene::Simulation;

const SIM_SRC: &str = include_str!("../../../shaders/sim.slang");
const AGENT_BYTES: u64 = 64;

pub(crate) struct SimKernels {
    clear: ComputePipeline,
    count: ComputePipeline,
    scan_local: ComputePipeline,
    scan_blocks: ComputePipeline,
    scan_add: ComputePipeline,
    scatter: ComputePipeline,
    emit: ComputePipeline,
    init: ComputePipeline,
    step: ComputePipeline,
}

impl SimKernels {
    /// `program_source` is the assembled scene program (with `sim_init` / `sim_step`).
    pub(crate) fn new(ctx: &mut Context, program_source: &str) -> Result<SimKernels> {
        let fixed: Vec<(u32, Binding)> = [20u32, 22, 23, 24, 25, 26, 27, 28].iter().map(|&b| (b, Binding::Storage)).collect();
        let mut k = |entry: &str| -> Result<ComputePipeline> {
            let spirv = pxl_gpu::slang::compile_compute(SIM_SRC, entry).with_context(|| format!("compiling sim kernel {entry}"))?;
            ctx.create_compute(&spirv, "main", &fixed, 64)
        };
        let (clear, count, scan_local, scan_blocks, scan_add, scatter, emit) = (
            k("grid_clear")?,
            k("grid_count")?,
            k("scan_local")?,
            k("scan_blocks")?,
            k("scan_add")?,
            k("grid_scatter")?,
            k("emit")?,
        );
        let user: Vec<(u32, Binding)> = [17u32, 20, 21, 22, 23, 24].iter().map(|&b| (b, Binding::Storage)).collect();
        let mut u = |entry: &str| -> Result<ComputePipeline> {
            let spirv = pxl_gpu::slang::compile_compute(program_source, entry)
                .with_context(|| format!("compiling simulation programs ({entry})"))?;
            ctx.create_compute(&spirv, "main", &user, 32)
        };
        let (init, step) = (u("sim_init")?, u("sim_step")?);
        Ok(SimKernels { clear, count, scan_local, scan_blocks, scan_add, scatter, emit, init, step })
    }
}

pub(crate) struct SimState {
    pub name: String,
    pub index: u32,
    pub count: u32,
    pub tris_per_agent: usize,
    table_size: u32,
    n_blocks: u32,
    cell_size: f32,
    substeps: u32,
    warmup: f64,
    fps: f64,
    seed: u32,
    agents_in: Buffer,
    agents_out: Buffer,
    offsets: Buffer,
    counts: Buffer,
    sorted: Buffer,
    cursor: Buffer,
    block_sums: Buffer,
    pub mesh: usize,
    pub material: GpuMaterial,
    scale: f32,
    orient: bool,
    object_id: u32,
    steps_done: u64,
    initialized: bool,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct SimInfo {
    count: u32,
    index: u32,
    dt: f32,
    time: f32,
    cell_size: f32,
    table_size: u32,
    step: u32,
    seed: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct SimPush {
    count: u32,
    table_size: u32,
    cell_size: f32,
    n_blocks: u32,
    base_instance: u32,
    material: u32,
    vertex_offset: u32,
    index_offset: u32,
    blas_lo: u32,
    blas_hi: u32,
    scale: f32,
    time_offset: f32,
    object_id: u32,
    orient: u32,
    pad: [u32; 2],
}

fn as_bytes<T: Copy>(v: &T) -> &[u8] {
    bytes(std::slice::from_ref(v))
}

impl SimState {
    pub(crate) fn new(
        ctx: &mut Context,
        lib: &mut Library,
        sim: &Simulation,
        index: u32,
        object_id: u32,
        fps: f64,
        seed: u32,
    ) -> Result<SimState> {
        let (mesh, material) = lib.sim_mesh(sim).with_context(|| format!("simulation '{}' mesh", sim.name))?;
        let table_size = (sim.count as u64 * 2).next_power_of_two().clamp(1024, 1 << 23) as u32;
        let storage = vk::BufferUsageFlags::STORAGE_BUFFER;
        let copy = vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST;
        let mut buf = |n: u64, name: &str| ctx.create_buffer(n, storage | copy, MemoryLocation::GpuOnly, name);
        let agents_in = buf(sim.count as u64 * AGENT_BYTES, "agents")?;
        let agents_out = buf(sim.count as u64 * AGENT_BYTES, "agents next")?;
        let offsets = buf(table_size as u64 * 4, "grid offsets")?;
        let counts = buf(table_size as u64 * 4, "grid counts")?;
        let sorted = buf(sim.count as u64 * 4, "grid sorted")?;
        let cursor = buf(table_size as u64 * 4, "grid cursor")?;
        let block_sums = buf((table_size as u64 / 1024).max(1) * 4, "grid blocks")?;
        Ok(SimState {
            name: sim.name.clone(),
            index,
            count: sim.count,
            tris_per_agent: lib.meshes[mesh].data.triangle_count(),
            table_size,
            n_blocks: table_size / 1024,
            cell_size: sim.cell_size,
            substeps: sim.substeps,
            warmup: sim.warmup as f64,
            fps,
            seed,
            agents_in,
            agents_out,
            offsets,
            counts,
            sorted,
            cursor,
            block_sums,
            mesh,
            material,
            scale: sim.scale,
            orient: sim.orient,
            object_id,
            steps_done: 0,
            initialized: false,
        })
    }

    fn dt(&self) -> f64 {
        1.0 / (self.fps * self.substeps as f64)
    }

    /// Steps needed so the simulation clock reaches scene time `frame / fps`.
    pub(crate) fn steps_for_frame(&self, frame: f64) -> u64 {
        (((self.warmup + frame / self.fps) * self.fps * self.substeps as f64).round()).max(0.0) as u64
    }

    fn grid_push(&self) -> SimPush {
        SimPush {
            count: self.count,
            table_size: self.table_size,
            cell_size: self.cell_size.max(1e-6),
            n_blocks: self.n_blocks,
            ..Default::default()
        }
    }
}

impl Renderer {
    /// Advance simulation `si` to the state for scene frame `frame` (re-initialising if the
    /// requested frame is earlier than the current state).
    pub(crate) fn sim_advance(&mut self, si: usize, frame: f64, params: &[[f32; 4]]) -> Result<()> {
        let kernels = self.sim_kernels.as_ref().expect("simulation kernels");
        let ctx = &mut self.ctx;
        let st = &mut self.sims[si];
        let target = st.steps_for_frame(frame);
        let par = ctx.create_buffer_init(bytes(params), vk::BufferUsageFlags::STORAGE_BUFFER, "sim params")?;
        let groups = st.count.div_ceil(256);
        let copy_size = st.count as u64 * AGENT_BYTES;

        if !st.initialized || target < st.steps_done {
            ctx.bind(&kernels.init, &[(17, Resource::Buffer(par)), (21, Resource::Buffer(st.agents_out))]);
            let info = SimInfo {
                count: st.count,
                index: st.index,
                dt: st.dt() as f32,
                time: -st.warmup as f32,
                cell_size: st.cell_size,
                table_size: st.table_size,
                step: 0,
                seed: st.seed,
            };
            let c = &*ctx;
            let (src, dst) = (st.agents_out.handle, st.agents_in.handle);
            c.submit(|cmd| unsafe {
                c.dispatch(cmd, &kernels.init, as_bytes(&info), [groups, 1, 1]);
                c.barrier_all(cmd);
                c.device.cmd_copy_buffer(cmd, src, dst, &[vk::BufferCopy { src_offset: 0, dst_offset: 0, size: copy_size }]);
            })?;
            st.steps_done = 0;
            st.initialized = true;
        }

        if st.steps_done < target {
            let grid = [
                (20, Resource::Buffer(st.agents_in)),
                (22, Resource::Buffer(st.offsets)),
                (23, Resource::Buffer(st.counts)),
                (24, Resource::Buffer(st.sorted)),
                (25, Resource::Buffer(st.cursor)),
                (26, Resource::Buffer(st.block_sums)),
            ];
            for p in [&kernels.clear, &kernels.count, &kernels.scan_local, &kernels.scan_blocks, &kernels.scan_add, &kernels.scatter] {
                ctx.bind(p, &grid);
            }
            ctx.bind(
                &kernels.step,
                &[
                    (17, Resource::Buffer(par)),
                    (20, Resource::Buffer(st.agents_in)),
                    (21, Resource::Buffer(st.agents_out)),
                    (22, Resource::Buffer(st.offsets)),
                    (23, Resource::Buffer(st.counts)),
                    (24, Resource::Buffer(st.sorted)),
                ],
            );
            let gp = st.grid_push();
            let gpb = as_bytes(&gp);
            let table_groups = st.table_size.div_ceil(256);
            // Keep each submission short (OS GPU watchdog): scale steps per batch with agent count.
            let batch = (8_000_000 / st.count.max(1) as u64).clamp(1, 64);
            while st.steps_done < target {
                let end = (st.steps_done + batch).min(target);
                let c = &*ctx;
                let (src, dst) = (st.agents_out.handle, st.agents_in.handle);
                let s = &*st;
                c.submit(|cmd| unsafe {
                    for step in s.steps_done..end {
                        if s.cell_size > 0.0 {
                            c.dispatch(cmd, &kernels.clear, gpb, [table_groups, 1, 1]);
                            c.barrier_all(cmd);
                            c.dispatch(cmd, &kernels.count, gpb, [groups, 1, 1]);
                            c.barrier_all(cmd);
                            c.dispatch(cmd, &kernels.scan_local, gpb, [s.n_blocks.max(1), 1, 1]);
                            c.barrier_all(cmd);
                            c.dispatch(cmd, &kernels.scan_blocks, gpb, [1, 1, 1]);
                            c.barrier_all(cmd);
                            c.dispatch(cmd, &kernels.scan_add, gpb, [s.n_blocks.max(1), 1, 1]);
                            c.barrier_all(cmd);
                            c.dispatch(cmd, &kernels.scatter, gpb, [groups, 1, 1]);
                            c.barrier_all(cmd);
                        }
                        let info = SimInfo {
                            count: s.count,
                            index: s.index,
                            dt: s.dt() as f32,
                            time: (step as f64 * s.dt() - s.warmup) as f32,
                            cell_size: s.cell_size,
                            table_size: s.table_size,
                            step: step as u32,
                            seed: s.seed,
                        };
                        c.dispatch(cmd, &kernels.step, as_bytes(&info), [groups, 1, 1]);
                        c.barrier_all(cmd);
                        c.device.cmd_copy_buffer(cmd, src, dst, &[vk::BufferCopy { src_offset: 0, dst_offset: 0, size: copy_size }]);
                        c.barrier_all(cmd);
                    }
                })?;
                st.steps_done = end;
            }
        }
        ctx.destroy_buffer(par);
        Ok(())
    }

    /// Write instance records for every agent of simulation `si` into `desc` (TLAS) and
    /// `inst` (renderer), starting at `base`. `time_offset` (s) extrapolates along velocity
    /// for motion blur.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn sim_emit(
        &mut self,
        si: usize,
        desc: Buffer,
        inst: Buffer,
        base: u32,
        material: u32,
        lib: &Library,
        time_offset: f32,
    ) -> Result<()> {
        let kernels = self.sim_kernels.as_ref().expect("simulation kernels");
        let st = &self.sims[si];
        let mesh = &lib.meshes[st.mesh];
        let blas = mesh.blas.expect("uploaded").address;
        let push = SimPush {
            count: st.count,
            base_instance: base,
            material,
            vertex_offset: mesh.vertex_offset,
            index_offset: mesh.index_offset,
            blas_lo: blas as u32,
            blas_hi: (blas >> 32) as u32,
            scale: st.scale,
            time_offset,
            object_id: st.object_id,
            orient: st.orient as u32,
            ..Default::default()
        };
        self.ctx.bind(
            &kernels.emit,
            &[(20, Resource::Buffer(st.agents_in)), (27, Resource::Buffer(desc)), (28, Resource::Buffer(inst))],
        );
        let c = &self.ctx;
        c.submit(|cmd| c.dispatch(cmd, &kernels.emit, as_bytes(&push), [st.count.div_ceil(256), 1, 1]))
    }
}
