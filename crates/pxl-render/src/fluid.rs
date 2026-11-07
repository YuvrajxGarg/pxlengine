//! Liquids: FLIP/PIC on a GPU MAC grid (shaders/fluid.slang), meshed every frame with
//! surface nets straight into the renderer's vertex/index buffers, then given a fresh BLAS.

use crate::library::GpuMaterial;
use crate::{bytes, Renderer};
use anyhow::{Context as _, Result};
use pxl_gpu::{vk, Accel, Binding, BlasInput, Buffer, ComputePipeline, Context, MemoryLocation, Resource};
use pxl_scene::Fluid;

const FLUID_SRC: &str = include_str!("../../../shaders/fluid.slang");
const KERNELS: [&str; 13] = [
    "clear_grid", "p2g", "normalize_grid", "forces", "compute_divergence", "pressure_rb", "project",
    "g2p_advect", "s_clear", "s_splat", "s_phi", "s_verts", "s_quads",
];
const K_CLEAR: usize = 0;
const K_P2G: usize = 1;
const K_NORMALIZE: usize = 2;
const K_FORCES: usize = 3;
const K_DIV: usize = 4;
const K_PRESSURE: usize = 5;
const K_PROJECT: usize = 6;
const K_G2P: usize = 7;
const K_S_CLEAR: usize = 8;
const K_S_SPLAT: usize = 9;
const K_S_PHI: usize = 10;
const K_S_VERTS: usize = 11;
const K_S_QUADS: usize = 12;

pub(crate) struct FluidKernels {
    pipes: Vec<ComputePipeline>,
}

impl FluidKernels {
    pub(crate) fn new(ctx: &mut Context) -> Result<FluidKernels> {
        let layout: Vec<(u32, Binding)> = (30..=44).map(|b| (b, Binding::Storage)).collect();
        let mut pipes = Vec::new();
        for k in KERNELS {
            let spirv = pxl_gpu::slang::compile_compute(FLUID_SRC, k).with_context(|| format!("compiling fluid kernel {k}"))?;
            pipes.push(ctx.create_compute(&spirv, "main", &layout, 112)?);
        }
        Ok(FluidKernels { pipes })
    }
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct FluidPush {
    nx: u32,
    ny: u32,
    nz: u32,
    n_particles: u32,
    h: f32,
    dt: f32,
    flip: f32,
    inv_h: f32,
    origin: [f32; 3],
    parity: u32,
    gravity: [f32; 3],
    pad0: u32,
    sx: u32,
    sy: u32,
    sz: u32,
    max_verts: u32,
    sh: f32,
    radius: f32,
    iso: f32,
    pad1: f32,
    vbase: u32,
    ibase: u32,
    max_tris: u32,
    pad2: u32,
}

pub(crate) struct FluidState {
    pub name: String,
    push: FluidPush,
    initial: Vec<[f32; 8]>,
    particles: Buffer,
    buffers: Vec<(u32, Buffer)>,
    counters_rb: Buffer,
    substeps: u32,
    warmup: f64,
    fps: f64,
    iterations: u32,
    steps_done: u64,
    initialized: bool,
    pub material: GpuMaterial,
    pub opaque: bool,
    pub vbase: u32,
    pub ibase: u32,
    pub triangles: u32,
    pub blas: Option<Accel>,
    pub overflow: bool,
}

/// Simple deterministic RNG for particle seeding.
fn lcg(s: &mut u64) -> f32 {
    *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    ((*s >> 40) as f32) / (1u64 << 24) as f32
}

impl FluidState {
    /// Vertices / indices this fluid needs reserved in the renderer's buffers.
    pub(crate) fn reservation(f: &Fluid) -> (u32, u32) {
        (f.max_triangles, f.max_triangles * 3)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        ctx: &mut Context,
        f: &Fluid,
        fps: f64,
        vbase: u32,
        ibase: u32,
        material: GpuMaterial,
        opaque: bool,
        seed: u32,
    ) -> Result<FluidState> {
        let ([nx, ny, nz], h) = f.grid();
        let origin = f.bounds[0];
        // Seed particles: `particles_per_cell` per grid cell inside each fill block, kept out of
        // the solid wall layer.
        let mut rng = 0x9e3779b97f4a7c15u64 ^ seed as u64;
        let mut initial: Vec<[f32; 8]> = Vec::new();
        for b in &f.fill {
            for k in 1..nz - 1 {
                for j in 1..ny - 1 {
                    for i in 1..nx - 1 {
                        let c = [
                            origin[0] + (i as f32 + 0.5) * h,
                            origin[1] + (j as f32 + 0.5) * h,
                            origin[2] + (k as f32 + 0.5) * h,
                        ];
                        if (0..3).any(|a| c[a] < b.min[a] || c[a] > b.max[a]) {
                            continue;
                        }
                        for _ in 0..f.particles_per_cell {
                            let p = [
                                origin[0] + (i as f32 + lcg(&mut rng)) * h,
                                origin[1] + (j as f32 + lcg(&mut rng)) * h,
                                origin[2] + (k as f32 + lcg(&mut rng)) * h,
                            ];
                            initial.push([p[0], p[1], p[2], 0.0, b.velocity[0], b.velocity[1], b.velocity[2], 0.0]);
                        }
                    }
                }
            }
        }
        anyhow::ensure!(!initial.is_empty(), "fluid '{}': fill blocks contain no grid cells", f.name);

        let r = f.surface_refine;
        let (sx, sy, sz) = (nx * r, ny * r, nz * r);
        let sh = h / r as f32;
        let radius = f.particle_radius * sh;
        // Iso level: half the expected interior density of the poly6-style kernel.
        let kernel_volume = 0.638 * (radius / h).powi(3);
        let iso = 0.5 * f.particles_per_cell as f32 * kernel_volume;
        let push = FluidPush {
            nx,
            ny,
            nz,
            n_particles: initial.len() as u32,
            h,
            dt: (1.0 / (fps * f.substeps as f64)) as f32,
            flip: f.flip,
            inv_h: 1.0 / h,
            origin,
            gravity: f.gravity,
            sx,
            sy,
            sz,
            max_verts: f.max_triangles,
            sh,
            radius,
            iso,
            vbase,
            ibase,
            max_tris: f.max_triangles,
            ..Default::default()
        };

        let faces = ((nx + 1) * ny * nz + nx * (ny + 1) * nz + nx * ny * (nz + 1)) as u64;
        let cells = (nx * ny * nz) as u64;
        let snodes = ((sx + 1) * (sy + 1) * (sz + 1)) as u64;
        let scells = (sx * sy * sz) as u64;
        let storage = vk::BufferUsageFlags::STORAGE_BUFFER;
        let src = vk::BufferUsageFlags::TRANSFER_SRC;
        let particles = ctx.create_buffer_init(bytes(&initial), storage, "fluid particles")?;
        let mut buffers = vec![(30, particles)];
        for (binding, n, name) in [
            (31u32, faces, "fluid vel"),
            (32, faces, "fluid vel old"),
            (33, faces, "fluid acc"),
            (34, faces, "fluid weights"),
            (35, cells, "fluid count"),
            (36, cells, "fluid type"),
            (37, cells, "fluid pressure"),
            (38, cells, "fluid divergence"),
            (39, snodes, "fluid density"),
            (40, snodes, "fluid phi"),
            (41, scells, "fluid cell verts"),
            (42, 4, "fluid counters"),
        ] {
            buffers.push((binding, ctx.create_buffer(n * 4, storage | src, MemoryLocation::GpuOnly, name)?));
        }
        let counters_rb = ctx.create_buffer(16, vk::BufferUsageFlags::TRANSFER_DST, MemoryLocation::GpuToCpu, "fluid counters rb")?;
        Ok(FluidState {
            name: f.name.clone(),
            push,
            initial,
            particles,
            buffers,
            counters_rb,
            substeps: f.substeps,
            warmup: f.warmup as f64,
            fps,
            iterations: f.pressure_iterations,
            steps_done: 0,
            initialized: false,
            material,
            opaque,
            vbase,
            ibase,
            triangles: 0,
            blas: None,
            overflow: false,
        })
    }

    pub(crate) fn particle_count(&self) -> u32 {
        self.push.n_particles
    }

    fn steps_for_frame(&self, frame: f64) -> u64 {
        (((self.warmup + frame / self.fps) * self.fps * self.substeps as f64).round()).max(0.0) as u64
    }

    fn counts(&self) -> (u32, u32, u32, u32, u32) {
        let p = &self.push;
        let faces = (p.nx + 1) * p.ny * p.nz + p.nx * (p.ny + 1) * p.nz + p.nx * p.ny * (p.nz + 1);
        let cells = p.nx * p.ny * p.nz;
        let snodes = (p.sx + 1) * (p.sy + 1) * (p.sz + 1);
        let scells = p.sx * p.sy * p.sz;
        (faces, cells, snodes, scells, p.n_particles)
    }
}

fn g(n: u32) -> [u32; 3] {
    [n.div_ceil(256), 1, 1]
}

impl Renderer {
    fn fluid_bind(&mut self, fi: usize, vbuf: Buffer, ibuf: Buffer) {
        let kernels = self.fluid_kernels.as_ref().expect("fluid kernels");
        let st = &self.fluids[fi];
        let mut res: Vec<(u32, Resource)> = st.buffers.iter().map(|&(b, buf)| (b, Resource::Buffer(buf))).collect();
        res.push((43, Resource::Buffer(vbuf)));
        res.push((44, Resource::Buffer(ibuf)));
        for p in &kernels.pipes {
            self.ctx.bind(p, &res);
        }
    }

    /// Step fluid `fi` to scene frame `frame`, then rebuild its surface mesh and BLAS.
    pub(crate) fn fluid_advance(&mut self, fi: usize, frame: f64, vbuf: Buffer, ibuf: Buffer) -> Result<()> {
        self.fluid_bind(fi, vbuf, ibuf);
        let kernels = self.fluid_kernels.as_ref().expect("fluid kernels");
        let ctx = &mut self.ctx;
        let st = &mut self.fluids[fi];
        let target = st.steps_for_frame(frame);
        if !st.initialized || target < st.steps_done {
            ctx.write(&st.particles, 0, bytes(&st.initial))?;
            st.steps_done = 0;
            st.initialized = true;
        }
        let (faces, cells, snodes, scells, parts) = st.counts();
        let fc = faces.max(cells);
        while st.steps_done < target {
            let end = (st.steps_done + 4).min(target);
            let n = end - st.steps_done;
            let c = &*ctx;
            let s = &*st;
            c.submit(|cmd| {
                let mut push = s.push;
                let run = |k: usize, push: &FluidPush, groups: [u32; 3]| {
                    c.dispatch(cmd, &kernels.pipes[k], bytes(std::slice::from_ref(push)), groups);
                    c.barrier_all(cmd);
                };
                for _ in 0..n {
                    run(K_CLEAR, &push, g(fc));
                    run(K_P2G, &push, g(parts));
                    run(K_NORMALIZE, &push, g(fc));
                    run(K_FORCES, &push, g(faces));
                    run(K_DIV, &push, g(cells));
                    for _ in 0..s.iterations {
                        for parity in 0..2 {
                            push.parity = parity;
                            run(K_PRESSURE, &push, g(cells));
                        }
                    }
                    run(K_PROJECT, &push, g(faces));
                    run(K_G2P, &push, g(parts));
                }
            })?;
            st.steps_done = end;
        }

        // Surface: density -> phi -> vertices -> quads, then read the counts back.
        let counters = st.buffers.iter().find(|(b, _)| *b == 42).map(|(_, b)| *b).expect("counters");
        let rb = st.counters_rb;
        let c = &*ctx;
        let s = &*st;
        c.submit(|cmd| unsafe {
            let run = |k: usize, groups: [u32; 3]| {
                c.dispatch(cmd, &kernels.pipes[k], bytes(std::slice::from_ref(&s.push)), groups);
                c.barrier_all(cmd);
            };
            run(K_S_CLEAR, g(snodes.max(scells)));
            run(K_S_SPLAT, g(parts));
            run(K_S_PHI, g(snodes));
            run(K_S_VERTS, g(scells));
            run(K_S_QUADS, g(snodes));
            c.device.cmd_copy_buffer(cmd, counters.handle, rb.handle, &[vk::BufferCopy { src_offset: 0, dst_offset: 0, size: 8 }]);
        })?;
        let raw = ctx.read(&rb)?;
        let verts = u32::from_le_bytes(raw[0..4].try_into().unwrap());
        let tris = u32::from_le_bytes(raw[4..8].try_into().unwrap());
        st.overflow = verts > st.push.max_verts || tris > st.push.max_tris;
        let verts = verts.min(st.push.max_verts);
        let tris = tris.min(st.push.max_tris) / 2 * 2;
        if let Some(old) = st.blas.take() {
            ctx.destroy_accel(old);
        }
        st.triangles = tris;
        if tris > 0 {
            st.blas = Some(ctx.build_blas(&BlasInput {
                vertex_address: vbuf.address + st.vbase as u64 * 32,
                vertex_stride: 32,
                vertex_count: verts.max(1),
                index_address: ibuf.address + st.ibase as u64 * 4,
                triangle_count: tris,
                opaque: st.opaque,
            })?);
        }
        Ok(())
    }
}
