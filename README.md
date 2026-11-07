# pxlengine

A headless 3D + VFX engine for agents, built in Rust + Slang on Vulkan hardware ray tracing.

An agent writes a scene as JSON, runs `pxl render`, inspects the PNG, and revises.
No editor, no Blender, no UI.

```
pxl help                       # agent docs (start here)
pxl info                       # GPU / slangc / cache
pxl render examples/first-scene.json --profile std/final --out out/first.png
```

## SDKs, tools, benchmarks

- `packages/python` — Python SDK (`from pxl import Scene, mesh, sphere ...`; `scene.render(...)`)
- `packages/ts` — TypeScript SDK with scene types generated from `pxl schema`
- `pxl check scene.json`, `pxl help field camera.fov`, `pxl schema`, `pxl track`
- `integrations/blender/import_pxl.py` — open a pxl scene in Blender
- `benchmarks/` — pxl vs Blender Cycles/EEVEE timings and method

## Build

Requirements: Rust (MSVC on Windows), the Vulkan SDK (for `slangc`), and a GPU with
`VK_KHR_ray_query` (NVIDIA RTX 20xx+, AMD RDNA2+, Intel Arc).

```
cargo build --release
./target/release/pxl info
```

## Layout

- `crates/pxl-scene` — scene JSON schema, std geometry, animation tracks
- `crates/pxl-assets` — glTF 2.0, texture and HDRI loading
- `crates/pxl-denoise` — Intel OIDN, loaded at runtime
- `crates/pxl-gpu` — Vulkan device, buffers, BLAS/TLAS, compute pipelines, Slang compiler cache
- `crates/pxl-render` — scene → GPU, progressive path tracing
- `crates/pxl-cli` — the `pxl` binary
- `shaders/` — Slang sources (embedded into the binary)
- `docs/help/` — agent-facing help topics (embedded)
- `docs/ROADMAP.md` — phased plan and known gaps

Status: all roadmap phases (1-7) complete (GPU path tracer; glTF/texture/HDRI import, animation, video;
OIDN denoising, depth of field, motion blur, compositing passes, EXR; scene Slang programs
for materials, cameras, light propagation and film - see `examples/black-hole.json`;
GPU agent simulations with neighbour search - see `examples/birds-1m.json`, a million boids;
CG over footage with tracked cameras, shadow catchers and holdouts - see `examples/vfx/`;
glass/water materials and FLIP liquid simulation - see `examples/dam-break.json`).
See the roadmap. Test assets: `sh scripts/fetch-test-assets.sh` (Khronos + Poly Haven).
Denoiser: `sh scripts/fetch-oidn.sh` (Intel Open Image Denoise into `vendor/oidn`).
