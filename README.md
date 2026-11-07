# pxlengine

A headless 3D + VFX engine for agents, built in Rust + Slang on Vulkan hardware ray tracing.

An agent writes a scene as JSON, runs `pxl render`, inspects the PNG, and revises.
No editor, no Blender, no UI.

```
pxl help                       # agent docs (start here)
pxl info                       # GPU / slangc / cache
pxl render examples/first-scene.json --profile std/final --out out/first.png
```

## Build

Requirements: Rust (MSVC on Windows), the Vulkan SDK (for `slangc`), and a GPU with
`VK_KHR_ray_query` (NVIDIA RTX 20xx+, AMD RDNA2+, Intel Arc).

```
cargo build --release
./target/release/pxl info
```

## Layout

- `crates/pxl-scene` — scene JSON schema + std geometry
- `crates/pxl-gpu` — Vulkan device, buffers, BLAS/TLAS, compute pipelines, Slang compiler cache
- `crates/pxl-render` — scene → GPU, progressive path tracing
- `crates/pxl-cli` — the `pxl` binary
- `shaders/` — Slang sources (embedded into the binary)
- `docs/help/` — agent-facing help topics (embedded)
- `docs/ROADMAP.md` — phased plan and known gaps

Status: phase 1 (core GPU path tracer) complete. See the roadmap.
