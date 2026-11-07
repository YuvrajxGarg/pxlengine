# pxlengine roadmap

pxlengine (`pxl`) is a headless 3D + VFX engine for agents.
An agent writes a scene as JSON
(plus optional Slang programs), runs `pxl render`, looks at the image, and revises.
There is no editor.

## Stack

| Layer | Choice |
|---|---|
| Host language | Rust (workspace of crates) |
| GPU API | Vulkan 1.3 via `ash`, memory via `gpu-allocator` |
| Ray tracing | `VK_KHR_acceleration_structure` + `VK_KHR_ray_query` (inline ray queries from compute) |
| Shading language | Slang, compiled to SPIR-V by `slangc` (Vulkan SDK) |
| Video | ffmpeg (NVENC when available) |
| Denoise | Intel OIDN (phase 3) |

## Crates

- `pxl-scene` – scene JSON schema (serde, strict: unknown fields are errors), std geometry, transforms.
- `pxl-gpu` – Vulkan context, buffers, acceleration structures, compute pipelines, Slang compiler + cache.
- `pxl-render` – turns a `Scene` into GPU data and renders it.
- `pxl-cli` – the `pxl` binary: `render`, `help`, `info`.
- `shaders/` – Slang sources embedded into the binary.

## Phases

- [x] **0. Toolchain** – Rust, Vulkan SDK (slangc, validation layers), RT-capable GPU check.
- [x] **1. Core path tracer** (done 2026-11-07; 640x480 @ 256 spp ≈ 35 ms on RTX 4080 Laptop) – Vulkan device with ray query, BLAS/TLAS for std primitives
      (sphere, grid, box), OpenPBR-named material inputs, point/rectangle/directional lights,
      look-at camera, progressive path tracing in a Slang compute shader, PNG out, `pxl help`.
- [ ] **2. Assets + animation + video** – glTF import (meshes, PBR textures), keyframed
      animation tracks on any node field, frame ranges, MP4 via ffmpeg, emissive meshes as lights
      (NEE on emissive triangles — today they are only found by BSDF bounces, so they're noisy),
      HDRI environment maps, AgX tone mapping.
- [ ] **3. Hybrid renderer + profiles** – raster G-buffer visibility + ray-query lighting
      (`std/previs`, `std/draft`, `std/final`) and full path transport (`std/path`); OIDN denoise;
      depth of field, motion blur; AOV/control passes (depth, normal, albedo, mattes).
- [ ] **4. Program hooks** – scene-supplied Slang for camera rays, material response, light
      propagation (e.g. Schwarzschild black hole), film response. Slang interfaces + module linking.
- [ ] **5. Simulation programs** – Slang compute stages owned by the scene: particles / agents
      (million-bird flock), GPU instancing into the TLAS, per-frame refit.
- [ ] **6. VFX + libraries** – plate footage, tracked cameras, proxy geometry & shadow catchers,
      occlusion mattes; fluids library (APIC/FLIP + surface reconstruction).
- [ ] **7. Agent ergonomics** – token-budgeted help (~2.5k tokens first page), `help field`
      lookups, Python/TypeScript SDKs, benchmarks vs Blender.
