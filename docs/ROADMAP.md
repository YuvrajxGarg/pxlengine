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
- [x] **2. Assets + animation + video** (done 2026-11-07; 72-frame 960x540 draft MP4 in 4.4 s) – glTF import (meshes, PBR textures), keyframed
      animation tracks on any node field, frame ranges, MP4 via ffmpeg, emissive meshes as lights
      (NEE + MIS, texture-weighted), importance-sampled HDRI environments, alpha-mask cut-outs,
      normal maps, AgX tone mapping, PNG sequences / MP4 (NVENC) / MOV (ProRes).
      Known gaps: alpha BLEND is treated as MASK; no transmission/glass; rectangle lights are
      invisible to camera and mirrors; no denoiser (phase 3).
- [x] **3. Denoise, lens, passes** (done 2026-11-07) – Intel OIDN 2.5.1 loaded at runtime (CUDA on
      NVIDIA, CPU fallback) guided by albedo + normal; profiles previs/draft/final (4/16/64 spp,
      denoised) and `std/path` (1024 spp reference); thin-lens depth of field (f-stop, focus
      distance/target); shutter motion blur by subframe accumulation (up to 16 instants);
      passes albedo/normal/depth/alpha/object; EXR output; transparent PNG.
      Decision: no raster G-buffer path — the ray-query previs already renders in ~5–10 ms, so a
      raster pipeline would add complexity without a speed win on RT hardware.
      Known gaps: motion blur rebuilds the scene per subframe (~200 ms/frame at 16 instants);
      object-id comes from the first sample (aliased edges); denoiser can soften fine grain.
- [x] **4. Program hooks** (done 2026-11-07) – scene `programs` block with Slang `source`/`code`
      and typed, animatable `params` (`P.<name>`). Hooks: material (per mesh), camera rays,
      light propagation (curved rays marched with `trace_segment`), film (per-pixel pass after
      denoising, can read neighbours). Code is spliced into the core shader under `#line` so
      compiler errors point at the user's file. Examples: Schwarzschild black hole with a
      Keplerian, Doppler-beamed accretion disk and lensed starfield; marble / lava / tiles
      materials; fisheye lens; aberration + vignette + grain film look.
      Known gaps: with a propagation program, emissive/HDRI light sampling is off (BSDF-only,
      noisier) and shadow rays stay straight; emission added by material programs isn't
      importance-sampled; programs can't add their own GPU buffers yet (phase 5).
- [x] **5. Simulation programs** (done 2026-11-07) – scene `simulations`: agents (64 B each)
      initialised and stepped by scene Slang functions with fixed substeps and warm-up;
      per-step GPU neighbour grid (hashed cells, counting sort: count -> 3-pass prefix scan ->
      scatter) behind a `neighbors()` iterator; a GPU emit kernel writes every agent as a TLAS
      instance + renderer instance record (velocity-oriented, per-agent tint); TLAS built from
      GPU memory each subframe; motion blur extrapolates along velocity. Measured on the RTX 4080
      Laptop: 1M boids ≈ 31 ms per step, instance+TLAS build ≈ 21 ms, 1280x720 16 spp trace ≈ 22 ms.
      Known gaps: no wing flapping / per-agent animation; one mesh per simulation; agents can't
      read the scene geometry (no collisions with meshes yet); state lives only within a render.
- [x] **6a. VFX on footage** (done 2026-11-07) – `plate` (video via ffmpeg, image sequence or
      still, frame offset, roto `matte`); material roles `shadow_catcher` (shadow ratio of light
      with vs without CG occluders, via instance masks: CG = 0x01, proxies = 0x02; NEE + cosine
      ambient, denoised) and `holdout`; camera `rotation` (ZXY) and `.chan` tracks (Nuke format)
      with `pxl track` export; display-space composite (plate untouched outside CG/shadows),
      linear composite for EXR; `shadow` pass. Validated with a synthetic round trip: a plate
      rendered by pxl, its camera exported to .chan, CG composited back with proxies — CG stays
      locked to the floor through a 48-frame dolly; video vs PNG plates agree within 1.2/255.
      Known gaps: no camera solver (bring a .chan from a tracker); shadow catchers don't show
      CG reflections on the real floor; plate lighting must be matched by hand (HDRI/lights);
      no lens distortion model; single-layer holdouts (no soft roto edges beyond the matte).
- [x] **6b. Fluids + glass** (done 2026-11-07) – dielectric BSDF (Fresnel reflect/refract, GGX
      rough transmission, Beer-Lambert absorption, glass lets shadow rays through; glTF
      KHR_materials_transmission/ior/volume). FLIP/PIC liquid on a GPU MAC grid in Slang:
      fixed-point atomic P2G, red-black Gauss-Seidel pressure, RK2 advection, closed box walls;
      surface nets meshing each frame straight into reserved vertex/index buffer space + a
      per-frame BLAS. Dam break: 139k particles, 64-cell grid, 8 substeps ≈ 12 ms/frame; 120
      frames at 1280x720 rendered in 18 s. Not APIC yet (FLIP/PIC blend).
      Known gaps: surfaces stay slightly lumpy (isotropic kernels; no anisotropic smoothing);
      no obstacles inside the box, no emitters/sinks, no spray/foam/bubbles; volume drifts a
      little (no density correction); no caustics through glass/water.
- [ ] **7. Agent ergonomics** – token-budgeted help (~2.5k tokens first page), `help field`
      lookups, Python/TypeScript SDKs, benchmarks vs Blender.
