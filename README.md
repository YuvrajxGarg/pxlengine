# pxlengine

A headless 3D + VFX engine for agents, built in Rust + Slang on Vulkan hardware ray tracing.

An agent writes a scene as JSON, runs `pxl render`, inspects the PNG, and revises.
No editor, no Blender, no UI.

```
pxl help                       # agent docs (start here)
pxl info                       # GPU / slangc / cache
pxl render examples/first-scene.json --profile std/final --out out/first.png
```

## What it does

You describe a shot in a JSON file: objects, materials, lights, camera, animation. `pxl`
turns it into a photoreal image or video. There's no window and nothing to click, just a
file in and a picture out.

- **Stills and video:** PNG/EXR frames, MP4 (H.264) or MOV (ProRes), from one command.
- **Realistic light:** a GPU path tracer (light bounces the way it does in reality). It gives
  reflections, soft shadows, glass, water, glowing neon, depth of field and motion blur.
- **Real-world content:** glTF 3D models, textures and HDRI skies (360° photos that light the scene).
- **Your own GPU code inside a shot:** short Slang programs for custom materials, lenses,
  bent light (e.g. a black hole) or film looks.
- **Simulations:** millions of GPU agents (flocks, swarms, particles) and liquid (FLIP fluid).
- **VFX:** put CG into real footage with tracked cameras, shadow catchers and holdouts, plus
  compositing passes (depth, normals, per-object mattes).

## What it's for

It's built for **AI agents** (and scripts) that make images, not for people dragging things
around in an editor. An agent can't use a mouse, but it can write a file, run a command and
look at a picture. So the whole workflow is:

1. write or edit `shot.json`
2. `pxl render shot.json --profile std/previs --out preview.png` (a rough preview takes milliseconds)
3. look at the image, decide what's wrong, fix the JSON, repeat
4. render the final still or video at full quality

Typical uses: generating product shots, concept art, animated clips, data-driven scenes,
synthetic training images, or VFX plates, all fully automated and repeatable (the same file
always gives the same image).

The tool is designed to be easy for an agent to drive. `pxl help` is short and complete,
errors name the exact field and line, unknown fields are rejected instead of silently ignored,
and `pxl schema` gives a machine-readable description of every option.

## How it works

```
scene.json ──► parse & validate ──► load assets ──► build GPU scene ──► path trace ──► denoise ──► tone map ──► PNG / EXR / MP4
 (+ .slang)     (pxl-scene)        (pxl-assets)     (pxl-gpu)         (pxl-render)   (OIDN)      (AgX)       (ffmpeg for video)
```

1. **Read the scene.** The JSON is parsed strictly. Built-in shapes (sphere, box, …) are
   generated, glTF models, textures and HDRIs are loaded, and animation tracks are evaluated
   for each frame.
2. **Put it on the GPU.** Using Vulkan, the geometry is packed into *acceleration structures*.
   These are search trees the graphics card's ray-tracing hardware uses to find what a ray of
   light hits, very fast.
3. **Trace light.** A compute shader written in Slang shoots many rays per pixel from the
   camera. At each surface it picks a bounce based on the material (metal, rough, glass…) and
   asks the lights directly how much light reaches that point. Averaging many such random light
   paths gives a physically correct image. Your own Slang programs are compiled into this
   shader, so custom materials, lenses and simulations run at full GPU speed.
4. **Clean up.** Few samples means a grainy image, so Intel Open Image Denoise (an AI denoiser)
   removes the grain, guided by colour and surface-direction images. That's why a
   preview can take milliseconds and a final frame under a second.
5. **Finish.** A filmic view transform (AgX) maps the bright light values to a natural-looking
   image. Frames are written as PNG/EXR or sent to ffmpeg for video.

Quality presets (`std/previs`, `std/draft`, `std/final`, `std/path`) just change how many rays
are traced per pixel, how many times light may bounce and whether the denoiser runs.

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
