# Rendering

    pxl render <scene.json> --out <output> [--profile P] [--passes LIST] [--set key=value ...]
                            [--frame N | --frame-start A --frame-end B]

Outputs (chosen by the --out name):
- `still.png`             one frame, display-ready (view transform applied)
- `still.exr`             one frame, scene-linear float RGBA (alpha = coverage), for compositing
- `frames/f_####.png|exr` numbered sequence; `#` run = zero-padded frame number
- `clip.mp4` / `clip.mov` H.264 (NVENC when available) / ProRes 422 HQ via ffmpeg
Sequences and video use the scene's `animation` range unless --frame-start/--frame-end
are given (inclusive: 0..47 = 48 frames). Geometry and textures load once.

Profiles (GPU path tracing with hardware ray queries; AI denoiser = Intel OIDN):

| profile    | samples | bounces | denoise | use |
|------------|---------|---------|---------|-----|
| std/previs | 4       | 1       | yes     | framing, blocking, timing |
| std/draft  | 16      | 4       | yes     | default; judge materials and light |
| std/final  | 64      | 8       | yes     | delivery |
| std/path   | 1024    | 16      | no      | ground-truth reference, very fine detail |

Denoising can smooth fine texture or grain; compare against std/path or raise samples
(`--set samples=256`) when detail matters. `--set denoise=false` shows raw samples.

`--set` keys: `samples=N`, `bounces=N`, `seed=N`, `exposure=EV`, `denoise=true|false`,
`motion_blur=SHUTTER` (0..1), `clamp=X` (max indirect radiance, default 32).

`--passes albedo,normal,depth,alpha,object,shadow` (or `all`) writes extra files beside the
output: `shot.png` -> `shot.albedo.png`, `shot.depth.exr` (metres along the camera axis,
always EXR), `shot.object.png` (one flat colour per scene node: a matte for each object),
`shot.shadow.png` (multiplier from shadow catchers: 1 = unshadowed).
With EXR output every pass is EXR. With video, passes are numbered files beside the clip.

Environment: `PXL_DEVICE=<name substring>` picks a GPU, `PXL_VALIDATION=1` enables
Vulkan validation, `PXL_SLANGC` / `PXL_FFMPEG` point at tools, `PXL_OIDN_DIR` at the
denoiser, `PXL_OIDN_DEVICE=cpu|cuda` forces its device, `PXL_CACHE_DIR` moves the
shader cache. `pxl info` shows what was found.
