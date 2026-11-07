# Rendering

    pxl render <scene.json> --out <output> [--profile P] [--set key=value ...]
                            [--frame N | --frame-start A --frame-end B]

Outputs (chosen by the --out name):
- `still.png`           one frame (frame_start of the animation, or `--frame N`)
- `frames/f_####.png`   numbered sequence; `#` run = zero-padded frame number
- `clip.mp4`            H.264 via ffmpeg (NVENC when available, else x264)
- `clip.mov`            ProRes 422 HQ via ffmpeg
Sequences and video use the scene's `animation` range unless --frame-start/--frame-end
are given (both inclusive: 0..47 = 48 frames). Geometry and textures load once.

Profiles (GPU path tracing with hardware ray queries):

| profile    | samples | bounces | use |
|------------|---------|---------|-----|
| std/previs | 8       | 1       | framing, blocking, animation timing |
| std/draft  | 64      | 4       | default; judge materials and light |
| std/final  | 256     | 8       | delivery |

`--set` keys: `samples=N`, `bounces=N`, `seed=N`, `exposure=EV`, `clamp=X` (max
indirect radiance, default 32; lower = fewer fireflies, slightly darker glossy bounces).
Noise falls with sqrt(samples): 4x samples = half the noise. No denoiser yet.

Lighting is sampled from analytic lights, glowing triangles, and the HDRI together
(multiple importance sampling), so small bright sources stay clean.

Environment: `PXL_DEVICE=<name substring>` picks a GPU, `PXL_VALIDATION=1` enables
Vulkan validation, `PXL_SLANGC` / `PXL_FFMPEG` point at tools, `PXL_CACHE_DIR` moves
the shader cache. `pxl info` shows what was found.
