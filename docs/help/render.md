# Rendering

    pxl render <scene.json> --out <file.png> [--profile P] [--set key=value ...]

Profiles (path traced on the GPU with hardware ray queries):

| profile    | samples | bounces | use |
|------------|---------|---------|-----|
| std/previs | 8       | 1       | framing, blocking; noisy, direct light only |
| std/draft  | 64      | 4       | default; judge materials and light |
| std/final  | 256     | 8       | delivery |

`--set` keys: `samples=N` (1..65536), `bounces=N`, `seed=N`, `exposure=EV`.
Noise falls with sqrt(samples): 4x samples = half the noise. No denoiser yet.

Output: 8-bit sRGB PNG, ACES tone mapping. Timings print on stderr.

Environment: `PXL_DEVICE=<name substring>` picks a GPU, `PXL_VALIDATION=1`
enables Vulkan validation, `PXL_SLANGC` points at slangc, `PXL_CACHE_DIR`
moves the shader cache. The first render compiles shaders (a few seconds);
later renders reuse the cache.
