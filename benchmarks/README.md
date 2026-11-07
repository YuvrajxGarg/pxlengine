# Benchmarks: pxl vs Blender

`python benchmarks/run.py` renders the same scenes in pxl and in Blender (Cycles on the GPU
with OptiX, and EEVEE). Scenes are translated by `integrations/blender/import_pxl.py`;
the importer was checked by rendering the scenes in both engines side by side (framing,
materials, lighting, HDRI orientation and depth of field match).

## Results (2026-11-07)

1920x1080, 64 samples, 8 bounces, OIDN denoising, adaptive sampling off;
NVIDIA GeForce RTX 4080 Laptop GPU (Cycles device: OptiX, CPU disabled); Blender 5.2.3 LTS;
Windows 11. Median of 3 runs after one untimed warm-up. Raw data: `results/results.json`.

| Scene | pxl std/final (s) | Cycles GPU (s) | Cycles render call (s) | EEVEE (s) | EEVEE render call (s) | Cycles / pxl |
|---|---|---|---|---|---|---|
| first-scene | 0.68 | 6.86 | 5.22 | 3.20 | 1.51 | 10.1x |
| materials | 0.73 | 6.83 | 5.22 | 3.14 | 1.49 | 9.4x |
| helmet (glTF + HDRI) | 0.98 | 7.66 | 5.74 | 3.63 | 1.54 | 7.8x |
| depth of field | 0.98 | 7.78 | 5.77 | 3.75 | 1.68 | 7.9x |
| glass | 0.77 | 7.44 | 5.68 | 3.24 | 1.46 | 9.6x |

## How to read this

- **Whole commands.** Every number is a complete process: start-up, scene loading,
  rendering, denoising and writing the PNG. Blender's start-up is about 1.5-2 s of each
  Cycles/EEVEE time; the "render call" columns time only `bpy.ops.render.render` (which
  still includes Blender's scene sync, BVH build and denoising). pxl's 0.7-1.0 s also
  includes its own start-up (Vulkan device, shader cache load, BLAS/TLAS build, OIDN).
- **Same budgets, not the same images.** Both path trace 64 samples with 8 bounces and
  denoise with OIDN, but Cycles' Principled BSDF is more complete than pxl's (multiple
  scattering, sheen, coat, subsurface, layered speculars), and its sampling is different.
  These are timings at matched settings, not equal-quality speedups.
- **EEVEE** rasterises with approximate lighting; it's a different trade-off altogether.
- **Material programs aren't translated.** The glass scene's tiled floor is a pxl material
  program, so Blender renders a plain grey floor there.
- Small, simple scenes. Heavy production scenes (millions of triangles, many textures,
  volumes) are not covered here and may behave differently.

## Rerun

```
cargo build --release
python benchmarks/run.py --runs 3            # --blender PATH, --width/--height, --samples
```
