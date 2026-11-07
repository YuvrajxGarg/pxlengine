1920x1080, 64 samples, 8 bounces, OIDN denoising; NVIDIA GeForce RTX 4080 Laptop GPU; Blender 5.2.3 LTS; median of 3 after a warm-up.

| Scene | pxl std/final (s) | Cycles GPU (s) | Cycles render call (s) | EEVEE (s) | EEVEE render call (s) | Cycles / pxl |
|---|---|---|---|---|---|---|
| first-scene | 0.68 | 6.86 | 5.22 | 3.20 | 1.51 | 10.1x |
| materials | 0.73 | 6.83 | 5.22 | 3.14 | 1.49 | 9.4x |
| helmet (glTF + HDRI) | 0.98 | 7.66 | 5.74 | 3.63 | 1.54 | 7.8x |
| depth of field | 0.98 | 7.78 | 5.77 | 3.75 | 1.68 | 7.9x |
| glass | 0.77 | 7.44 | 5.68 | 3.24 | 1.46 | 9.6x |
