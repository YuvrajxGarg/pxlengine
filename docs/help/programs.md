# Programs: Slang code inside a scene

Programs change how the renderer works for one shot: custom lenses, procedural
materials, bent light, film looks. They run on the GPU inside the path tracer.

    "programs": {
      "source": "fx.slang",          // file beside the scene (and/or "code": "...inline...")
      "camera": "fisheye",           // optional hook functions, by name
      "propagate": "black_hole",
      "film": "grade",
      "params": {"strength": 2, "tint": [1, 0.5, 0.2]}   // -> P.strength (float), P.tint (float3)
    }
    material: {"program": "marble", "inputs": {...}}     // material hook, per mesh

Params are animatable: track `{"path":"programs/params/strength","keys":[[0,1],[48,4]]}`.
`time()` = seconds, `frame_number()` = frame. Compile errors name your file and line.

## Hook signatures

    void   marble(inout Surface s, SurfaceIn i)                 // material
    bool   fisheye(CameraIn c, inout float3 origin, inout float3 dir)   // false = black
    Hit    black_hole(inout Ray r, inout float3 radiance, inout float3 throughput, inout uint rng)
    float3 grade(FilmIn f)                                       // film, per pixel

Surface (edit any field after textures are applied):
  `p` position, `n` shading normal, `ng` geometric normal, `base` colour, `rough`,
  `metal`, `emission` (radiance; glowing via programs is found by path tracing only).
SurfaceIn: `position`, `object_position` (patterns that stick to moving objects),
  `geometric_normal`, `uv`, `view_dir`, `object` (scene node index).
CameraIn: `ndc` (-1..1, +y up), `pixel`, `resolution`, `position`, `forward`,
  `right`, `up`, `tan_half_fov`, `aspect`. Defaults (incl. depth of field) are in
  origin/dir on entry.
FilmIn: `color` (linear HDR, denoised), `alpha`, `pixel`, `resolution`, `normal`,
  `depth`, `albedo`. `film_color(int2 p)` reads any pixel (blur, glow, distortion).
  Runs before the view transform.

## Propagation (bent light)

Replaces "trace a straight ray". March the curved path in short chords and test each
chord against the scene with `trace_segment(origin, dir, length)` -> `Hit` (`.hit`, `.t`).
- Return the `Hit` of the chord that hit geometry, with `r.origin`/`r.dir` set to that chord.
- Emission along the way (disks, gas): `radiance += throughput * L`; absorption:
  `throughput *= transmittance`. `throughput = 0` ends the path (e.g. event horizon).
- Escaping: return `no_hit()` with `r.dir` = final direction; the world lights it unless
  you paint the sky yourself (`radiance += throughput * sky(dir); throughput = 0`).
While a propagation program is active, glowing surfaces and the HDRI are found by path
tracing alone (noisier: use more samples); lights keep straight shadow rays.

## Helpers

`PI`, `rnd(rng)` uniform 0..1, `lum(rgb)`, `env_radiance(dir)`, `trace_segment`,
`no_hit()`, `time()`, `frame_number()`, `film_color(p)`, `P.<param>`.
Examples: examples/black-hole.json, examples/programs.json, examples/fisheye.json.
