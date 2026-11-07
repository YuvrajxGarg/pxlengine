# Scene reference (version 1)

    {"version":1, "nodes":[...], "world":{...}, "render":{...}, "animation":{...}}

Every node may have a `"name"` (needed to animate it). Paths are relative to the scene file.

## mesh
    {"type":"mesh", "translation":[0,0,0], "rotation":[0,0,0] (Euler XYZ deg),
     "scale":[1,1,1], "geometry":{...}, "materials":[{"inputs":{...}}]}

geometry (`std` selects the generator; all have UVs):
- `{"std":"sphere","radius":1,"segments":64}`
- `{"std":"grid","size":[w,d]}`        flat, XZ plane, faces +Y (alias "plane")
- `{"std":"box","size":[x,y,z]}`
- `{"std":"cylinder","radius":1,"height":2,"segments":64}`   Y axis, capped
- `{"std":"torus","radius":1,"tube":0.25,"segments":64}`     XZ plane
- `{"std":"bird"}`   low-poly bird, wingspan 1, +Z forward (for simulations)

material inputs (OpenPBR names, defaults shown):
- `base_color` [0.8,0.8,0.8]  ·  `base_color_texture` "albedo.jpg" (sRGB, multiplies)
- `specular_roughness` 0.5 (0 mirror .. 1 matte)  ·  `specular_roughness_texture` (red channel)
- `base_metalness` 0 (1 = metal)  ·  `base_metalness_texture` (red channel)
- `emission_color` [1,1,1], `emission_luminance` 0 (glow; try 5..50)  ·  `emission_color_texture`
- `normal_texture` (OpenGL/+Y, e.g. Poly Haven `nor_gl`), `normal_scale` 1
- `uv_scale` [1,1]   texture tiling (e.g. [8,8] on a 10 m grid)
- `transmission_weight` 0, `specular_ior` 1.5, `transmission_color` [1,1,1],
  `transmission_depth` 0   glass/water (see `pxl help fluids`)
A material may also name a program: `{"program":"marble","inputs":{...}}` (see
`pxl help programs`); it runs after inputs and textures and can change any of them.
`"role"`: "surface" (default), "shadow_catcher" or "holdout" (see `pxl help vfx`).
Glowing meshes are real light sources (they light and shadow the scene).

## asset (glTF 2.0)
    {"type":"asset", "file":"model.glb", "translation":[..], "rotation":[..], "scale":[..]}
Imports every mesh with its PBR materials: base colour, metal/roughness, normal,
emissive (+ KHR_materials_emissive_strength) textures, alpha MASK cut-outs.
Units are metres; adjust `scale` if the model is tiny or huge.

## light
    {"type":"light", "shape":"point"|"rectangle"|"directional",
     "translation":[x,y,z], "target":[0,0,0], "energy":100, "color":[1,1,1], "size":1}
- point: `energy` watts. rectangle: watts, one-sided toward `target`, `size` m or [w,h].
- directional: `energy` = irradiance W/m² (sun ~3..5), direction translation→target.

## camera
    {"type":"camera", "translation":[0,1,5], "target":[0,0,0], "fov":40,
     "fstop":2.8, "focus_distance":?, "focus_target":[x,y,z]}
`rotation` [rx,ry,rz] deg (ZXY order) replaces `target`; `track` reads a .chan
camera track (`pxl help vfx`). `fov` = vertical, deg. `fstop` turns on depth of field (full-frame lens: 1.4 = very
shallow, 4 = moderate, 16 = nearly all sharp). Focus defaults to `focus_target`, else
`target`; `focus_distance` (m) overrides both. Animate `focus_distance` for rack focus.

## world
    {"color":[0.05,0.055,0.065], "strength":1, "hdri":"sky.hdr", "rotation":0}
`hdri` = equirectangular .hdr/.exr (e.g. Poly Haven); lights the scene and is the
background. `rotation` spins it around +Y (deg). Without `hdri`, `color` is uniform.

## render
    {"resolution":[640,480], "samples":?, "bounces":?, "exposure":0, "view":"agx",
     "fps":24, "seed":0, "motion_blur":0.5, "denoise":?, "transparent":false}
`view`: "agx" (filmic, default), "aces", "standard". `exposure` in stops.
`motion_blur` = shutter as a fraction of a frame (0.5 = 180° film look); everything
animated blurs, including the camera. `transparent` makes the PNG background alpha 0.

## animation
    {"frame_start":0, "frame_end":47, "tracks":[
      {"node":"cam", "path":"translation", "keys":[[0,[3,2,5]],[47,[-3,2,5]]], "interp":"spline"},
      {"node":"lamp", "path":"materials/0/inputs/emission_luminance", "keys":[[0,0],[24,30]]},
      {"path":"world/rotation", "keys":[[0,0],[47,90]]},
      {"node":"cam", "path":"translation",
       "orbit":{"center":[0,1,0], "radius":5, "height":1, "start_deg":0, "degrees":360, "frames":[0,47]}}]}
A track writes any numeric field: `node` = node name (omit for scene root), `path` =
slash path into that node. `keys` = [frame, number or array]. `interp`: "linear"
(default), "smooth" (ease in/out), "spline" (smooth through keys, best for cameras),
"step". `orbit` circles a point instead of keys (0° = +Z side). `frame_end` defaults
to the last key. Resolution and `world.hdri` can't be animated.
