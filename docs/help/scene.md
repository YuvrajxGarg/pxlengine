# Scene reference (version 1)

    {"version":1, "nodes":[...], "world":{...}, "render":{...}}

## mesh
    {"type":"mesh", "name":?, "translation":[0,0,0], "rotation":[0,0,0] (Euler XYZ deg),
     "scale":[1,1,1], "geometry":{...}, "materials":[{"inputs":{...}}]}

geometry (`std` selects the generator):
- `{"std":"sphere","radius":1,"segments":64}`
- `{"std":"grid","size":[w,d]}`        flat, XZ plane, faces +Y (alias "plane")
- `{"std":"box","size":[x,y,z]}`
- `{"std":"cylinder","radius":1,"height":2,"segments":64}`   Y axis, capped
- `{"std":"torus","radius":1,"tube":0.25,"segments":64}`     XZ plane

material inputs (OpenPBR names, defaults shown):
- `base_color` [0.8,0.8,0.8]       albedo, keep <= ~0.9
- `specular_roughness` 0.5         0 mirror .. 1 matte
- `base_metalness` 0               1 = metal (base_color tints reflections)
- `emission_color` [1,1,1], `emission_luminance` 0   glowing surfaces (try 5..50)

## light
    {"type":"light", "shape":"point"|"rectangle"|"directional",
     "translation":[x,y,z], "target":[0,0,0], "energy":100, "color":[1,1,1], "size":1}
- point: `energy` watts, emits in all directions.
- rectangle: `energy` watts over the area, one-sided, faces `target`;
  `size` = metres, number or [w,h]. Soft shadows grow with size.
- directional: `energy` = irradiance W/m² (sun ~3..5), shines from
  `translation` toward `target`; only the direction matters.

## camera
    {"type":"camera", "translation":[0,1,5], "target":[0,0,0], "fov":40}
`fov` = vertical field of view in degrees.

## world
    {"color":[0.05,0.055,0.065], "strength":1}
Uniform environment light for rays that leave the scene; also the background.

## render
    {"resolution":[640,480], "samples":?, "bounces":?, "exposure":0, "seed":0}
`samples`/`bounces` override the profile. `exposure` in stops (+1 = 2x brighter).
