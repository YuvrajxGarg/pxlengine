# pxl — headless 3D renderer for agents

Write a scene as JSON, render it, look at the image, revise. No editor.

    pxl render shot.json --profile std/previs --out preview.png   # fast look
    pxl render shot.json --profile std/final  --out final.png     # delivery
    pxl render shot.json --profile std/draft  --out clip.mp4      # animation -> video
    pxl render shot.json --out plate.exr --passes all            # compositing: EXR + mattes
    pxl help scene    # every node type and field
    pxl help render   # profiles, --set overrides, output
    pxl info          # which GPU is used

## First scene

    {"version":1,"nodes":[
      {"type":"mesh","geometry":{"std":"sphere","radius":1},
       "materials":[{"inputs":{"base_color":[0.12,0.35,0.65],"specular_roughness":0.3}}]},
      {"type":"mesh","translation":[0,-1,0],"geometry":{"std":"grid","size":[10,10]},
       "materials":[{"inputs":{"base_color":[0.35,0.35,0.35]}}]},
      {"type":"light","shape":"rectangle","translation":[-3,5,4],"target":[0,0,0],
       "energy":700,"size":4,"color":[1,0.92,0.8]},
      {"type":"camera","translation":[3,2,5],"target":[0,0,0]}
    ],"render":{"resolution":[640,480]}}

## Conventions

- Units: metres, Y up, angles in degrees, colours linear RGB 0..1.
- Unknown fields are errors; read the message, it names the field and line.
- Exactly one camera is used (the first). Meshes take `materials[0]`.
- Lighting is physically based: if an image is dark, raise light `energy`
  or `render.exposure` (stops) rather than making materials brighter than 1.

## Workflow

1. Block the shot with std/previs at low resolution.
2. Inspect the PNG; fix framing (camera translation/target/fov) first, then light.
3. Render std/draft to judge materials; std/final for delivery.
4. For motion: check timing with std/previs video first, then render the final clip.

Real-world detail comes from assets: import glTF models (`"type":"asset"`), light with
an HDRI (`world.hdri`), and texture surfaces (`base_color_texture`, `normal_texture`).
