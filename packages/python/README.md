# pxlengine Python SDK

Build pxl scene JSON in Python and render it with the `pxl` binary. No dependencies.

```python
from pxl import Scene, mesh, sphere, grid, material, light, camera

s = Scene(resolution=(640, 480))
s.add(mesh(sphere(1.0), material(base_color=(0.12, 0.35, 0.65), specular_roughness=0.3), name="ball"))
s.add(mesh(grid(10, 10), material(base_color=(0.35, 0.35, 0.35)), translation=(0, -1, 0)))
s.add(light.rectangle((-3, 5, 4), target=(0, 0, 0), energy=700, size=4))
s.add(camera((3, 2, 5), target=(0, 0, 0), name="cam"))
s.animate("cam", "translation", [(0, (3, 2, 5)), (47, (-3, 2, 5))], interp="spline")

print(s.check())                                   # validate: frames, files, programs
r = s.render("preview.png", profile="std/previs")  # or "clip.mp4", "f_####.png", "shot.exr"
print(r.outputs, r.summary, r.seconds)
```

- Everything is a plain dict: `s.data` is the scene; add any field the engine accepts.
- `pxl.help("field", "camera.fov")` documents any field; `pxl.schema()` returns the JSON Schema.
- Errors from the engine raise `pxl.PxlError` with the engine's message (field, line, fix).
- The binary is found via `PXL_BIN`, `pxl` on `PATH`, or this repository's `target/release`.
- Relative asset paths resolve from `Scene(base_dir=...)` (default: the current directory).

Install: `pip install packages/python`. Tests: `python -m unittest discover -s packages/python/tests`.
