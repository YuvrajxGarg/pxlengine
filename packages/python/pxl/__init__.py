"""pxlengine Python SDK: build scene JSON in Python and render it with the `pxl` binary.

    from pxl import Scene, mesh, sphere, grid, material, light, camera

    s = Scene(resolution=(640, 480))
    s.add(mesh(sphere(1.0), material(base_color=(0.12, 0.35, 0.65), specular_roughness=0.3)))
    s.add(mesh(grid(10, 10), material(base_color=(0.35, 0.35, 0.35)), translation=(0, -1, 0)))
    s.add(light.rectangle((-3, 5, 4), target=(0, 0, 0), energy=700, size=4))
    s.add(camera((3, 2, 5), target=(0, 0, 0), name="cam"))
    s.animate("cam", "translation", [(0, (3, 2, 5)), (47, (-3, 2, 5))], interp="spline")
    print(s.render("preview.png", profile="std/previs").outputs)

Everything is plain dicts underneath: anything the engine accepts can be added with
`Scene.data` or keyword arguments, and `pxl help field <path>` documents every field.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence, Union

__all__ = [
    "Scene", "RenderResult", "PxlError", "find_pxl",
    "mesh", "asset", "camera", "light", "material",
    "sphere", "grid", "box", "cylinder", "torus", "bird",
]
__version__ = "0.1.0"

Vec3 = Sequence[float]
Json = dict[str, Any]


class PxlError(RuntimeError):
    """The engine rejected a scene or failed to render; `str(err)` is its message."""


def _clean(d: Json) -> Json:
    return {k: v for k, v in d.items() if v is not None}


def _vec(v: Optional[Iterable[float]]) -> Optional[list[float]]:
    return None if v is None else [float(x) for x in v]


# ------------------------------------------------------------------ geometry
def sphere(radius: float = 1.0, segments: int = 64) -> Json:
    return {"std": "sphere", "radius": radius, "segments": segments}


def grid(width: float = 1.0, depth: float = 1.0) -> Json:
    return {"std": "grid", "size": [width, depth]}


def box(x: float = 1.0, y: float = 1.0, z: float = 1.0) -> Json:
    return {"std": "box", "size": [x, y, z]}


def cylinder(radius: float = 1.0, height: float = 2.0, segments: int = 64) -> Json:
    return {"std": "cylinder", "radius": radius, "height": height, "segments": segments}


def torus(radius: float = 1.0, tube: float = 0.25, segments: int = 64) -> Json:
    return {"std": "torus", "radius": radius, "tube": tube, "segments": segments}


def bird() -> Json:
    return {"std": "bird"}


# ------------------------------------------------------------------ materials
def material(program: Optional[str] = None, role: Optional[str] = None, **inputs: Any) -> Json:
    """OpenPBR-style inputs, e.g. base_color=(r,g,b), specular_roughness=0.3,
    base_metalness=1, emission_luminance=20, transmission_weight=1, specular_ior=1.33."""
    out: Json = {"inputs": {k: (_vec(v) if isinstance(v, (tuple, list)) else v) for k, v in inputs.items()}}
    if program:
        out["program"] = program
    if role:
        out["role"] = role
    return out


# ------------------------------------------------------------------ nodes
def mesh(geometry: Json, mat: Optional[Json] = None, *, name: Optional[str] = None,
         translation: Optional[Vec3] = None, rotation: Optional[Vec3] = None,
         scale: Optional[Union[Vec3, float]] = None) -> Json:
    if isinstance(scale, (int, float)):
        scale = (scale, scale, scale)
    return _clean({"type": "mesh", "name": name, "geometry": geometry,
                   "materials": [mat] if mat else None, "translation": _vec(translation),
                   "rotation": _vec(rotation), "scale": _vec(scale)})


def asset(file: Union[str, Path], *, name: Optional[str] = None, translation: Optional[Vec3] = None,
          rotation: Optional[Vec3] = None, scale: Optional[Union[Vec3, float]] = None) -> Json:
    if isinstance(scale, (int, float)):
        scale = (scale, scale, scale)
    return _clean({"type": "asset", "name": name, "file": str(file).replace("\\", "/"),
                   "translation": _vec(translation), "rotation": _vec(rotation), "scale": _vec(scale)})


def camera(translation: Vec3 = (0, 1, 5), target: Vec3 = (0, 0, 0), fov: float = 40.0, *,
           name: Optional[str] = None, fstop: Optional[float] = None, **extra: Any) -> Json:
    return _clean({"type": "camera", "name": name, "translation": _vec(translation),
                   "target": _vec(target), "fov": fov, "fstop": fstop, **extra})


class light:  # namespace: light.point(...), light.rectangle(...), light.sun(...)
    @staticmethod
    def point(translation: Vec3, energy: float = 100, color: Vec3 = (1, 1, 1), *, name: Optional[str] = None) -> Json:
        return _clean({"type": "light", "shape": "point", "name": name, "translation": _vec(translation),
                       "energy": energy, "color": _vec(color)})

    @staticmethod
    def rectangle(translation: Vec3, target: Vec3 = (0, 0, 0), energy: float = 500,
                  size: Union[float, Sequence[float]] = 1.0, color: Vec3 = (1, 1, 1), *,
                  name: Optional[str] = None) -> Json:
        return _clean({"type": "light", "shape": "rectangle", "name": name, "translation": _vec(translation),
                       "target": _vec(target), "energy": energy,
                       "size": size if isinstance(size, (int, float)) else _vec(size), "color": _vec(color)})

    @staticmethod
    def sun(direction_from: Vec3, energy: float = 3.0, color: Vec3 = (1, 0.95, 0.9), *,
            name: Optional[str] = None) -> Json:
        """Directional light shining from `direction_from` towards the origin."""
        return _clean({"type": "light", "shape": "directional", "name": name,
                       "translation": _vec(direction_from), "target": [0, 0, 0],
                       "energy": energy, "color": _vec(color)})


# ------------------------------------------------------------------ binary
def find_pxl() -> str:
    """`PXL_BIN`, then `pxl` on PATH, then a build in this repository."""
    env = os.environ.get("PXL_BIN")
    if env:
        return env
    on_path = shutil.which("pxl")
    if on_path:
        return on_path
    exe = "pxl.exe" if os.name == "nt" else "pxl"
    root = Path(__file__).resolve().parents[3]
    for build in ("release", "debug"):
        p = root / "target" / build / exe
        if p.is_file():
            return str(p)
    raise PxlError("pxl binary not found: build it (cargo build --release) or set PXL_BIN")


def _run(args: list[str]) -> subprocess.CompletedProcess:
    proc = subprocess.run([find_pxl(), *args], capture_output=True, text=True)
    if proc.returncode != 0:
        raise PxlError((proc.stderr or proc.stdout).strip().removeprefix("error: "))
    return proc


@dataclass
class RenderResult:
    outputs: list[str]
    summary: str
    seconds: float
    log: str = field(repr=False, default="")


# ------------------------------------------------------------------ scene
class Scene:
    """A scene document. `base_dir` is where relative asset paths are resolved from."""

    def __init__(self, resolution: Sequence[int] = (640, 480), *, base_dir: Union[str, Path, None] = None,
                 **render: Any) -> None:
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()
        self.data: Json = {"version": 1, "nodes": [],
                           "render": {"resolution": [int(resolution[0]), int(resolution[1])], **render}}

    # -- building
    def add(self, *nodes: Json) -> "Scene":
        self.data["nodes"].extend(nodes)
        return self

    def world(self, color: Optional[Vec3] = None, strength: Optional[float] = None,
              hdri: Optional[Union[str, Path]] = None, rotation: Optional[float] = None) -> "Scene":
        self.data["world"] = _clean({"color": _vec(color), "strength": strength,
                                     "hdri": str(hdri).replace("\\", "/") if hdri else None, "rotation": rotation})
        return self

    def programs(self, source: Optional[Union[str, Path]] = None, *, code: Optional[str] = None,
                 params: Optional[Json] = None, **hooks: str) -> "Scene":
        """Slang programs: hooks camera=, propagate=, film= name functions in `source`/`code`."""
        self.data["programs"] = _clean({"source": str(source).replace("\\", "/") if source else None,
                                        "code": code, "params": params, **hooks})
        return self

    def animate(self, node: Optional[str], path: str, keys: Sequence[tuple[float, Any]],
                interp: str = "linear", frame_end: Optional[int] = None) -> "Scene":
        """Keyframe any numeric field: animate("cam", "translation", [(0, (0,1,5)), (48, (4,1,3))])."""
        anim = self.data.setdefault("animation", {"tracks": []})
        track = _clean({"node": node, "path": path, "interp": interp,
                        "keys": [[f, _vec(v) if isinstance(v, (tuple, list)) else v] for f, v in keys]})
        anim["tracks"].append(track)
        if frame_end is not None:
            anim["frame_end"] = frame_end
        return self

    def simulation(self, **spec: Any) -> "Scene":
        self.data.setdefault("simulations", []).append(spec)
        return self

    def fluid(self, **spec: Any) -> "Scene":
        self.data.setdefault("fluids", []).append(spec)
        return self

    # -- output
    def to_json(self, indent: Optional[int] = 1) -> str:
        return json.dumps(self.data, indent=indent)

    def save(self, path: Union[str, Path]) -> Path:
        p = Path(path)
        p.write_text(self.to_json(), encoding="utf-8")
        return p

    def _with_file(self, fn):
        # Write next to base_dir so relative asset paths resolve as the user intended.
        fd, tmp = tempfile.mkstemp(prefix=".pxl-scene-", suffix=".json", dir=self.base_dir)
        os.close(fd)
        try:
            Path(tmp).write_text(self.to_json(), encoding="utf-8")
            return fn(tmp)
        finally:
            os.unlink(tmp)

    def check(self) -> str:
        """Validate without rendering (schema, frames, files, programs). Raises PxlError."""
        return self._with_file(lambda f: _run(["check", f]).stdout.strip())

    def render(self, out: Union[str, Path], profile: str = "std/draft", *,
               passes: Optional[Sequence[str]] = None, frame: Optional[float] = None,
               frame_start: Optional[int] = None, frame_end: Optional[int] = None,
               **settings: Any) -> RenderResult:
        """Render to .png/.exr/name_####.png/.mp4. `settings` become --set key=value."""
        out = str(Path(out).resolve())
        args = ["render", "", "--out", out, "--profile", profile]
        if passes:
            args += ["--passes", ",".join(passes)]
        if frame is not None:
            args += ["--frame", str(frame)]
        if frame_start is not None:
            args += ["--frame-start", str(frame_start)]
        if frame_end is not None:
            args += ["--frame-end", str(frame_end)]
        for k, v in settings.items():
            args += ["--set", f"{k}={str(v).lower() if isinstance(v, bool) else v}"]

        def go(path: str) -> RenderResult:
            args[1] = path
            t = time.perf_counter()
            proc = _run(args)
            summary = next((l for l in proc.stderr.splitlines() if l.startswith("std/")), "")
            return RenderResult(outputs=proc.stdout.split(), summary=summary,
                                seconds=time.perf_counter() - t, log=proc.stderr)

        return self._with_file(go)


def help(topic: str = "index", *path: str) -> str:  # noqa: A001 - mirrors `pxl help`
    """Engine documentation, e.g. help("field", "camera.fov")."""
    return _run(["help", topic, *path]).stdout


def schema() -> Json:
    """JSON Schema of scene files."""
    return json.loads(_run(["schema"]).stdout)
