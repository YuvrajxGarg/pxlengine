"""Run: python -m unittest discover -s packages/python/tests  (needs a built pxl binary)."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pxl  # noqa: E402
from pxl import PxlError, Scene, camera, grid, light, material, mesh, sphere  # noqa: E402


def first_scene() -> Scene:
    s = Scene(resolution=(96, 72))
    s.add(mesh(sphere(1.0), material(base_color=(0.12, 0.35, 0.65), specular_roughness=0.3), name="ball"))
    s.add(mesh(grid(10, 10), material(base_color=(0.35, 0.35, 0.35)), translation=(0, -1, 0)))
    s.add(light.rectangle((-3, 5, 4), target=(0, 0, 0), energy=700, size=4))
    s.add(camera((3, 2, 5), target=(0, 0, 0), name="cam"))
    return s


class SdkTest(unittest.TestCase):
    def test_json_matches_engine_format(self):
        d = json.loads(first_scene().to_json())
        self.assertEqual(d["version"], 1)
        self.assertEqual(d["nodes"][0]["geometry"], {"std": "sphere", "radius": 1.0, "segments": 64})
        self.assertEqual(d["nodes"][3]["type"], "camera")

    def test_check_accepts_valid_scene(self):
        self.assertIn("ok:", first_scene().check())

    def test_check_reports_engine_errors(self):
        s = first_scene()
        s.data["nodes"][3]["fvo"] = 30
        with self.assertRaises(PxlError) as e:
            s.check()
        self.assertIn("fvo", str(e.exception))

    def test_render_png_and_animation(self):
        s = first_scene().animate("cam", "translation", [(0, (3, 2, 5)), (2, (-3, 2, 5))], interp="smooth")
        with tempfile.TemporaryDirectory() as d:
            r = s.render(Path(d) / "still.png", profile="std/previs")
            self.assertTrue(Path(r.outputs[0]).is_file(), r)
            self.assertIn("96x72", r.summary)
            seq = s.render(Path(d) / "f_##.png", profile="std/previs", samples=2)
            self.assertEqual(len(seq.outputs), 3)

    def test_help_and_schema(self):
        self.assertIn("Vertical field of view", pxl.help("field", "camera.fov"))
        self.assertIn("$defs", pxl.schema())


if __name__ == "__main__":
    unittest.main()
