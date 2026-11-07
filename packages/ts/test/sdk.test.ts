import { test } from "node:test";
import assert from "node:assert/strict";
import { existsSync, mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { PxlError, Scene, camera, grid, help, light, material, mesh, schema, sphere } from "../src/index.js";

function firstScene(): Scene {
  const s = new Scene({ resolution: [96, 72] });
  s.add(mesh(sphere(1), material({ base_color: [0.12, 0.35, 0.65], specular_roughness: 0.3 }), { name: "ball" }));
  s.add(mesh(grid(10, 10), material({ base_color: [0.35, 0.35, 0.35] }), { translation: [0, -1, 0] }));
  s.add(light.rectangle([-3, 5, 4], [0, 0, 0], 700, 4));
  s.add(camera([3, 2, 5], [0, 0, 0], { name: "cam" }));
  return s;
}

test("builds engine JSON", () => {
  const d = firstScene().toJSON();
  assert.equal(d.version, 1);
  assert.deepEqual(d.nodes?.[0], {
    type: "mesh",
    geometry: { std: "sphere", radius: 1, segments: 64 },
    materials: [{ inputs: { base_color: [0.12, 0.35, 0.65], specular_roughness: 0.3 } }],
    name: "ball",
  });
});

test("check accepts a valid scene and reports engine errors", async () => {
  assert.match(await firstScene().check(), /^ok:/);
  const bad = firstScene();
  (bad.data.nodes![3] as Record<string, unknown>).fvo = 30;
  await assert.rejects(bad.check(), (e: unknown) => e instanceof PxlError && /fvo/.test(e.message));
});

test("renders a still and an animated sequence", async () => {
  const dir = mkdtempSync(path.join(tmpdir(), "pxl-ts-"));
  const s = firstScene().animate("cam", "translation", [[0, [3, 2, 5]], [2, [-3, 2, 5]]], "smooth");
  const still = await s.render(path.join(dir, "still.png"), { profile: "std/previs" });
  assert.ok(existsSync(still.outputs[0]), JSON.stringify(still));
  assert.match(still.summary, /96x72/);
  const seq = await s.render(path.join(dir, "f_##.png"), { profile: "std/previs", set: { samples: 2 } });
  assert.equal(seq.outputs.length, 3);
});

test("help and schema", async () => {
  assert.match(await help("field", "camera.fov"), /Vertical field of view/);
  assert.ok("$defs" in (await schema()));
});
