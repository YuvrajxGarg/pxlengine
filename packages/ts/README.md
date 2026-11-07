# @pxlengine/sdk

Typed pxl scenes for TypeScript / Node, rendered through the `pxl` binary. Scene types in
`src/scene-types.ts` are generated from the engine's JSON Schema (`npm run gen-types`), so the
compiler checks every field name and type the engine accepts.

```ts
import { Scene, mesh, sphere, grid, material, light, camera } from "@pxlengine/sdk";

const s = new Scene({ resolution: [640, 480] });
s.add(mesh(sphere(1), material({ base_color: [0.12, 0.35, 0.65], specular_roughness: 0.3 }), { name: "ball" }));
s.add(mesh(grid(10, 10), material({ base_color: [0.35, 0.35, 0.35] }), { translation: [0, -1, 0] }));
s.add(light.rectangle([-3, 5, 4], [0, 0, 0], 700, 4));
s.add(camera([3, 2, 5], [0, 0, 0], { name: "cam" }));
s.animate("cam", "translation", [[0, [3, 2, 5]], [47, [-3, 2, 5]]], "spline");

console.log(await s.check());
const r = await s.render("preview.png", { profile: "std/previs" });
console.log(r.outputs, r.summary);
```

`help("field", "camera.fov")`, `schema()`, and `PxlError` (engine messages) are exported.
The binary is found via `PXL_BIN` or this repository's `target/release`, then `PATH`.

Develop: `npm install`, `npm run gen-types` (after engine schema changes), `npm test`.
