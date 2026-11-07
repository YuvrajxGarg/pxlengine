// Regenerates src/scene-types.ts from the engine's JSON Schema (`pxl schema`).
import { execFileSync } from "node:child_process";
import { writeFileSync } from "node:fs";
import { compile } from "json-schema-to-typescript";
import { findPxl } from "./find-pxl.mjs";

const schema = JSON.parse(execFileSync(findPxl(), ["schema"], { encoding: "utf8" }));
const ts = await compile(schema, "Scene", {
  bannerComment: "/* Generated from `pxl schema` by scripts/gen-types.mjs. Do not edit. */",
  additionalProperties: false,
  unreachableDefinitions: true,
  style: { singleQuote: false },
});
writeFileSync(new URL("../src/scene-types.ts", import.meta.url), ts);
console.log("wrote src/scene-types.ts");
