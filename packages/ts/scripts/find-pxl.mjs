import { existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";

export function findPxl() {
  if (process.env.PXL_BIN) return process.env.PXL_BIN;
  const exe = process.platform === "win32" ? "pxl.exe" : "pxl";
  const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../../..");
  for (const b of ["release", "debug"]) {
    const p = path.join(root, "target", b, exe);
    if (existsSync(p)) return p;
  }
  return "pxl";
}
