/**
 * pxlengine TypeScript SDK: typed scene JSON (generated from `pxl schema`) plus helpers that
 * render through the `pxl` binary.
 *
 *   import { Scene, mesh, sphere, grid, material, light, camera } from "@pxlengine/sdk";
 *   const s = new Scene({ resolution: [640, 480] });
 *   s.add(mesh(sphere(1), material({ base_color: [0.12, 0.35, 0.65], specular_roughness: 0.3 })));
 *   s.add(camera([3, 2, 5], [0, 0, 0], { name: "cam" }));
 *   const r = await s.render("preview.png", { profile: "std/previs" });
 */
import { execFile } from "node:child_process";
import { randomBytes } from "node:crypto";
import { existsSync, rmSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import type * as T from "./scene-types.js";

export type * from "./scene-types.js";

type Vec3 = [number, number, number];
type Node = T.Node;
type MeshNode = Extract<Node, { type: "mesh" }>;
type CameraNode = Extract<Node, { type: "camera" }>;
type LightNode = Extract<Node, { type: "light" }>;
type AssetNode = Extract<Node, { type: "asset" }>;

export class PxlError extends Error {}

// ------------------------------------------------------------------ geometry
export const sphere = (radius = 1, segments = 64): T.Geometry => ({ std: "sphere", radius, segments });
export const grid = (width = 1, depth = 1): T.Geometry => ({ std: "grid", size: [width, depth] });
export const box = (x = 1, y = 1, z = 1): T.Geometry => ({ std: "box", size: [x, y, z] });
export const cylinder = (radius = 1, height = 2, segments = 64): T.Geometry => ({ std: "cylinder", radius, height, segments });
export const torus = (radius = 1, tube = 0.25, segments = 64): T.Geometry => ({ std: "torus", radius, tube, segments });
export const bird = (): T.Geometry => ({ std: "bird" });

// ------------------------------------------------------------------ materials & nodes
export function material(inputs: Partial<T.MaterialInputs> = {}, opts: { program?: string; role?: T.Role } = {}): T.Material {
  return { inputs: inputs as T.MaterialInputs, ...opts } as T.Material;
}

type Placement = { name?: string; translation?: Vec3; rotation?: Vec3; scale?: Vec3 | number };
const place = (p: Placement) => {
  const { scale, ...rest } = p;
  return { ...rest, ...(scale === undefined ? {} : { scale: typeof scale === "number" ? ([scale, scale, scale] as Vec3) : scale }) };
};

export function mesh(geometry: T.Geometry, mat?: T.Material, p: Placement = {}): MeshNode {
  return { type: "mesh", geometry, ...(mat ? { materials: [mat] } : {}), ...place(p) } as MeshNode;
}

export function asset(file: string, p: Placement = {}): AssetNode {
  return { type: "asset", file: file.replaceAll("\\", "/"), ...place(p) } as AssetNode;
}

export function camera(translation: Vec3 = [0, 1, 5], target: Vec3 = [0, 0, 0], extra: Partial<Omit<CameraNode, "type">> = {}): CameraNode {
  return { type: "camera", translation, target, ...extra } as CameraNode;
}

export const light = {
  point: (translation: Vec3, energy = 100, extra: Partial<LightNode> = {}): LightNode =>
    ({ type: "light", shape: "point", translation, energy, ...extra }) as LightNode,
  rectangle: (translation: Vec3, target: Vec3 = [0, 0, 0], energy = 500, size: T.Size2 = 1, extra: Partial<LightNode> = {}): LightNode =>
    ({ type: "light", shape: "rectangle", translation, target, energy, size, ...extra }) as LightNode,
  /** Directional light shining from `from` towards the origin. */
  sun: (from: Vec3, energy = 3, extra: Partial<LightNode> = {}): LightNode =>
    ({ type: "light", shape: "directional", translation: from, target: [0, 0, 0], energy, ...extra }) as LightNode,
};

// ------------------------------------------------------------------ binary
export function findPxl(): string {
  if (process.env.PXL_BIN) return process.env.PXL_BIN;
  const exe = process.platform === "win32" ? "pxl.exe" : "pxl";
  // dist/src/index.js -> repo root is four levels up from dist/src.
  const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../../../..");
  for (const b of ["release", "debug"]) {
    const p = path.join(root, "target", b, exe);
    if (existsSync(p)) return p;
  }
  return "pxl";
}

function run(args: string[]): Promise<{ stdout: string; stderr: string }> {
  return new Promise((resolve, reject) => {
    execFile(findPxl(), args, { maxBuffer: 64 << 20 }, (err, stdout, stderr) => {
      if (err) reject(new PxlError((stderr || stdout || String(err)).trim().replace(/^error: /, "")));
      else resolve({ stdout, stderr });
    });
  });
}

export interface RenderOptions {
  profile?: "std/previs" | "std/draft" | "std/final" | "std/path";
  passes?: Array<"albedo" | "normal" | "depth" | "alpha" | "object" | "shadow" | "all">;
  frame?: number;
  frameStart?: number;
  frameEnd?: number;
  /** --set overrides: samples, bounces, seed, exposure, denoise, motion_blur, clamp. */
  set?: Record<string, string | number | boolean>;
}

export interface RenderResult {
  outputs: string[];
  summary: string;
  seconds: number;
  log: string;
}

// ------------------------------------------------------------------ scene
export class Scene {
  data: T.Scene;
  /** Where relative asset paths resolve from (the temp scene file is written here). */
  baseDir: string;

  constructor(opts: { resolution?: [number, number]; baseDir?: string; render?: Partial<T.RenderSettings> } = {}) {
    this.baseDir = opts.baseDir ?? process.cwd();
    this.data = { version: 1, nodes: [], render: { resolution: opts.resolution ?? [640, 480], ...opts.render } as T.RenderSettings };
  }

  add(...nodes: Node[]): this {
    (this.data.nodes ??= []).push(...nodes);
    return this;
  }

  world(world: Partial<T.World>): this {
    this.data.world = world as T.World;
    return this;
  }

  programs(programs: T.Programs): this {
    this.data.programs = programs;
    return this;
  }

  /** Keyframe any numeric field: animate("cam", "translation", [[0, [0,1,5]], [48, [4,1,3]]]). */
  animate(node: string | null, fieldPath: string, keys: Array<[number, number | number[]]>, interp: T.Interp = "linear"): this {
    const anim = (this.data.animation ??= { tracks: [] } as T.Animation);
    (anim.tracks ??= []).push({ ...(node ? { node } : {}), path: fieldPath, keys, interp } as T.Track);
    return this;
  }

  simulation(sim: T.Simulation): this {
    (this.data.simulations ??= []).push(sim);
    return this;
  }

  fluid(fluid: T.Fluid): this {
    (this.data.fluids ??= []).push(fluid);
    return this;
  }

  toJSON(): T.Scene {
    return this.data;
  }

  /** Write the scene into baseDir (so relative asset paths resolve), run, then delete it. */
  private async withFile<R>(fn: (file: string) => Promise<R>): Promise<R> {
    const file = path.join(this.baseDir, `.pxl-scene-${process.pid}-${randomBytes(4).toString("hex")}.json`);
    writeFileSync(file, JSON.stringify(this.data));
    try {
      return await fn(file);
    } finally {
      rmSync(file, { force: true });
    }
  }

  /** Validate without rendering (schema, frames, files, programs). Throws PxlError. */
  check(): Promise<string> {
    return this.withFile(async (f) => (await run(["check", f])).stdout.trim());
  }

  /** Render to .png / .exr / name_####.png / .mp4 / .mov. */
  render(out: string, opts: RenderOptions = {}): Promise<RenderResult> {
    const args = ["render", "", "--out", path.resolve(out), "--profile", opts.profile ?? "std/draft"];
    if (opts.passes?.length) args.push("--passes", opts.passes.join(","));
    if (opts.frame !== undefined) args.push("--frame", String(opts.frame));
    if (opts.frameStart !== undefined) args.push("--frame-start", String(opts.frameStart));
    if (opts.frameEnd !== undefined) args.push("--frame-end", String(opts.frameEnd));
    for (const [k, v] of Object.entries(opts.set ?? {})) args.push("--set", `${k}=${v}`);
    return this.withFile(async (f) => {
      args[1] = f;
      const t = performance.now();
      const { stdout, stderr } = await run(args);
      return {
        outputs: stdout.split(/\r?\n/).filter(Boolean),
        summary: stderr.split(/\r?\n/).find((l) => l.startsWith("std/")) ?? "",
        seconds: (performance.now() - t) / 1000,
        log: stderr,
      };
    });
  }
}

/** Engine documentation, e.g. help("field", "camera.fov"). */
export async function help(topic = "index", ...rest: string[]): Promise<string> {
  return (await run(["help", topic, ...rest])).stdout;
}

/** JSON Schema of scene files. */
export async function schema(): Promise<object> {
  return JSON.parse((await run(["schema"])).stdout);
}
