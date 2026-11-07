/* Generated from `pxl schema` by scripts/gen-types.mjs. Do not edit. */

/**
 * This interface was referenced by `Scene`'s JSON-Schema
 * via the `definition` "Node".
 */
export type Node =
  | {
      geometry: Geometry;
      materials?: Material[];
      name?: string | null;
      /**
       * Euler XYZ, degrees.
       *
       * @minItems 3
       * @maxItems 3
       */
      rotation?: [number, number, number];
      /**
       * @minItems 3
       * @maxItems 3
       */
      scale?: [number, number, number];
      /**
       * @minItems 3
       * @maxItems 3
       */
      translation?: [number, number, number];
      type: "mesh";
    }
  | {
      /**
       * Path relative to the scene file.
       */
      file: string;
      name?: string | null;
      /**
       * @minItems 3
       * @maxItems 3
       */
      rotation?: [number, number, number];
      /**
       * @minItems 3
       * @maxItems 3
       */
      scale?: [number, number, number];
      /**
       * @minItems 3
       * @maxItems 3
       */
      translation?: [number, number, number];
      type: "asset";
    }
  | {
      /**
       * @minItems 3
       * @maxItems 3
       */
      color?: [number, number, number];
      /**
       * Watts for point and rectangle lights; irradiance (W/m²) for directional.
       */
      energy?: number;
      name?: string | null;
      shape?: "point" | "rectangle" | "directional";
      /**
       * Rectangle size in metres: number or [w, h].
       */
      size?: number | [number, number];
      /**
       * Point the light (rectangle / directional) at this world position.
       *
       * @minItems 3
       * @maxItems 3
       */
      target?: [number, number, number];
      /**
       * @minItems 3
       * @maxItems 3
       */
      translation?: [number, number, number];
      type: "light";
    }
  | {
      /**
       * Distance to the plane in focus, metres. Defaults to the distance to `focus_target`,
       * else to `target`.
       */
      focus_distance?: number | null;
      /**
       * @minItems 3
       * @maxItems 3
       */
      focus_target?: [number, number, number] | null;
      /**
       * Vertical field of view, degrees.
       */
      fov?: number;
      /**
       * Lens f-number for depth of field (e.g. 1.4 = very shallow, 16 = deep). Omit for a pinhole.
       */
      fstop?: number | null;
      name?: string | null;
      /**
       * Euler rotation in degrees [rx, ry, rz], applied Z first, then X, then Y (Nuke/Maya
       * "ZXY"). Looks down -Z with +Y up. Overrides `target` when given.
       *
       * @minItems 3
       * @maxItems 3
       */
      rotation?: [number, number, number] | null;
      /**
       * @minItems 3
       * @maxItems 3
       */
      target?: [number, number, number];
      /**
       * Camera track (.chan: frame tx ty tz rx ry rz [vertical fov]) driving translation,
       * rotation and fov per frame. Relative to the scene file.
       */
      track?: string | null;
      /**
       * @minItems 3
       * @maxItems 3
       */
      translation?: [number, number, number];
      type: "camera";
    };
/**
 * This interface was referenced by `Scene`'s JSON-Schema
 * via the `definition` "Geometry".
 */
export type Geometry =
  | {
      radius?: number;
      segments?: number;
      std: "sphere";
    }
  | {
      /**
       * @minItems 2
       * @maxItems 2
       */
      size?: [number, number];
      std: "grid";
    }
  | {
      /**
       * @minItems 3
       * @maxItems 3
       */
      size?: [number, number, number];
      std: "box";
    }
  | {
      height?: number;
      radius?: number;
      segments?: number;
      std: "cylinder";
    }
  | {
      radius?: number;
      segments?: number;
      std: "torus";
      tube?: number;
    }
  | {
      std: "bird";
    };
/**
 * This interface was referenced by `Scene`'s JSON-Schema
 * via the `definition` "Interp".
 */
export type Interp = "linear" | "smooth" | "spline" | "step";
/**
 * This interface was referenced by `Scene`'s JSON-Schema
 * via the `definition` "LightShape".
 */
export type LightShape = "point" | "rectangle" | "directional";
/**
 * Material roles for VFX: proxies of real-world things in the plate.
 *
 * This interface was referenced by `Scene`'s JSON-Schema
 * via the `definition` "Role".
 */
export type Role = "surface" | "shadow_catcher" | "holdout";
/**
 * This interface was referenced by `Scene`'s JSON-Schema
 * via the `definition` "Size2".
 */
export type Size2 = number | [number, number];
/**
 * This interface was referenced by `Scene`'s JSON-Schema
 * via the `definition` "View".
 */
export type View = "aces" | "agx" | "standard";

export interface Scene {
  animation?: Animation | null;
  fluids?: Fluid[];
  nodes?: Node[];
  /**
   * Background footage the render is composited over.
   */
  plate?: Plate | null;
  programs?: Programs | null;
  render?: RenderSettings;
  simulations?: Simulation[];
  version: number;
  world?: World;
}
/**
 * This interface was referenced by `Scene`'s JSON-Schema
 * via the `definition` "Animation".
 */
export interface Animation {
  /**
   * Defaults to the last key of any track.
   */
  frame_end?: number | null;
  frame_start?: number;
  tracks?: Track[];
}
/**
 * This interface was referenced by `Scene`'s JSON-Schema
 * via the `definition` "Track".
 */
export interface Track {
  interp?: "linear" | "smooth" | "spline" | "step";
  keys?: [unknown, unknown][];
  /**
   * Node `name`; omit to address the scene root (e.g. "world/strength").
   */
  node?: string | null;
  orbit?: Orbit | null;
  /**
   * Slash-separated field path, e.g. "translation" or "materials/0/inputs/base_color".
   */
  path: string;
}
/**
 * This interface was referenced by `Scene`'s JSON-Schema
 * via the `definition` "Orbit".
 */
export interface Orbit {
  /**
   * @minItems 3
   * @maxItems 3
   */
  center?: [number, number, number];
  degrees?: number;
  /**
   * @minItems 2
   * @maxItems 2
   */
  frames: [number, number];
  /**
   * Height above `center`.
   */
  height?: number;
  radius: number;
  /**
   * Angle at the first frame; 0 = +Z side of center.
   */
  start_deg?: number;
}
/**
 * A liquid simulated with FLIP on a grid inside a closed box and rendered as a mesh.
 *
 * This interface was referenced by `Scene`'s JSON-Schema
 * via the `definition` "Fluid".
 */
export interface Fluid {
  /**
   * Closed container [min, max] in metres; walls are solid.
   *
   * @minItems 2
   * @maxItems 2
   */
  bounds: [[number, number, number], [number, number, number]];
  /**
   * Liquid present at time 0 (and after warm-up): boxes with optional initial velocity.
   */
  fill: FluidBlock[];
  /**
   * 0 = PIC (smooth, viscous), 1 = FLIP (lively, noisy); 0.95 is typical.
   */
  flip?: number;
  /**
   * @minItems 3
   * @maxItems 3
   */
  gravity?: [number, number, number];
  materials?: Material[];
  max_triangles?: number;
  name: string;
  /**
   * Particle radius for the surface, in surface cells (larger = smoother, softer detail).
   */
  particle_radius?: number;
  particles_per_cell?: number;
  pressure_iterations?: number;
  /**
   * Grid cells along the longest side of `bounds` (16..256).
   */
  resolution?: number;
  /**
   * Simulation steps per frame; raise for fast, splashy motion.
   */
  substeps?: number;
  /**
   * Surface mesh resolution relative to the grid (1..3).
   */
  surface_refine?: number;
  warmup?: number;
}
/**
 * This interface was referenced by `Scene`'s JSON-Schema
 * via the `definition` "FluidBlock".
 */
export interface FluidBlock {
  /**
   * @minItems 3
   * @maxItems 3
   */
  max: [number, number, number];
  /**
   * @minItems 3
   * @maxItems 3
   */
  min: [number, number, number];
  /**
   * @minItems 3
   * @maxItems 3
   */
  velocity?: [number, number, number];
}
/**
 * OpenPBR-style input names (the same vocabulary MaterialX uses).
 *
 * This interface was referenced by `Scene`'s JSON-Schema
 * via the `definition` "Material".
 */
export interface Material {
  inputs?: MaterialInputs;
  /**
   * Name of a material function in `programs` that edits the surface after inputs/textures.
   */
  program?: string | null;
  /**
   * How the camera sees this surface when compositing over a plate / background.
   */
  role?: "surface" | "shadow_catcher" | "holdout";
}
export interface MaterialInputs {
  /**
   * @minItems 3
   * @maxItems 3
   */
  base_color?: [number, number, number];
  /**
   * sRGB image multiplied with base_color.
   */
  base_color_texture?: string | null;
  base_metalness?: number;
  base_metalness_texture?: string | null;
  /**
   * @minItems 3
   * @maxItems 3
   */
  emission_color?: [number, number, number];
  emission_color_texture?: string | null;
  /**
   * Emitted radiance scale (W/sr/m²); 0 = not emissive.
   */
  emission_luminance?: number;
  normal_scale?: number;
  /**
   * Tangent-space normal map (OpenGL / +Y convention, as glTF and Poly Haven "nor_gl").
   */
  normal_texture?: string | null;
  /**
   * Index of refraction: water 1.33, glass 1.5, diamond 2.42.
   */
  specular_ior?: number;
  specular_roughness?: number;
  /**
   * Greyscale (red channel) image multiplied with specular_roughness.
   */
  specular_roughness_texture?: string | null;
  /**
   * Colour of light after travelling `transmission_depth` metres inside (0 = thin tint).
   *
   * @minItems 3
   * @maxItems 3
   */
  transmission_color?: [number, number, number];
  transmission_depth?: number;
  /**
   * 0 = opaque, 1 = glass / water (refracts; reflection by Fresnel from `specular_ior`).
   */
  transmission_weight?: number;
  /**
   * Texture tiling.
   *
   * @minItems 2
   * @maxItems 2
   */
  uv_scale?: [number, number];
}
/**
 * Footage (or a still) that CG is composited over. Frame `f` of the scene uses plate frame
 * `f + offset`. Shadow catchers darken it, holdouts and the `matte` reveal it.
 *
 * This interface was referenced by `Scene`'s JSON-Schema
 * via the `definition` "Plate".
 */
export interface Plate {
  /**
   * Video (.mp4/.mov/...), image sequence with a `#` run (plate_####.png) or a single image.
   */
  file: string;
  /**
   * Optional roto matte sequence/image: white = real foreground in front of all CG.
   */
  matte?: string | null;
  offset?: number;
  /**
   * Shadow strength on the plate (0 = no shadows, 1 = physically matched).
   */
  shadow?: number;
}
/**
 * Scene-supplied Slang code hooked into the renderer. See `pxl help programs`.
 *
 * This interface was referenced by `Scene`'s JSON-Schema
 * via the `definition` "Programs".
 */
export interface Programs {
  /**
   * `bool f(CameraIn c, inout float3 origin, inout float3 dir)`
   */
  camera?: string | null;
  /**
   * Inline Slang code (appended after `source`).
   */
  code?: string | null;
  /**
   * `float3 f(FilmIn i)`
   */
  film?: string | null;
  /**
   * Values exposed to programs as `P.<name>`: number -> float, [a,b] -> float2, ... float4.
   * Animatable with tracks on path `programs/params/<name>`.
   */
  params?: {
    [k: string]: unknown;
  };
  /**
   * `Hit f(inout Ray r, inout float3 radiance, inout float3 throughput, inout uint rng)`
   */
  propagate?: string | null;
  /**
   * Slang file, relative to the scene file.
   */
  source?: string | null;
}
export interface RenderSettings {
  bounces?: number | null;
  /**
   * Run the AI denoiser (default: per profile).
   */
  denoise?: boolean | null;
  /**
   * Exposure in stops.
   */
  exposure?: number;
  fps?: number;
  /**
   * Shutter open time as a fraction of a frame (0.5 = 180° shutter). Omit for no motion blur.
   */
  motion_blur?: number | null;
  /**
   * @minItems 2
   * @maxItems 2
   */
  resolution?: [number, number];
  samples?: number | null;
  seed?: number;
  /**
   * Transparent background in PNG output (alpha = geometry coverage).
   */
  transparent?: boolean;
  view?: "aces" | "agx" | "standard";
}
/**
 * A population of GPU agents stepped by scene programs and drawn as instances of one mesh.
 *
 * This interface was referenced by `Scene`'s JSON-Schema
 * via the `definition` "Simulation".
 */
export interface Simulation {
  asset?: string | null;
  /**
   * Neighbour grid cell size (≈ interaction radius). 0 = no neighbour queries.
   */
  cell_size?: number;
  /**
   * Number of agents (1 .. 16M).
   */
  count: number;
  /**
   * Mesh drawn for every agent (+Z = forward, +Y = up), or `asset` (first mesh of a glTF).
   */
  geometry?: Geometry | null;
  /**
   * `void f(uint id, inout Agent a, inout uint rng)`, run once.
   */
  init: string;
  materials?: Material[];
  name: string;
  /**
   * Turn each instance to face along its velocity.
   */
  orient?: boolean;
  /**
   * Mesh scale (multiplied by each agent's `size`).
   */
  scale?: number;
  /**
   * `void f(uint id, inout Agent a, inout uint rng)`, run every substep (`SIM.dt`).
   */
  step: string;
  substeps?: number;
  /**
   * Seconds simulated before time 0, so the system has settled when the shot starts.
   */
  warmup?: number;
}
export interface World {
  /**
   * Uniform environment radiance (used when no `hdri`).
   *
   * @minItems 3
   * @maxItems 3
   */
  color?: [number, number, number];
  /**
   * Equirectangular .hdr / .exr environment, path relative to the scene file.
   */
  hdri?: string | null;
  /**
   * Rotate the HDRI around +Y, degrees.
   */
  rotation?: number;
  strength?: number;
}
/**
 * This interface was referenced by `Scene`'s JSON-Schema
 * via the `definition` "MaterialInputs".
 */
export interface MaterialInputs1 {
  /**
   * @minItems 3
   * @maxItems 3
   */
  base_color?: [number, number, number];
  /**
   * sRGB image multiplied with base_color.
   */
  base_color_texture?: string | null;
  base_metalness?: number;
  base_metalness_texture?: string | null;
  /**
   * @minItems 3
   * @maxItems 3
   */
  emission_color?: [number, number, number];
  emission_color_texture?: string | null;
  /**
   * Emitted radiance scale (W/sr/m²); 0 = not emissive.
   */
  emission_luminance?: number;
  normal_scale?: number;
  /**
   * Tangent-space normal map (OpenGL / +Y convention, as glTF and Poly Haven "nor_gl").
   */
  normal_texture?: string | null;
  /**
   * Index of refraction: water 1.33, glass 1.5, diamond 2.42.
   */
  specular_ior?: number;
  specular_roughness?: number;
  /**
   * Greyscale (red channel) image multiplied with specular_roughness.
   */
  specular_roughness_texture?: string | null;
  /**
   * Colour of light after travelling `transmission_depth` metres inside (0 = thin tint).
   *
   * @minItems 3
   * @maxItems 3
   */
  transmission_color?: [number, number, number];
  transmission_depth?: number;
  /**
   * 0 = opaque, 1 = glass / water (refracts; reflection by Fresnel from `specular_ior`).
   */
  transmission_weight?: number;
  /**
   * Texture tiling.
   *
   * @minItems 2
   * @maxItems 2
   */
  uv_scale?: [number, number];
}
/**
 * This interface was referenced by `Scene`'s JSON-Schema
 * via the `definition` "RenderSettings".
 */
export interface RenderSettings1 {
  bounces?: number | null;
  /**
   * Run the AI denoiser (default: per profile).
   */
  denoise?: boolean | null;
  /**
   * Exposure in stops.
   */
  exposure?: number;
  fps?: number;
  /**
   * Shutter open time as a fraction of a frame (0.5 = 180° shutter). Omit for no motion blur.
   */
  motion_blur?: number | null;
  /**
   * @minItems 2
   * @maxItems 2
   */
  resolution?: [number, number];
  samples?: number | null;
  seed?: number;
  /**
   * Transparent background in PNG output (alpha = geometry coverage).
   */
  transparent?: boolean;
  view?: "aces" | "agx" | "standard";
}
/**
 * This interface was referenced by `Scene`'s JSON-Schema
 * via the `definition` "World".
 */
export interface World1 {
  /**
   * Uniform environment radiance (used when no `hdri`).
   *
   * @minItems 3
   * @maxItems 3
   */
  color?: [number, number, number];
  /**
   * Equirectangular .hdr / .exr environment, path relative to the scene file.
   */
  hdri?: string | null;
  /**
   * Rotate the HDRI around +Y, degrees.
   */
  rotation?: number;
  strength?: number;
}
