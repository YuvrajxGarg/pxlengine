"""Build a pxl scene in Blender (for comparisons and benchmarks) and optionally render it.

    blender -b --factory-startup -P import_pxl.py -- scene.json [out.png] [--engine CYCLES|EEVEE]
            [--samples 64] [--bounces 8] [--denoise 1] [--resolution 1920 1080]

Supported: std geometry, glTF assets, OpenPBR-style material inputs (-> Principled BSDF,
including transmission/IOR), point/rectangle/directional lights, look-at or rotated cameras
with fov / f-stop, world colour or HDRI, static scenes (the first frame of animations).
Not translated: material/camera/propagate/film programs, simulations, fluids, plates.
pxl is Y-up; Blender is Z-up: a pxl point (x, y, z) becomes (x, -z, y).
Lighting units are close but not identical, so brightness can differ slightly.
"""

import argparse
import json
import math
import os
import sys

import bpy
from mathutils import Euler, Matrix, Vector

# pxl (Y-up) -> Blender (Z-up)
C = Matrix(((1, 0, 0, 0), (0, 0, -1, 0), (0, 1, 0, 0), (0, 0, 0, 1)))
C_INV = C.inverted()


def to_bl(v):
    return Vector((v[0], -v[2], v[1]))


def pxl_matrix(t, r_deg, s):
    """pxl node transform (Euler XYZ degrees, Y-up) as a Blender world matrix."""
    rx, ry, rz = (math.radians(a) for a in r_deg)
    R = (Matrix.Rotation(rx, 4, "X") @ Matrix.Rotation(ry, 4, "Y") @ Matrix.Rotation(rz, 4, "Z"))
    # glam EulerRot::XYZ = Rx * Ry * Rz (intrinsic), matching the engine.
    S = Matrix.Diagonal((s[0], s[1], s[2], 1.0))
    T = Matrix.Translation(Vector(t))
    return C @ (T @ R @ S) @ C_INV


def material(spec, name):
    inputs = (spec or {}).get("inputs", {})
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    p = next(n for n in m.node_tree.nodes if n.type == "BSDF_PRINCIPLED")

    def setin(names, value):
        for n in names:
            if n in p.inputs:
                p.inputs[n].default_value = value
                return

    bc = inputs.get("base_color", [0.8, 0.8, 0.8])
    setin(["Base Color"], (*bc, 1))
    setin(["Roughness"], inputs.get("specular_roughness", 0.5))
    setin(["Metallic"], inputs.get("base_metalness", 0.0))
    lum = inputs.get("emission_luminance", 0.0)
    if lum > 0:
        ec = inputs.get("emission_color", [1, 1, 1])
        setin(["Emission Color", "Emission"], (*ec, 1))
        setin(["Emission Strength"], lum)
    tw = inputs.get("transmission_weight", 0.0)
    if tw > 0:
        setin(["Transmission Weight", "Transmission"], tw)
        setin(["IOR"], inputs.get("specular_ior", 1.5))
        tc = inputs.get("transmission_color", [1, 1, 1])
        setin(["Base Color"], (*tc, 1))
    tex = inputs.get("base_color_texture")
    if tex:
        img = m.node_tree.nodes.new("ShaderNodeTexImage")
        img.image = bpy.data.images.load(os.path.join(BASE, tex))
        m.node_tree.links.new(img.outputs["Color"], p.inputs["Base Color"])
    return m


def add_geometry(g):
    kind = g["std"]
    if kind == "sphere":
        bpy.ops.mesh.primitive_uv_sphere_add(radius=g.get("radius", 1), segments=g.get("segments", 64), ring_count=max(g.get("segments", 64) // 2, 3))
        bpy.ops.object.shade_smooth()
    elif kind in ("grid", "plane"):
        w, d = g.get("size", [1, 1])
        bpy.ops.mesh.primitive_plane_add(size=1)
        bpy.context.object.scale = (w, d, 1)
        bpy.ops.object.transform_apply(scale=True)
    elif kind == "box":
        x, y, z = g.get("size", [1, 1, 1])
        bpy.ops.mesh.primitive_cube_add(size=1)
        bpy.context.object.scale = (x, z, y)          # pxl Y-up -> Blender Z-up
        bpy.ops.object.transform_apply(scale=True)
    elif kind == "cylinder":
        bpy.ops.mesh.primitive_cylinder_add(radius=g.get("radius", 1), depth=g.get("height", 2), vertices=g.get("segments", 64))
        bpy.ops.object.shade_smooth()
    elif kind == "torus":
        bpy.ops.mesh.primitive_torus_add(major_radius=g.get("radius", 1), minor_radius=g.get("tube", 0.25),
                                         major_segments=g.get("segments", 64), minor_segments=max(g.get("segments", 64) // 2, 3))
        bpy.ops.object.shade_smooth()
    else:
        raise SystemExit(f"geometry std={kind} is not supported by the Blender importer")
    return bpy.context.object


def build(scene_path, args):
    global BASE
    BASE = os.path.dirname(os.path.abspath(scene_path))
    with open(scene_path, encoding="utf-8") as f:
        s = json.load(f)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene

    for i, n in enumerate(s.get("nodes", [])):
        t = n.get("translation", [0, 0, 0])
        r = n.get("rotation", [0, 0, 0]) if n["type"] != "camera" else [0, 0, 0]
        sc_ = n.get("scale", [1, 1, 1])
        if n["type"] == "mesh":
            ob = add_geometry(n["geometry"])
            ob.matrix_world = pxl_matrix(t, r, sc_)
            mats = n.get("materials") or [None]
            ob.data.materials.append(material(mats[0], f"m{i}"))
        elif n["type"] == "asset":
            before = set(bpy.data.objects)
            bpy.ops.import_scene.gltf(filepath=os.path.join(BASE, n["file"]))
            M = pxl_matrix(t, r, sc_)
            for ob in set(bpy.data.objects) - before:
                if ob.parent is None:
                    ob.matrix_world = M @ ob.matrix_world
        elif n["type"] == "light":
            shape = n.get("shape", "point")
            energy = n.get("energy", 100)
            color = n.get("color", [1, 1, 1])
            pos = to_bl(t)
            target = to_bl(n.get("target", [0, 0, 0]))
            if shape == "point":
                ld = bpy.data.lights.new(f"l{i}", "POINT")
                ld.energy = energy
            elif shape == "rectangle":
                ld = bpy.data.lights.new(f"l{i}", "AREA")
                size = n.get("size", 1)
                w, h = (size, size) if isinstance(size, (int, float)) else size
                ld.shape = "RECTANGLE"
                ld.size, ld.size_y = w, h
                ld.energy = energy
            else:
                ld = bpy.data.lights.new(f"l{i}", "SUN")
                ld.energy = energy
            ld.color = color
            ob = bpy.data.objects.new(f"l{i}", ld)
            sc.collection.objects.link(ob)
            ob.location = pos
            d = (target - pos) if shape != "point" else Vector((0, 0, -1))
            ob.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
        elif n["type"] == "camera":
            if sc.camera:
                continue
            cd = bpy.data.cameras.new("cam")
            cd.sensor_fit = "VERTICAL"
            cd.angle_y = math.radians(n.get("fov", 40))
            cd.clip_start = 0.01
            if n.get("fstop"):
                cd.dof.use_dof = True
                cd.dof.aperture_fstop = n["fstop"]
                cd.sensor_height = 24
                focus = n.get("focus_distance")
                if focus is None:
                    ft = n.get("focus_target", n.get("target", [0, 0, 0]))
                    focus = (to_bl(ft) - to_bl(t)).length
                cd.dof.focus_distance = focus
            ob = bpy.data.objects.new("cam", cd)
            sc.collection.objects.link(ob)
            ob.location = to_bl(t)
            if n.get("rotation") is not None:
                rx, ry, rz = (math.radians(a) for a in n["rotation"])
                Rp = Matrix.Rotation(ry, 4, "Y") @ Matrix.Rotation(rx, 4, "X") @ Matrix.Rotation(rz, 4, "Z")
                ob.matrix_world = Matrix.Translation(to_bl(t)) @ (C @ Rp @ C_INV)
            else:
                d = to_bl(n.get("target", [0, 0, 0])) - to_bl(t)
                ob.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
            sc.camera = ob

    w = s.get("world", {})
    world = bpy.data.worlds.new("world")
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs["Strength"].default_value = w.get("strength", 1.0)
    if w.get("hdri"):
        env = world.node_tree.nodes.new("ShaderNodeTexEnvironment")
        env.image = bpy.data.images.load(os.path.join(BASE, w["hdri"]))
        mapping = world.node_tree.nodes.new("ShaderNodeMapping")
        coord = world.node_tree.nodes.new("ShaderNodeTexCoord")
        # pxl uses Blender's equirectangular convention, and world.rotation = Mapping Z rotation.
        mapping.inputs["Rotation"].default_value[2] = math.radians(w.get("rotation", 0.0))
        world.node_tree.links.new(coord.outputs["Generated"], mapping.inputs["Vector"])
        world.node_tree.links.new(mapping.outputs["Vector"], env.inputs["Vector"])
        world.node_tree.links.new(env.outputs["Color"], bg.inputs["Color"])
    else:
        c = w.get("color", [0.05, 0.055, 0.065])
        bg.inputs["Color"].default_value = (*c, 1)
    sc.world = world

    res = args.resolution or s.get("render", {}).get("resolution", [640, 480])
    sc.render.resolution_x, sc.render.resolution_y = res
    sc.render.resolution_percentage = 100
    sc.view_settings.view_transform = "AgX"

    if args.engine == "CYCLES":
        sc.render.engine = "CYCLES"
        prefs = bpy.context.preferences.addons["cycles"].preferences
        for kind in ("OPTIX", "CUDA", "HIP", "METAL", "ONEAPI"):
            try:
                prefs.compute_device_type = kind
                prefs.get_devices()
                gpus = [d for d in prefs.devices if d.type == kind]
                if gpus:
                    for d in prefs.devices:
                        d.use = d.type == kind
                    sc.cycles.device = "GPU"
                    break
            except TypeError:
                continue
        sc.cycles.samples = args.samples
        sc.cycles.use_adaptive_sampling = False
        sc.cycles.max_bounces = args.bounces
        sc.cycles.use_denoising = bool(args.denoise)
        sc.cycles.denoiser = "OPENIMAGEDENOISE"
    else:
        for engine in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"):
            try:
                sc.render.engine = engine
                break
            except TypeError:
                continue
        sc.eevee.taa_render_samples = args.samples


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("scene")
    ap.add_argument("out", nargs="?")
    ap.add_argument("--engine", default="CYCLES", choices=["CYCLES", "EEVEE"])
    ap.add_argument("--samples", type=int, default=64)
    ap.add_argument("--bounces", type=int, default=8)
    ap.add_argument("--denoise", type=int, default=1)
    ap.add_argument("--resolution", type=int, nargs=2)
    args = ap.parse_args(argv)
    build(args.scene, args)
    if args.out:
        sc = bpy.context.scene
        sc.render.filepath = os.path.abspath(args.out)
        sc.render.image_settings.file_format = "PNG"
        import time
        t = time.perf_counter()
        bpy.ops.render.render(write_still=True)
        print(f"PXL_BENCH_RENDER_SECONDS {time.perf_counter() - t:.3f}")


main()
