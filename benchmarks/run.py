"""pxl vs Blender (Cycles GPU + EEVEE) on the same scenes, same resolution, samples, bounces.

    python benchmarks/run.py [--blender PATH] [--runs 3] [--width 1920 --height 1080]

Each scene is translated to Blender by integrations/blender/import_pxl.py. Times are complete
command wall times (process start to image written: startup, loading, shader caches warm,
rendering, denoising, writing), median of `--runs` after one untimed warm-up run. Blender's
own render-call time is reported separately. Quality differs between engines (sampling,
denoisers, EEVEE's rasterised lighting), so these are timings, not equal-quality speedups.
"""

import argparse
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCENES = [
    ("first-scene", "examples/first-scene.json", {}),
    ("materials", "examples/materials.json", {}),
    ("helmet (glTF + HDRI)", "examples/helmet.json", {}),
    ("depth of field", "examples/lens.json", {"frame": 0}),
    ("glass", "examples/glass.json", {}),
]


def pxl_bin() -> str:
    exe = "pxl.exe" if os.name == "nt" else "pxl"
    p = ROOT / "target" / "release" / exe
    if not p.is_file():
        sys.exit("build pxl first: cargo build --release")
    return str(p)


def find_blender(arg):
    if arg:
        return arg
    for base in [Path("C:/Program Files/Blender Foundation"), Path("/Applications/Blender.app/Contents/MacOS")]:
        if base.is_dir():
            hits = sorted(base.rglob("blender.exe" if os.name == "nt" else "blender"))
            if hits:
                return str(hits[-1])
    return "blender"


def timed(cmd):
    t = time.perf_counter()
    p = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.perf_counter() - t
    if p.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed:\n{p.stdout[-2000:]}\n{p.stderr[-2000:]}")
    return dt, p.stdout + p.stderr


def median_runs(cmd, runs):
    timed(cmd)  # warm-up: shader caches, OptiX kernels, file cache
    walls, logs = [], []
    for _ in range(runs):
        dt, log = timed(cmd)
        walls.append(dt)
        logs.append(log)
    return statistics.median(walls), logs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blender")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--samples", type=int, default=64)
    ap.add_argument("--bounces", type=int, default=8)
    ap.add_argument("--out", default=str(ROOT / "benchmarks" / "results"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    blender = find_blender(args.blender)
    pxl = pxl_bin()
    importer = str(ROOT / "integrations" / "blender" / "import_pxl.py")
    rows = []
    for name, rel, opts in SCENES:
        src = ROOT / rel
        scene = json.loads(src.read_text(encoding="utf-8"))
        scene.setdefault("render", {})["resolution"] = [args.width, args.height]
        scene["render"]["bounces"] = args.bounces
        scene["render"].pop("motion_blur", None)
        tmp = src.with_name(f".bench-{src.stem}.json")       # beside the original: same relative paths
        tmp.write_text(json.dumps(scene), encoding="utf-8")
        slug = src.stem
        try:
            cmd = [pxl, "render", str(tmp), "--profile", "std/final", "--set", f"samples={args.samples}",
                   "--set", "denoise=true", "--out", str(out / f"{slug}.pxl.png")]
            if "frame" in opts:
                cmd += ["--frame", str(opts["frame"])]
            pxl_s, _ = median_runs(cmd, args.runs)
            row = {"scene": name, "pxl_s": pxl_s}
            for engine in ("CYCLES", "EEVEE"):
                bcmd = [blender, "-b", "--factory-startup", "-P", importer, "--", str(tmp),
                        str(out / f"{slug}.{engine.lower()}.png"), "--engine", engine,
                        "--samples", str(args.samples), "--bounces", str(args.bounces), "--denoise", "1"]
                wall, logs = median_runs(bcmd, args.runs)
                internal = statistics.median(
                    float(m.group(1)) for log in logs for m in [re.search(r"PXL_BENCH_RENDER_SECONDS ([0-9.]+)", log)] if m
                )
                row[f"{engine.lower()}_s"] = wall
                row[f"{engine.lower()}_render_s"] = internal
            rows.append(row)
            print(f"{name:24} pxl {row['pxl_s']:6.2f}s | cycles {row['cycles_s']:6.2f}s "
                  f"(render {row['cycles_render_s']:.2f}s) | eevee {row['eevee_s']:6.2f}s (render {row['eevee_render_s']:.2f}s)",
                  flush=True)
        finally:
            tmp.unlink(missing_ok=True)

    bver = subprocess.run([blender, "--version"], capture_output=True, text=True).stdout.splitlines()[0]
    gpu = subprocess.run([pxl, "info"], capture_output=True, text=True).stdout.splitlines()[0].replace("device:", "").strip()
    meta = {"gpu": gpu, "blender": bver, "os": platform.platform(), "width": args.width, "height": args.height,
            "samples": args.samples, "bounces": args.bounces, "runs": args.runs}
    (out / "results.json").write_text(json.dumps({"meta": meta, "rows": rows}, indent=1), encoding="utf-8")
    lines = [
        "| Scene | pxl std/final (s) | Cycles GPU (s) | Cycles render call (s) | EEVEE (s) | EEVEE render call (s) | Cycles / pxl |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r['scene']} | {r['pxl_s']:.2f} | {r['cycles_s']:.2f} | {r['cycles_render_s']:.2f} | "
                     f"{r['eevee_s']:.2f} | {r['eevee_render_s']:.2f} | {r['cycles_s'] / r['pxl_s']:.1f}x |")
    table = "\n".join(lines)
    (out / "results.md").write_text(
        f"{meta['width']}x{meta['height']}, {meta['samples']} samples, {meta['bounces']} bounces, OIDN denoising; "
        f"{meta['gpu']}; {meta['blender']}; median of {meta['runs']} after a warm-up.\n\n{table}\n", encoding="utf-8")
    print("\n" + table)


if __name__ == "__main__":
    main()
