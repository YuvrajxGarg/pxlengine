# VFX: CG over real footage

    "plate": {"file": "shot.mp4", "offset": 0, "matte": "roto_####.png", "shadow": 1},
    {"type":"camera", "track": "shot.chan"}          // matchmoved camera

1. Plate: video, `plate_####.png` sequence or one still; scaled to `render.resolution`.
   Scene frame f shows plate frame f + `offset`. Plate pixels pass through untouched
   except where CG covers them or CG shadows fall.
2. Camera: `track` = Nuke .chan (`frame tx ty tz rx ry rz [vfov]`, degrees, rotation
   order ZXY; exported by Blender, SynthEyes, 3DEqualizer, PFTrack). Or set
   `rotation` [rx,ry,rz] / `target` by hand for a locked-off shot. Match `fov`.
3. Proxies for the real world, by material `role`:
   - `"shadow_catcher"`: ground/walls. Invisible to the camera; CG shadows on it darken
     the plate. Give it the real surface's colour: CG reflects and bounces off it.
   - `"holdout"`: real objects in front of CG (pillars, people's bodies). Invisible to
     the camera and they hide CG behind them; they still cast shadows on CG.
4. Light CG like the plate: an HDRI shot on set (`world.hdri`, `rotation` to align the
   sun) or lights matching the plate's shadows.
5. `matte` (white = real foreground) puts the plate back over everything (roto).

Output: PNG/video are composited in display space (CG through `render.view`, plate
unchanged). EXR is composited in linear light. Without a plate, `"transparent":true`
gives the CG layer alone, and `--passes shadow,alpha` the mattes for another compositor.

Check the lock: render previs over the whole range; CG must not slide on the ground.
`pxl track scene.json --out cam.chan` writes any pxl camera as a .chan (round trips).
Example: examples/vfx/plate-shot.json (makes a test plate + track),
examples/vfx/composite.json (puts CG into it).
