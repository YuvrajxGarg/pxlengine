# Fluids: liquid simulation

A built-in FLIP liquid solver on the GPU. Declare it; no code needed.

    "fluids": [{
      "name": "water",
      "bounds": [[-1, 0, -0.5], [1, 1.2, 0.5]],     // closed container (walls are solid)
      "resolution": 64,                              // grid cells on the longest side (16..256)
      "fill": [{"min": [-0.97, 0.03, -0.47], "max": [-0.35, 0.95, 0.47], "velocity": [0, 0, 0]}],
      "substeps": 8,           // per frame; raise for fast splashes (keep motion < 1 cell/step)
      "flip": 0.95,            // 0 = smooth/viscous (PIC) .. 1 = lively/noisy (FLIP)
      "gravity": [0, -9.81, 0],
      "particles_per_cell": 8, "pressure_iterations": 80,
      "surface_refine": 2,     // mesh resolution vs grid
      "particle_radius": 2,    // surface smoothness (surface cells)
      "warmup": 0, "max_triangles": 2000000,
      "materials": [{"inputs": {"transmission_weight": 1, "specular_ior": 1.33,
                                "transmission_color": [0.45, 0.85, 0.92], "transmission_depth": 0.8}}]
    }]

The liquid is meshed every frame and rendered like any mesh (shadows, reflections,
refraction, motion through frames). Show the container with glass meshes just outside
`bounds`. Cost grows with resolution³ × substeps: 64 cells ≈ 12 ms per frame on an RTX 4080
Laptop; 128 cells is ~8× more. Frames simulate in order (a still at frame 100 simulates 0..100).

# Glass and water (any mesh)

Material inputs: `transmission_weight` 1 = fully transmissive, `specular_ior` (water 1.33,
glass 1.5), `specular_roughness` (0 clear, 0.2+ frosted), `transmission_color` +
`transmission_depth` (metres for that colour; 0 = thin tint). Glass lets shadows through
(no caustics). Use `render.bounces` 8..12 so light can pass in and out. glTF
KHR_materials_transmission / ior / volume import as glass.

Example: examples/dam-break.json, examples/glass.json.
