# Simulations: millions of GPU agents

A simulation is a population of agents stepped by your Slang functions every substep and
drawn as instances of one mesh (oriented along velocity). Flocks, swarms, particles,
debris, crowds.

    "simulations": [{
      "name": "starlings", "count": 1000000,
      "init": "flock_init", "step": "flock_step",      // functions in programs.source
      "cell_size": 1.6,      // neighbour grid cell ~ interaction radius (0 = none)
      "substeps": 2,         // steps per frame (dt = 1 / (fps * substeps))
      "warmup": 6,           // seconds simulated before time 0
      "geometry": {"std": "bird"},   // or "asset": "crow.glb" (first mesh); +Z forward, +Y up
      "scale": 0.4, "orient": true,
      "materials": [{"inputs": {"base_color": [0.07, 0.065, 0.06]}}]
    }]

## Functions

    void flock_init(uint id, inout Agent a, inout uint rng)   // once
    void flock_step(uint id, inout Agent a, inout uint rng)   // every substep

    struct Agent { float3 position; float size; float3 velocity; float age;
                   float4 color;   // a > 0 tints the base colour
                   float4 data; }  // yours

`SIM.dt`, `SIM.time` (scene seconds), `SIM.count`, `SIM.step`; `P.<param>`, `rnd(rng)`.
Init starts from position 0, size 1, everything else 0. Write the agent you were given;
read others with `agent(j)`: everyone sees the previous step's snapshot.

Neighbours (needs `cell_size`), scanning the 27 cells around a point:

    Neighbors nb = neighbors(a.position);
    uint j;
    while (next_neighbor(nb, j)) {
        if (j == id) continue;
        Agent b = agent(j);
        float3 d = b.position - a.position;
        if (dot(d, d) > R * R) continue;      // cells are hashed: always check distance
        ...
    }

Cap the neighbours you use (e.g. 24) in dense flocks; cost grows with density.

Rendering: every agent is a ray-traced instance (shadows, reflections, depth of field
and motion blur from velocity all work). Frames are simulated in order, so a still at
`--frame 200` first simulates frames 0..200. Example: examples/birds.json (100k),
examples/birds-1m.json (1M). Timings print as `sim ... + build ... + trace ...`.
