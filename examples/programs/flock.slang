// A murmuration: boids over a lagoon. Each agent steers by separation, alignment and
// cohesion over its grid neighbours, is drawn to a slowly wandering roost point, and is
// pushed by a smooth noise wind so the flock folds and pulses.
//
// Params: center, radius, perception, separation, alignment, cohesion, attraction,
//         wander, turbulence, min_speed, max_speed, min_height

float hash13(float3 p) {
    p = frac(p * 0.1031);
    p += dot(p, p.zyx + 31.32);
    return frac((p.x + p.y) * p.z);
}

float vnoise(float3 p) {
    float3 i = floor(p), f = frac(p);
    f = f * f * (3 - 2 * f);
    return lerp(lerp(lerp(hash13(i), hash13(i + float3(1, 0, 0)), f.x),
                     lerp(hash13(i + float3(0, 1, 0)), hash13(i + float3(1, 1, 0)), f.x), f.y),
                lerp(lerp(hash13(i + float3(0, 0, 1)), hash13(i + float3(1, 0, 1)), f.x),
                     lerp(hash13(i + float3(0, 1, 1)), hash13(i + float3(1, 1, 1)), f.x), f.y), f.z);
}

float3 random_in_sphere(inout uint rng) {
    for (int k = 0; k < 8; k++) {
        float3 p = float3(rnd(rng), rnd(rng), rnd(rng)) * 2 - 1;
        if (dot(p, p) <= 1) return p;
    }
    return 0;
}

void flock_init(uint id, inout Agent a, inout uint rng) {
    float3 r = random_in_sphere(rng);
    a.position = P.center + r * P.radius * float3(1, 0.3, 1);
    float ang = rnd(rng) * 2 * PI;
    a.velocity = float3(cos(ang), (rnd(rng) - 0.5) * 0.2, sin(ang)) * P.min_speed;
    a.size = 0.8 + 0.4 * rnd(rng);
    float shade = 0.55 + 0.45 * rnd(rng);                 // plumage variation (tints base colour)
    a.color = float4(shade, shade * (0.95 + 0.05 * rnd(rng)), shade * 0.95, 1);
    a.data = float4(rnd(rng), 0, 0, 0);
}

void flock_step(uint id, inout Agent a, inout uint rng) {
    float R = P.perception;
    float3 sep = 0, ali = 0, coh = 0;
    float n = 0;
    Neighbors nb = neighbors(a.position);
    uint j;
    int seen = 0;
    while (seen < 24 && next_neighbor(nb, j)) {
        if (j == id) continue;
        Agent b = agent(j);
        float3 d = b.position - a.position;
        float d2 = dot(d, d);
        if (d2 > R * R) continue;
        seen++;
        sep -= d / max(d2, 0.01);
        ali += b.velocity;
        coh += b.position;
        n += 1;
    }
    float3 acc = 0;
    if (n > 0) {
        acc += sep * P.separation;
        acc += (ali / n - a.velocity) * P.alignment;
        acc += (coh / n - a.position) * P.cohesion;
    }
    float t = SIM.time;
    float3 roost = P.center + float3(sin(t * 0.11) * P.wander, sin(t * 0.23) * 0.25 * P.wander, cos(t * 0.07) * P.wander);
    float3 to = roost - a.position;
    float dist = length(to);
    acc += to / max(dist, 1) * P.attraction * saturate(dist / P.radius);
    float3 q = a.position * 0.025 + t * 0.08;
    acc += (float3(vnoise(q), vnoise(q + 17.3), vnoise(q + 31.7)) - 0.5) * P.turbulence;
    if (a.position.y < P.min_height) acc.y += (P.min_height - a.position.y) * 3;

    a.velocity += acc * SIM.dt;
    float sp = length(a.velocity);
    a.velocity *= clamp(sp, P.min_speed, P.max_speed) / max(sp, 1e-4);
    a.position += a.velocity * SIM.dt;
}

/// Calm water: sum of directional waves with an analytic slope -> rippled normal.
void lagoon_water(inout Surface s, SurfaceIn i) {
    float2 p = i.position.xz;
    float2 grad = 0;
    // Fade ripples with distance: finer than a pixel they only alias into moire.
    float fade = saturate(40.0 / max(length(i.position - camera_position()), 1e-3));
    float amp = 0.035, freq = 0.35;
    for (int k = 0; k < 7; k++) {
        float a = 1.7 + k * 2.39996;                      // golden-angle directions
        float2 dir = float2(cos(a), sin(a));
        float ph = dot(dir, p) * freq + time() * (0.8 + 0.25 * k);
        grad += dir * freq * amp * cos(ph) * fade;
        amp *= 0.72;
        freq *= 1.83;
    }
    s.n = normalize(float3(-grad.x, 1, -grad.y));
    s.base = float3(0.015, 0.03, 0.035);
    s.rough = 0.04;
    s.metal = 0;
}
