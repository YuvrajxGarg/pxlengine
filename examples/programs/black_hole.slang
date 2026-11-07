// Schwarzschild black hole with a thin emissive accretion disk and a procedural starfield.
//
// Light propagation hook: rays are integrated along photon orbits of the Schwarzschild
// metric using the Cartesian form  x'' = -3/2 * rs * h^2 * x / r^5  (h = |x × x'|),
// which reproduces null geodesics for a non-rotating hole. Each integration step is a
// short straight chord that is also traced against the scene, so geometry is lensed too.
// A numerical approximation: no frequency shift of the background, thin-disk only.
//
// Params (scene `programs.params`):
//   center, rs                        hole position and Schwarzschild radius
//   disk_inner, disk_outer            disk radii (rs = 1: ISCO is 3)
//   disk_brightness, disk_opacity     emission scale, optical thickness
//   disk_speed                        animation speed of the turbulence
//   star_density, escape              fraction of sky cells with a star, escape radius

float hash13(float3 p) {
    p = frac(p * 0.1031);
    p += dot(p, p.zyx + 31.32);
    return frac((p.x + p.y) * p.z);
}

float vnoise(float3 p) {
    float3 i = floor(p), f = frac(p);
    f = f * f * (3 - 2 * f);
    float n000 = hash13(i), n100 = hash13(i + float3(1, 0, 0));
    float n010 = hash13(i + float3(0, 1, 0)), n110 = hash13(i + float3(1, 1, 0));
    float n001 = hash13(i + float3(0, 0, 1)), n101 = hash13(i + float3(1, 0, 1));
    float n011 = hash13(i + float3(0, 1, 1)), n111 = hash13(i + float3(1, 1, 1));
    return lerp(lerp(lerp(n000, n100, f.x), lerp(n010, n110, f.x), f.y),
                lerp(lerp(n001, n101, f.x), lerp(n011, n111, f.x), f.y), f.z);
}

float fbm(float3 p) {
    float a = 0.5, s = 0;
    for (int i = 0; i < 5; i++) { s += a * vnoise(p); p = p * 2.03 + 17.1; a *= 0.5; }
    return s;
}

float3 starfield(float3 d) {
    float3 col = 0;
    for (int layer = 0; layer < 2; layer++) {
        float scale = layer == 0 ? 160.0 : 420.0;
        float3 q = d * scale;
        float3 c = floor(q);
        if (hash13(c) > 1 - P.star_density) {
            float3 sp = c + 0.25 + 0.5 * float3(hash13(c + 1.3), hash13(c + 7.1), hash13(c + 3.7));
            float dist = length(q - sp);
            float b = exp(-dist * dist * 14) * pow(hash13(c + 11.0), 5) * 60;
            col += b * lerp(float3(1.0, 0.72, 0.5), float3(0.7, 0.82, 1.0), hash13(c + 5.0));
        }
    }
    float band = exp(-pow(dot(d, normalize(float3(0.35, 1, 0.2))) * 3.5, 2));
    col += band * 0.025 * float3(0.75, 0.8, 1.0) * (0.3 + fbm(d * 7));
    return col;
}

/// Emission (rgb) and opacity (a) of the disk at hole-space point p, seen along ray dir.
float4 disk(float3 p, float3 dir) {
    float r = length(p.xz);
    float u = (r - P.disk_inner) / (P.disk_outer - P.disk_inner);
    // Keplerian angular velocity: inner gas turns faster, which shears the turbulence.
    float omega = P.disk_speed / pow(r / P.rs, 1.5);
    float a = -time() * omega;
    float2 q = float2(p.x * cos(a) - p.z * sin(a), p.x * sin(a) + p.z * cos(a));
    float n = fbm(float3(q * 1.6, r * 0.7));
    float streaks = 0.5 + 0.5 * sin(r * 9 + n * 6);
    float temp = pow(P.disk_inner / r, 0.75);                   // hottest at the inner edge
    // Hot inner gas is near-white-yellow, cooler outer gas deep orange-red.
    float3 col = lerp(float3(1.0, 0.18, 0.02), float3(1.0, 0.62, 0.22), saturate(temp * temp)) * temp * temp;
    // Relativistic beaming: gas moving towards the camera is brighter (D^3).
    float3 tangent = normalize(float3(-p.z, 0, p.x));
    float beta = sqrt(P.rs / (2 * r));
    float D = 1 / max(1 - beta * dot(tangent, -dir), 0.05);
    float edge = smoothstep(0.0, 0.06, u) * (1 - smoothstep(0.6, 1.0, u));
    float density = edge * (0.35 + 0.65 * n) * (0.6 + 0.4 * streaks);
    return float4(col * P.disk_brightness * D * D * D * (0.4 + n), saturate(density * P.disk_opacity));
}

float3 bh_accel(float3 x, float h2) {
    float r2 = dot(x, x);
    return -1.5 * P.rs * h2 * x / (r2 * r2 * sqrt(r2));
}

Hit black_hole(inout Ray ray, inout float3 radiance, inout float3 throughput, inout uint rng) {
    float3 x = ray.origin - P.center;
    float3 v = normalize(ray.dir);
    float3 L = cross(x, v);
    float h2 = dot(L, L);
    for (int i = 0; i < 600; i++) {
        float r = length(x);
        if (r < P.rs * 1.0001) {                     // crossed the event horizon
            throughput = 0;
            return no_hit();
        }
        if (r > P.escape && dot(x, v) > 0) break;    // escaping: bending is negligible now
        float dt = clamp(0.06 * (r - P.rs) + 0.01 * P.rs, 0.005 * P.rs, 0.5 * P.escape);
        float3 a0 = bh_accel(x, h2);                 // velocity Verlet
        float3 xn = x + v * dt + 0.5 * a0 * dt * dt;
        float3 vn = normalize(v + 0.5 * (a0 + bh_accel(xn, h2)) * dt);
        float3 seg = xn - x;
        float len = length(seg);
        float3 dir = seg / len;

        Hit h = trace_segment(x + P.center, dir, len);
        if (h.hit) {                                 // geometry on this chord: shade it
            ray.origin = x + P.center;
            ray.dir = dir;
            return h;
        }
        if (x.y * xn.y <= 0 && x.y != xn.y) {        // crossed the disk plane
            float3 p = lerp(x, xn, x.y / (x.y - xn.y));
            float pr = length(p.xz);
            if (pr > P.disk_inner && pr < P.disk_outer) {
                float4 e = disk(p, dir);
                radiance += throughput * e.rgb * e.a;
                throughput *= 1 - e.a;
                if (max(throughput.x, max(throughput.y, throughput.z)) < 0.003) {
                    throughput = 0;
                    return no_hit();
                }
            }
        }
        x = xn;
        v = vn;
    }
    radiance += throughput * starfield(v);           // the sky, seen along the bent ray
    throughput = 0;
    ray.dir = v;
    return no_hit();
}
