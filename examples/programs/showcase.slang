// Material, camera and film programs. See `pxl help programs`.

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

float fbm(float3 p) {
    float a = 0.5, s = 0;
    for (int i = 0; i < 6; i++) { s += a * vnoise(p); p = p * 2.02 + 11.7; a *= 0.5; }
    return s;
}

// ---------------------------------------------------------------- materials
// Signature: void name(inout Surface s, SurfaceIn i)
// Surface fields: p, n (shading normal), ng, base, rough, metal, emission.

/// White marble with dark veins, sticking to the object (object-space pattern).
void marble(inout Surface s, SurfaceIn i) {
    float3 q = i.object_position * P.marble_scale;
    float t = q.x + 2.5 * fbm(q * 0.8);
    float vein = pow(abs(sin(t * 3.14159)), 0.18);       // thin dark lines where sin ~ 0
    s.base = lerp(float3(0.12, 0.13, 0.15), float3(0.86, 0.85, 0.82), vein);
    s.rough = lerp(0.35, 0.08, vein);
}

/// Cooling lava: dark crust plates with glowing, slowly flowing cracks.
void lava(inout Surface s, SurfaceIn i) {
    float3 q = i.object_position * 2.0 + float3(0, -time() * 0.15, 0);
    // Domain-warped ridge noise: |2n-1| is ~0 along thin wandering lines -> cracks.
    float3 w = q + 0.7 * float3(fbm(q), fbm(q + 5.2), fbm(q + 9.1));
    float ridge = abs(2 * vnoise(w * 1.6) - 1);
    float crack = 1 - smoothstep(0.02, 0.1, ridge);
    float heat = crack * (0.6 + 0.4 * vnoise(q * 4 + time()));
    s.base = lerp(float3(0.05, 0.045, 0.04), float3(0.02, 0.01, 0.0), heat);
    s.rough = lerp(0.9, 0.4, heat);
    s.emission = float3(1.0, 0.28, 0.04) * heat * P.lava_glow;
}

/// Ceramic tiles: grout lines get a bevelled normal, so the bump catches light.
void tiles(inout Surface s, SurfaceIn i) {
    float2 uv = i.position.xz * P.tile_size;
    float2 f = frac(uv) - 0.5;
    float2 edge = smoothstep(0.42, 0.5, abs(f));
    float grout = max(edge.x, edge.y);
    float shade = hash13(float3(floor(uv), 3)) * 0.12;
    s.base = lerp(float3(0.62, 0.6, 0.56) - shade, float3(0.18, 0.17, 0.16), grout);
    s.rough = lerp(0.18, 0.8, grout);
    // Tilt the normal down towards the grout: d(height)/d(x,z).
    float2 slope = sign(f) * float2(edge.x, edge.y) * 0.6;
    s.n = normalize(s.n - float3(slope.x, 0, slope.y));
}

// ---------------------------------------------------------------- camera
// Signature: bool name(CameraIn c, inout float3 origin, inout float3 dir)

/// Equidistant fisheye with `P.fisheye_fov` degrees across the image diagonal... height.
bool fisheye(CameraIn c, inout float3 origin, inout float3 dir) {
    float2 p = float2(c.ndc.x * c.aspect, c.ndc.y);
    float r = length(p);
    float theta = r * radians(P.fisheye_fov) * 0.5;
    if (theta > 3.1) return false;                       // outside the lens circle: black
    float2 dir2 = r > 0 ? p / r : float2(0, 0);
    dir = c.forward * cos(theta) + (c.right * dir2.x + c.up * dir2.y) * sin(theta);
    return true;
}

// ---------------------------------------------------------------- film
// Signature: float3 name(FilmIn f); runs per pixel on the finished linear HDR frame.

float3 lens_finish(FilmIn f) {
    float2 res = float2(f.resolution);
    float2 uv = (float2(f.pixel) + 0.5) / res;
    float2 c = uv - 0.5;
    // Chromatic aberration: red and blue sampled slightly in and out of the centre.
    float2 shift = c * P.aberration * res;
    float3 col;
    col.r = film_color(int2(float2(f.pixel) - shift)).r;
    col.g = f.color.g;
    col.b = film_color(int2(float2(f.pixel) + shift)).b;
    // Vignette.
    col *= lerp(1.0, smoothstep(0.85, 0.2, length(c * float2(res.x / res.y, 1))), P.vignette);
    // Gentle split-tone: cool shadows, warm highlights.
    float l = dot(col, float3(0.2126, 0.7152, 0.0722));
    col *= lerp(float3(0.95, 1.0, 1.08), float3(1.05, 1.0, 0.94), saturate(l * 2));
    // Film grain, different every frame.
    float g = hash13(float3(float2(f.pixel), frame_number())) - 0.5;
    return max(col * (1 + g * P.grain), 0);
}
