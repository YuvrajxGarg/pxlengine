// pxlengine path tracer: inline ray queries from compute.
// One dispatch = one sample per pixel, accumulated into `accum`.
//
// Lighting: next-event estimation over three light categories (analytic lights,
// emissive triangles, HDRI environment) combined with BSDF sampling via MIS
// (power heuristic). Analytic point/directional lights are delta lights; rectangle
// lights are not hittable by rays, so they need no MIS.

static const float PI = 3.14159265358979;
static const uint NONE = 0xffffffffu;

struct Instance { uint4 meta; float4 n0; float4 n1; float4 n2; };   // vertexOffset, indexOffset, material, emissiveOffset; n0.w = object id
struct Material {
    float4 base_rough;   // base colour rgb, roughness
    float4 emit_metal;   // emission rgb (pre-scaled), metalness
    int4 tex0;           // base colour, roughness, metalness, emission texture (-1 = none)
    int4 tex1;           // normal texture, roughness channel, metalness channel, alpha mode (1 = mask)
    float4 p0;           // normal scale, alpha cutoff, uv scale x, uv scale y
    float4 p1;           // base alpha
};
struct Light { float4 pos_type; float4 radiance; float4 u; float4 v; }; // 0 point, 1 rect, 2 directional
struct Frame {
    float4 cam_pos;
    float4 cam_fwd;
    float4 cam_right;    // pre-scaled by tan(fov/2)*aspect
    float4 cam_up;       // pre-scaled by tan(fov/2)
    float4 world;        // constant environment rgb, w = HDRI strength
    uint4 dims;          // width, height, light count, max bounces
    uint4 misc;          // seed, emissive triangle count, env width, env height
    float4 envp;         // env rotation (turns), env weight total, P(analytic), P(emissive)
    float4 envq;         // P(env), indirect clamp
    float4 lens;         // aperture radius (m), focus distance (m), time (s), frame
};
struct Push { uint sample_index; uint p0; uint p1; uint p2; };

[[vk::binding(0, 0)]]  RaytracingAccelerationStructure tlas;
[[vk::binding(1, 0)]]  StructuredBuffer<float4> verts;        // 2 per vertex: (position, u), (normal, v)
[[vk::binding(2, 0)]]  StructuredBuffer<uint> indices;
[[vk::binding(3, 0)]]  StructuredBuffer<Instance> instances;
[[vk::binding(4, 0)]]  StructuredBuffer<Material> materials;
[[vk::binding(5, 0)]]  StructuredBuffer<Light> lights;
[[vk::binding(6, 0)]]  RWStructuredBuffer<float4> accum;
[[vk::binding(7, 0)]]  StructuredBuffer<uint4> tex_info;      // texel offset, width, height, -
[[vk::binding(8, 0)]]  StructuredBuffer<uint> texels;         // RGBA8
[[vk::binding(9, 0)]]  StructuredBuffer<float4> emissive;     // 4 per triangle: (p0,u0) (p1,v0) (p2,u1) (v1,u2,v2,material)
[[vk::binding(10, 0)]] StructuredBuffer<float> emissive_cdf;  // count + 1
[[vk::binding(11, 0)]] StructuredBuffer<float> env_pix;       // RGB
[[vk::binding(12, 0)]] StructuredBuffer<float> env_marg;      // height + 1
[[vk::binding(13, 0)]] StructuredBuffer<float> env_cond;      // height * (width + 1)
[[vk::binding(14, 0)]] StructuredBuffer<Frame> frames;
[[vk::binding(15, 0)]] RWStructuredBuffer<float4> aux;         // 3 per pixel: (albedo, alpha) (normal, depth) (object id)
[[vk::binding(17, 0)]] StructuredBuffer<float4> params;        // scene program params, one float4 each
[[vk::push_constant]] ConstantBuffer<Push> pc;

static Frame F;

// ---------------------------------------------------------------- random
uint hash(uint x) {
    x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
    return x;
}
float rnd(inout uint s) {
    s = s * 747796405u + 2891336453u;
    uint w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    w = (w >> 22u) ^ w;
    return float(w >> 8) * (1.0 / 16777216.0);
}

float lum(float3 c) { return dot(c, float3(0.2126, 0.7152, 0.0722)); }
float mis(float a, float b) { float a2 = a * a; return a2 / max(a2 + b * b, 1e-20); }

// Wächter & Binder, "A Fast and Robust Method for Avoiding Self-Intersection" (Ray Tracing Gems).
float3 offset_ray(float3 p, float3 n) {
    const float origin = 1.0 / 32.0, float_scale = 1.0 / 65536.0, int_scale = 256.0;
    int3 o = int3(int_scale * n);
    float3 pi = float3(asfloat(asint(p.x) + (p.x < 0 ? -o.x : o.x)),
                       asfloat(asint(p.y) + (p.y < 0 ? -o.y : o.y)),
                       asfloat(asint(p.z) + (p.z < 0 ? -o.z : o.z)));
    return float3(abs(p.x) < origin ? p.x + float_scale * n.x : pi.x,
                  abs(p.y) < origin ? p.y + float_scale * n.y : pi.y,
                  abs(p.z) < origin ? p.z + float_scale * n.z : pi.z);
}

// ---------------------------------------------------------------- textures
float3 srgb_to_linear(float3 c) {
    return lerp(c / 12.92, pow((c + 0.055) / 1.055, float3(2.4)), step(float3(0.04045), c));
}

float4 texel(uint4 t, int x, int y) {
    int w = int(t.y), h = int(t.z);
    x = ((x % w) + w) % w;
    y = ((y % h) + h) % h;
    uint p = texels[t.x + uint(y * w + x)];
    return float4(float(p & 255u), float((p >> 8) & 255u), float((p >> 16) & 255u), float(p >> 24)) / 255.0;
}

float4 sample_tex(int idx, float2 uv) {
    if (idx < 0) return float4(1, 1, 1, 1);
    uint4 t = tex_info[idx];
    float2 st = uv * float2(float(t.y), float(t.z)) - 0.5;
    float2 f = floor(st);
    float2 fr = st - f;
    int x = int(f.x), y = int(f.y);
    float4 a = lerp(texel(t, x, y), texel(t, x + 1, y), fr.x);
    float4 b = lerp(texel(t, x, y + 1), texel(t, x + 1, y + 1), fr.x);
    return lerp(a, b, fr.y);
}

// ---------------------------------------------------------------- geometry
struct Tri { float3 p0; float3 p1; float3 p2; float3 n0; float3 n1; float3 n2; float2 uv0; float2 uv1; float2 uv2; };

Tri load_tri(Instance inst, uint prim) {
    uint io = inst.meta.y + prim * 3;
    uint vo = inst.meta.x;
    uint i0 = indices[io] + vo, i1 = indices[io + 1] + vo, i2 = indices[io + 2] + vo;
    float4 a0 = verts[i0 * 2], b0 = verts[i0 * 2 + 1];
    float4 a1 = verts[i1 * 2], b1 = verts[i1 * 2 + 1];
    float4 a2 = verts[i2 * 2], b2 = verts[i2 * 2 + 1];
    Tri t;
    t.p0 = a0.xyz; t.p1 = a1.xyz; t.p2 = a2.xyz;
    t.n0 = b0.xyz; t.n1 = b1.xyz; t.n2 = b2.xyz;
    t.uv0 = float2(a0.w, b0.w); t.uv1 = float2(a1.w, b1.w); t.uv2 = float2(a2.w, b2.w);
    return t;
}

bool alpha_pass(uint inst_id, uint prim, float2 bc) {
    Instance inst = instances[inst_id];
    Material m = materials[inst.meta.z];
    if (m.tex1.w == 0) return true;
    Tri t = load_tri(inst, prim);
    float2 uv = (t.uv0 * (1 - bc.x - bc.y) + t.uv1 * bc.x + t.uv2 * bc.y) * m.p0.zw;
    return m.p1.x * sample_tex(m.tex0.x, uv).a >= m.p0.y;
}

struct Hit { bool hit; uint inst; uint prim; float2 bc; float t; float3x4 o2w; };

Hit trace(float3 o, float3 d) { return trace_segment(o, d, 1e30); }

/// Closest hit along o + t*d for t in [0, tmax]. Programs use this to march curved rays.
Hit trace_segment(float3 o, float3 d, float tmax) {
    RayQuery<RAY_FLAG_NONE> q;
    RayDesc r; r.Origin = o; r.Direction = d; r.TMin = 0.0; r.TMax = tmax;
    q.TraceRayInline(tlas, RAY_FLAG_NONE, 0xff, r);
    while (q.Proceed()) {
        if (q.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE &&
            alpha_pass(q.CandidateInstanceID(), q.CandidatePrimitiveIndex(), q.CandidateTriangleBarycentrics()))
            q.CommitNonOpaqueTriangleHit();
    }
    Hit h;
    h.hit = q.CommittedStatus() == COMMITTED_TRIANGLE_HIT;
    h.inst = 0; h.prim = 0; h.bc = 0; h.t = 0; h.o2w = float3x4(0);
    if (!h.hit) return h;
    h.inst = q.CommittedInstanceID();
    h.prim = q.CommittedPrimitiveIndex();
    h.bc = q.CommittedTriangleBarycentrics();
    h.t = q.CommittedRayT();
    h.o2w = q.CommittedObjectToWorld3x4();
    return h;
}

bool visible(float3 o, float3 d, float tmax) {
    RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> q;
    RayDesc r; r.Origin = o; r.Direction = d; r.TMin = 0.0; r.TMax = tmax;
    q.TraceRayInline(tlas, RAY_FLAG_NONE, 0xff, r);
    while (q.Proceed()) {
        if (q.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE &&
            alpha_pass(q.CandidateInstanceID(), q.CandidatePrimitiveIndex(), q.CandidateTriangleBarycentrics()))
            q.CommitNonOpaqueTriangleHit();
    }
    return q.CommittedStatus() == COMMITTED_NOTHING;
}

struct Surface {
    float3 p; float3 n; float3 ng;
    float3 base; float rough; float metal; float3 emission;
    uint emissive_index;   // index into emissive triangle list, NONE if not a light
};

Surface make_surface(Hit h, float3 d) {
    Instance inst = instances[h.inst];
    Material m = materials[inst.meta.z];
    Tri t = load_tri(inst, h.prim);
    float w0 = 1 - h.bc.x - h.bc.y;
    float2 uv = (t.uv0 * w0 + t.uv1 * h.bc.x + t.uv2 * h.bc.y) * m.p0.zw;
    float3x3 M = float3x3(h.o2w[0].xyz, h.o2w[1].xyz, h.o2w[2].xyz);
    float3x3 N = float3x3(inst.n0.xyz, inst.n1.xyz, inst.n2.xyz);
    float3 e1 = t.p1 - t.p0, e2 = t.p2 - t.p0;

    Surface s;
    s.p = mul(h.o2w, float4(t.p0 * w0 + t.p1 * h.bc.x + t.p2 * h.bc.y, 1));
    s.ng = normalize(mul(N, cross(e1, e2)));
    s.n = normalize(mul(N, t.n0 * w0 + t.n1 * h.bc.x + t.n2 * h.bc.y));
    if (dot(s.ng, d) > 0) s.ng = -s.ng;
    if (dot(s.n, s.ng) < 0) s.n = -s.n;

    float4 bc = sample_tex(m.tex0.x, uv);
    s.base = m.base_rough.rgb * (m.tex0.x >= 0 ? srgb_to_linear(bc.rgb) : float3(1));
    s.rough = m.base_rough.w * (m.tex0.y >= 0 ? sample_tex(m.tex0.y, uv)[m.tex1.y] : 1.0);
    s.metal = m.emit_metal.w * (m.tex0.z >= 0 ? sample_tex(m.tex0.z, uv)[m.tex1.z] : 1.0);
    s.rough = clamp(s.rough, 0.02, 1.0);
    s.metal = saturate(s.metal);
    s.emission = m.emit_metal.rgb * (m.tex0.w >= 0 ? srgb_to_linear(sample_tex(m.tex0.w, uv).rgb) : float3(1));
    s.emissive_index = inst.meta.w == NONE ? NONE : inst.meta.w + h.prim;

    if (m.tex1.x >= 0) {   // tangent-space normal map
        float2 d1 = t.uv1 - t.uv0, d2 = t.uv2 - t.uv0;
        float det = d1.x * d2.y - d1.y * d2.x;
        if (abs(det) > 1e-12) {
            float r = 1.0 / det;
            float3 T = mul(M, (e1 * d2.y - e2 * d1.y) * r);
            float3 B = mul(M, (e2 * d1.x - e1 * d2.x) * r);
            T = T - s.n * dot(s.n, T);
            if (dot(T, T) > 1e-20) {
                T = normalize(T);
                float hand = dot(cross(s.n, T), B) < 0 ? -1.0 : 1.0;
                float3 Bn = cross(s.n, T) * hand;
                float3 tn = sample_tex(m.tex1.x, uv).xyz * 2 - 1;
                tn.xy *= m.p0.x;
                float3 nn = normalize(T * tn.x + Bn * tn.y + s.n * max(tn.z, 1e-3));
                if (dot(nn, s.ng) > 0) s.n = nn;
            }
        }
    }
    return s;
}

// ---------------------------------------------------------------- BSDF (diffuse + GGX)
float3 fresnel(float3 f0, float c) { return f0 + (1 - f0) * pow(1 - c, 5); }
float ggx_d(float nh, float a2) { float d = nh * nh * (a2 - 1) + 1; return a2 / (PI * d * d); }

float3 eval_brdf(Surface s, float3 v, float3 l) {
    float nl = dot(s.n, l), nv = dot(s.n, v);
    if (nl <= 0 || nv <= 0 || dot(s.ng, l) <= 0) return 0;
    float3 h = normalize(v + l);
    float nh = saturate(dot(s.n, h)), vh = saturate(dot(v, h));
    float a = max(s.rough * s.rough, 2e-3), a2 = a * a;
    float k = a * 0.5;
    float G = (nl / (nl * (1 - k) + k)) * (nv / (nv * (1 - k) + k));
    float3 Fr = fresnel(lerp(float3(0.04), s.base, s.metal), vh);
    float3 spec = ggx_d(nh, a2) * G * Fr / (4 * nl * nv);
    float3 diff = (1 - Fr) * (1 - s.metal) * s.base / PI;
    return diff + spec;
}

float spec_prob(Surface s) { return lerp(0.3, 1.0, s.metal); }

float pdf_brdf(Surface s, float3 v, float3 l) {
    float nl = dot(s.n, l);
    if (nl <= 0) return 0;
    float3 h = normalize(v + l);
    float a = max(s.rough * s.rough, 2e-3);
    float ps = spec_prob(s);
    float nh = saturate(dot(s.n, h));
    float pdf_s = ggx_d(nh, a * a) * nh / (4 * max(dot(v, h), 1e-4));
    return ps * pdf_s + (1 - ps) * nl / PI;
}

void basis(float3 n, out float3 t, out float3 b) {
    float sg = n.z >= 0 ? 1.0 : -1.0;
    float a = -1.0 / (sg + n.z);
    float c = n.x * n.y * a;
    t = float3(1 + sg * n.x * n.x * a, sg * c, -sg * n.x);
    b = float3(c, sg + n.y * n.y * a, -n.y);
}

float3 sample_brdf(Surface s, float3 v, inout uint rng) {
    float3 t, b; basis(s.n, t, b);
    float u1 = rnd(rng), u2 = rnd(rng);
    if (rnd(rng) < spec_prob(s)) {
        float a = max(s.rough * s.rough, 2e-3);
        float phi = 2 * PI * u1;
        float ct = sqrt((1 - u2) / (1 + (a * a - 1) * u2));
        float st = sqrt(max(0, 1 - ct * ct));
        float3 h = t * (st * cos(phi)) + b * (st * sin(phi)) + s.n * ct;
        return reflect(-v, h);
    }
    float r = sqrt(u1), phi = 2 * PI * u2;
    return normalize(t * (r * cos(phi)) + b * (r * sin(phi)) + s.n * sqrt(max(0, 1 - u1)));
}

// ---------------------------------------------------------------- environment
bool has_env() { return F.misc.z > 0; }

float2 dir_to_uv(float3 d) {
    float u = frac(0.5 + atan2(d.x, -d.z) / (2 * PI) - F.envp.x);
    float v = acos(clamp(d.y, -1.0, 1.0)) / PI;
    return float2(u, v);
}

float3 uv_to_dir(float2 uv) {
    float phi = (frac(uv.x + F.envp.x) - 0.5) * 2 * PI;
    float th = uv.y * PI;
    return float3(sin(th) * sin(phi), cos(th), -sin(th) * cos(phi));
}

float3 env_pixel(uint x, uint y) {
    uint i = (y * F.misc.z + x) * 3;
    return float3(env_pix[i], env_pix[i + 1], env_pix[i + 2]);
}

float3 env_radiance(float3 d) {
    if (!has_env()) return F.world.rgb;
    float2 uv = dir_to_uv(d);
    uint x = min(uint(uv.x * F.misc.z), F.misc.z - 1), y = min(uint(uv.y * F.misc.w), F.misc.w - 1);
    return env_pixel(x, y) * F.world.w;
}

float env_pdf(float3 d) {
    float2 uv = dir_to_uv(d);
    uint w = F.misc.z, h = F.misc.w;
    uint x = min(uint(uv.x * w), w - 1), y = min(uint(uv.y * h), h - 1);
    float row_sin = sin(PI * (float(y) + 0.5) / float(h));
    float weight = max(lum(env_pixel(x, y)), 0.0) * row_sin;
    float st = sqrt(max(0, 1 - d.y * d.y));
    if (st < 1e-6 || F.envp.y <= 0) return 0;
    return weight / F.envp.y * float(w * h) / (2 * PI * PI * st);
}

uint search_marg(uint n, float u) {
    uint lo = 0, hi = n;
    while (lo + 1 < hi) { uint mid = (lo + hi) / 2; if (env_marg[mid] <= u) lo = mid; else hi = mid; }
    return lo;
}
uint search_cond(uint base, uint n, float u) {
    uint lo = 0, hi = n;
    while (lo + 1 < hi) { uint mid = (lo + hi) / 2; if (env_cond[base + mid] <= u) lo = mid; else hi = mid; }
    return lo;
}
uint search_emissive(uint n, float u) {
    uint lo = 0, hi = n;
    while (lo + 1 < hi) { uint mid = (lo + hi) / 2; if (emissive_cdf[mid] <= u) lo = mid; else hi = mid; }
    return lo;
}

// ---------------------------------------------------------------- emissive triangles
float3 emissive_radiance(uint mat, float2 uv) {
    Material m = materials[mat];
    return m.emit_metal.rgb * (m.tex0.w >= 0 ? srgb_to_linear(sample_tex(m.tex0.w, uv * m.p0.zw).rgb) : float3(1));
}

// Solid-angle pdf of having chosen emissive triangle `i` and point at distance `dist` with cosine `cl`.
float emissive_pdf(uint i, float dist, float cl) {
    float4 a = emissive[i * 4], b = emissive[i * 4 + 1], c = emissive[i * 4 + 2];
    float area = 0.5 * length(cross(b.xyz - a.xyz, c.xyz - a.xyz));
    float prob = emissive_cdf[i + 1] - emissive_cdf[i];
    return F.envp.w * prob * dist * dist / max(area * cl, 1e-12);
}

// ---------------------------------------------------------------- next-event estimation
float3 direct_light(Surface s, float3 v, inout uint rng) {
    float pa = F.envp.z, pe = F.envp.w, pv = F.envq.x;
    float r = rnd(rng);
    float3 o = offset_ray(s.p, s.ng);

    if (r < pa) {                                     // analytic lights
        uint li = min(uint(rnd(rng) * F.dims.z), F.dims.z - 1);
        Light L = lights[li];
        float weight = float(F.dims.z) / pa;
        uint type = uint(L.pos_type.w);
        if (type == 2) {
            float3 l = -L.u.xyz;
            float3 f = eval_brdf(s, v, l);
            if (any(f > 0) && visible(o, l, 1e30)) return L.radiance.rgb * f * dot(s.n, l) * weight;
            return 0;
        }
        float3 lp = L.pos_type.xyz;
        if (type == 1) lp += L.u.xyz * (2 * rnd(rng) - 1) + L.v.xyz * (2 * rnd(rng) - 1);
        float3 dl = lp - o;
        float d2 = max(dot(dl, dl), 1e-8);
        float dist = sqrt(d2);
        float3 l = dl / dist;
        float area_term = 1;
        if (type == 1) {
            float3 ln = normalize(cross(L.u.xyz, L.v.xyz));
            float cl = dot(ln, -l);
            if (cl <= 0) return 0;
            area_term = cl * 4 * length(L.u.xyz) * length(L.v.xyz);
        }
        float3 f = eval_brdf(s, v, l);
        if (any(f > 0) && visible(o, l, dist * 0.999))
            return L.radiance.rgb * area_term / d2 * f * dot(s.n, l) * weight;
        return 0;
    }

    if (r < pa + pe) {                                // emissive triangles
        uint i = search_emissive(F.misc.y, rnd(rng));
        float4 a = emissive[i * 4], b = emissive[i * 4 + 1], c = emissive[i * 4 + 2], e = emissive[i * 4 + 3];
        float su = sqrt(rnd(rng)), u2 = rnd(rng);
        float b0 = 1 - su, b1 = u2 * su, b2 = 1 - b0 - b1;
        float3 x = a.xyz * b0 + b.xyz * b1 + c.xyz * b2;
        float2 uv = float2(a.w, b.w) * b0 + float2(c.w, e.x) * b1 + float2(e.y, e.z) * b2;
        float3 nl = normalize(cross(b.xyz - a.xyz, c.xyz - a.xyz));
        float3 dl = x - o;
        float dist = length(dl);
        if (dist < 1e-6) return 0;
        float3 l = dl / dist;
        float cl = abs(dot(nl, l));
        if (cl < 1e-6) return 0;
        float pdf_l = emissive_pdf(i, dist, cl);
        float3 f = eval_brdf(s, v, l);
        if (all(f <= 0) || !visible(o, l, dist * (1 - 1e-3))) return 0;
        float3 Le = emissive_radiance(asuint(e.w), uv);
        return Le * f * dot(s.n, l) * mis(pdf_l, pdf_brdf(s, v, l)) / pdf_l;
    }

    if (pv > 0) {                                     // HDRI environment
        uint w = F.misc.z, h = F.misc.w;
        uint y = search_marg(h, rnd(rng));
        uint x = search_cond(y * (w + 1), w, rnd(rng));
        float2 uv = float2((float(x) + rnd(rng)) / float(w), (float(y) + rnd(rng)) / float(h));
        float3 l = uv_to_dir(uv);
        float pdf_l = pv * env_pdf(l);
        if (pdf_l <= 0) return 0;
        float3 f = eval_brdf(s, v, l);
        if (all(f <= 0) || !visible(o, l, 1e30)) return 0;
        return env_radiance(l) * f * dot(s.n, l) * mis(pdf_l, pdf_brdf(s, v, l)) / pdf_l;
    }
    return 0;
}

// ---------------------------------------------------------------- program interface
// Types and helpers available to scene programs (see `pxl help programs`).

struct Ray { float3 origin; float3 dir; };

struct CameraIn {
    float2 ndc;            // -1..1, +y up
    float2 pixel;          // pixel coordinate (with sub-pixel jitter)
    float2 resolution;
    float3 position;       // camera position
    float3 forward;        // unit vectors
    float3 right;
    float3 up;
    float tan_half_fov;    // vertical
    float aspect;
};

struct SurfaceIn {
    float3 position;       // world space
    float3 object_position;// object space: procedural patterns stick to moving objects
    float3 geometric_normal;
    float2 uv;
    float3 view_dir;       // towards the viewer
    uint object;           // scene node index
};

struct FilmIn {
    float3 color;          // linear HDR (after denoising)
    float alpha;
    uint2 pixel;
    uint2 resolution;
    float3 normal;
    float depth;
    float3 albedo;
};

float time() { return F.lens.z; }
Hit no_hit() { Hit h; h.hit = false; h.inst = 0; h.prim = 0; h.bc = 0; h.t = 0; h.o2w = float3x4(0); return h; }
float frame_number() { return F.lens.w; }

SurfaceIn surface_input(Hit h, float3 d, Surface s) {
    Instance inst = instances[h.inst];
    Tri t = load_tri(inst, h.prim);
    float w0 = 1 - h.bc.x - h.bc.y;
    SurfaceIn i;
    i.position = s.p;
    i.object_position = t.p0 * w0 + t.p1 * h.bc.x + t.p2 * h.bc.y;
    i.geometric_normal = s.ng;
    i.uv = t.uv0 * w0 + t.uv1 * h.bc.x + t.uv2 * h.bc.y;
    i.view_dir = -d;
    i.object = asuint(inst.n0.w);
    return i;
}

uint material_program(Hit h) { return asuint(materials[instances[h.inst].meta.z].p1.y); }

// Film pass resources: declared always so any program file compiles in any scene; only the
// film pass (PXL_HOOK_FILM) binds them.
[[vk::binding(16, 0)]] StructuredBuffer<float4> film_in;     // 3 per pixel: (color, alpha) (normal, depth) (albedo, -)
[[vk::binding(18, 0)]] RWStructuredBuffer<float4> film_out;  // 1 per pixel
struct FilmPush { uint width; uint height; float time; float frame; };
[[vk::push_constant]] ConstantBuffer<FilmPush> film_pc;

/// Film programs: linear colour of another pixel (clamped to the image), for blurs,
/// glows and distortion. Only meaningful inside a film program.
float3 film_color(int2 p) {
    p = clamp(p, int2(0, 0), int2(film_pc.width, film_pc.height) - 1);
    return film_in[(uint(p.y) * film_pc.width + uint(p.x)) * 3].rgb;
}

//@PXL_USER_CODE

// ---------------------------------------------------------------- main
[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 id: SV_DispatchThreadID) {
    F = frames[0];
    pxl_load_params();
    if (id.x >= F.dims.x || id.y >= F.dims.y) return;
    uint idx = id.y * F.dims.x + id.x;
    uint rng = hash(idx ^ hash(pc.sample_index * 0x9e3779b9u + F.misc.x * 0x85ebca6bu));

    float2 jitter = float2(rnd(rng), rnd(rng));
    float2 uv = (float2(id.xy) + jitter) / float2(F.dims.xy) * 2 - 1;
    float3 o = F.cam_pos.xyz;
    float3 d = normalize(F.cam_fwd.xyz + uv.x * F.cam_right.xyz - uv.y * F.cam_up.xyz);
    if (F.lens.x > 0) {                               // thin lens: sample the aperture, aim at the focal plane
        float3 focus = o + d * (F.lens.y / dot(d, F.cam_fwd.xyz));
        float r = F.lens.x * sqrt(rnd(rng)), a = 2 * PI * rnd(rng);
        o += normalize(F.cam_right.xyz) * (r * cos(a)) + normalize(F.cam_up.xyz) * (r * sin(a));
        d = normalize(focus - o);
    }
    float4 aov0 = float4(0), aov1 = float4(0);
    uint object_id = NONE;
    float3 radiance = 0, thr = 1;

#ifdef PXL_HOOK_CAMERA
    {
        CameraIn ci;
        ci.ndc = float2(uv.x, -uv.y);
        ci.pixel = float2(id.xy) + jitter;
        ci.resolution = float2(F.dims.xy);
        ci.position = F.cam_pos.xyz;
        ci.forward = F.cam_fwd.xyz;
        ci.right = normalize(F.cam_right.xyz);
        ci.up = normalize(F.cam_up.xyz);
        ci.tan_half_fov = length(F.cam_up.xyz);
        ci.aspect = length(F.cam_right.xyz) / max(length(F.cam_up.xyz), 1e-8);
        if (!pxl_hook_camera(ci, o, d)) thr = 0;
        d = normalize(d);
    }
#endif

    float prev_pdf = 0;       // BSDF pdf of the ray that reached this vertex (0 = camera ray)
    float clamp_v = F.envq.y;
    for (uint bounce = 0; bounce <= F.dims.w && any(thr > 0); bounce++) {
#ifdef PXL_HOOK_PROPAGATE
        Ray pr; pr.origin = o; pr.dir = d;
        float3 emitted = 0;
        Hit h = pxl_hook_propagate(pr, emitted, thr, rng);
        d = normalize(pr.dir);
        radiance += bounce > 1 ? min(emitted, float3(clamp_v)) : emitted;
        if (all(thr <= 0)) {
            if (bounce == 0) aov0 = float4(0, 0, 0, h.hit ? 1 : 0);
            break;
        }
#else
        Hit h = trace(o, d);
#endif
        if (!h.hit) {
            float3 Le = env_radiance(d);
            if (bounce == 0) aov0 = float4(saturate(Le), 0);
            float w = (prev_pdf > 0 && has_env() && F.envq.x > 0) ? mis(prev_pdf, F.envq.x * env_pdf(d)) : 1.0;
            float3 c = thr * Le * w;
            radiance += bounce > 1 ? min(c, float3(clamp_v)) : c;
            break;
        }
        Surface s = make_surface(h, d);
        float3 v = -d;
        uint prog = material_program(h);
        if (prog != 0) {
            pxl_run_material_program(prog, s, surface_input(h, d, s));
            s.rough = clamp(s.rough, 0.02, 1.0);
            s.metal = saturate(s.metal);
            s.n = normalize(s.n);
            if (dot(s.n, s.ng) < 0) s.n = normalize(s.n - s.ng * (dot(s.n, s.ng) - 1e-3));
        }
        if (bounce == 0) {
            aov0 = float4(s.base, 1);
            aov1 = float4(s.n, dot(s.p - F.cam_pos.xyz, F.cam_fwd.xyz));
            object_id = asuint(instances[h.inst].n0.w);
        }

        if (any(s.emission > 0)) {
            float w = 1;
            if (prev_pdf > 0 && s.emissive_index != NONE && F.envp.w > 0)
                w = mis(prev_pdf, emissive_pdf(s.emissive_index, h.t, abs(dot(s.ng, d))));
            float3 c = thr * s.emission * w;
            radiance += bounce > 1 ? min(c, float3(clamp_v)) : c;
        }
        if (bounce == F.dims.w) break;

        float3 c = thr * direct_light(s, v, rng);
        radiance += bounce > 0 ? min(c, float3(clamp_v)) : c;

        float3 l = sample_brdf(s, v, rng);
        float pdf = pdf_brdf(s, v, l);
        float3 f = eval_brdf(s, v, l);
        if (pdf <= 1e-8 || all(f <= 0)) break;
        thr *= f * dot(s.n, l) / pdf;
        prev_pdf = pdf;
        if (bounce >= 2) {                            // Russian roulette
            float pr = clamp(max(thr.r, max(thr.g, thr.b)), 0.05, 0.95);
            if (rnd(rng) > pr) break;
            thr /= pr;
        }
        o = offset_ray(s.p, s.ng);
        d = l;
    }
    if (any(isnan(radiance)) || any(isinf(radiance))) radiance = 0;
    bool first = pc.sample_index == 0;
    accum[idx] = (first ? float4(0) : accum[idx]) + float4(radiance, 1);
    aux[idx * 3] = (first ? float4(0) : aux[idx * 3]) + aov0;
    aux[idx * 3 + 1] = (first ? float4(0) : aux[idx * 3 + 1]) + aov1;
    if (first) aux[idx * 3 + 2] = float4(asfloat(object_id), 0, 0, 0);
}

// ---------------------------------------------------------------- film program pass
#ifdef PXL_HOOK_FILM
[shader("compute")]
[numthreads(8, 8, 1)]
void film_main(uint3 id: SV_DispatchThreadID) {
    F.lens = float4(0, 0, film_pc.time, film_pc.frame);
    F.dims = uint4(film_pc.width, film_pc.height, 0, 0);
    pxl_load_params();
    if (id.x >= film_pc.width || id.y >= film_pc.height) return;
    uint i = id.y * film_pc.width + id.x;
    FilmIn f;
    f.color = film_in[i * 3].rgb;
    f.alpha = film_in[i * 3].a;
    f.normal = film_in[i * 3 + 1].xyz;
    f.depth = film_in[i * 3 + 1].w;
    f.albedo = film_in[i * 3 + 2].rgb;
    f.pixel = id.xy;
    f.resolution = uint2(film_pc.width, film_pc.height);
    float3 c = pxl_hook_film(f);
    if (any(isnan(c)) || any(isinf(c))) c = 0;
    film_out[i] = float4(c, f.alpha);
}
#endif
