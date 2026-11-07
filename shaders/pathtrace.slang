// pxlengine core path tracer: inline ray queries from compute.
// One dispatch = one sample per pixel, accumulated into `accum`.

static const float PI = 3.14159265358979;

struct Instance { uint4 meta; float4 n0; float4 n1; float4 n2; };   // meta: vertexOffset, indexOffset, material
struct Material { float4 base_rough; float4 emit_metal; };            // rgb+roughness, emission rgb + metalness
struct Light { float4 pos_type; float4 radiance; float4 u; float4 v; }; // type 0 point, 1 rect, 2 directional

struct Frame {
    float4 cam_pos;
    float4 cam_fwd;
    float4 cam_right;   // pre-scaled by tan(fov/2)*aspect
    float4 cam_up;      // pre-scaled by tan(fov/2)
    float4 world;       // environment radiance
    uint width; uint height; uint sample_index; uint light_count;
    uint seed; uint max_bounces; uint pad0; uint pad1;
};

[[vk::binding(0, 0)]] RaytracingAccelerationStructure tlas;
[[vk::binding(1, 0)]] StructuredBuffer<float4> verts;      // 2 per vertex: position, normal
[[vk::binding(2, 0)]] StructuredBuffer<uint> indices;
[[vk::binding(3, 0)]] StructuredBuffer<Instance> instances;
[[vk::binding(4, 0)]] StructuredBuffer<Material> materials;
[[vk::binding(5, 0)]] StructuredBuffer<Light> lights;
[[vk::binding(6, 0)]] RWStructuredBuffer<float4> accum;
[[vk::push_constant]] ConstantBuffer<Frame> frame;

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

// ---------------------------------------------------------------- tracing
struct Hit { bool hit; float3 p; float3 n; float3 ng; uint mat; };

Hit trace(float3 o, float3 d) {
    RayQuery<RAY_FLAG_FORCE_OPAQUE> q;
    RayDesc r; r.Origin = o; r.Direction = d; r.TMin = 0.0; r.TMax = 1e30;
    q.TraceRayInline(tlas, RAY_FLAG_NONE, 0xff, r);
    while (q.Proceed()) {}
    Hit h; h.hit = false; h.p = 0; h.n = 0; h.ng = 0; h.mat = 0;
    if (q.CommittedStatus() != COMMITTED_TRIANGLE_HIT) return h;

    Instance inst = instances[q.CommittedInstanceID()];
    uint io = inst.meta.y + q.CommittedPrimitiveIndex() * 3;
    uint vo = inst.meta.x;
    uint i0 = indices[io] + vo, i1 = indices[io + 1] + vo, i2 = indices[io + 2] + vo;
    float3 p0 = verts[i0 * 2].xyz, p1 = verts[i1 * 2].xyz, p2 = verts[i2 * 2].xyz;
    float3 n0 = verts[i0 * 2 + 1].xyz, n1 = verts[i1 * 2 + 1].xyz, n2 = verts[i2 * 2 + 1].xyz;
    float2 bc = q.CommittedTriangleBarycentrics();
    float3x3 N = float3x3(inst.n0.xyz, inst.n1.xyz, inst.n2.xyz);
    float3 ns = normalize(mul(N, n0 * (1 - bc.x - bc.y) + n1 * bc.x + n2 * bc.y));
    float3 ng = normalize(mul(N, cross(p1 - p0, p2 - p0)));
    // Face both normals toward the incoming ray.
    if (dot(ng, d) > 0) ng = -ng;
    if (dot(ns, ng) < 0) ns = -ns;
    h.hit = true;
    h.p = o + d * q.CommittedRayT();
    h.n = ns; h.ng = ng; h.mat = inst.meta.z;
    return h;
}

bool visible(float3 o, float3 d, float tmax) {
    RayQuery<RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> q;
    RayDesc r; r.Origin = o; r.Direction = d; r.TMin = 0.0; r.TMax = tmax;
    q.TraceRayInline(tlas, RAY_FLAG_NONE, 0xff, r);
    while (q.Proceed()) {}
    return q.CommittedStatus() == COMMITTED_NOTHING;
}

// ---------------------------------------------------------------- BRDF (diffuse + GGX)
float3 fresnel(float3 f0, float c) { return f0 + (1 - f0) * pow(1 - c, 5); }

float ggx_d(float nh, float a2) { float d = nh * nh * (a2 - 1) + 1; return a2 / (PI * d * d); }

float3 eval_brdf(float3 n, float3 v, float3 l, float3 base, float rough, float metal) {
    float nl = dot(n, l), nv = dot(n, v);
    if (nl <= 0 || nv <= 0) return 0;
    float3 h = normalize(v + l);
    float nh = saturate(dot(n, h)), vh = saturate(dot(v, h));
    float a = max(rough * rough, 2e-3), a2 = a * a;
    float k = a * 0.5;
    float G = (nl / (nl * (1 - k) + k)) * (nv / (nv * (1 - k) + k));
    float3 F = fresnel(lerp(float3(0.04), base, metal), vh);
    float3 spec = ggx_d(nh, a2) * G * F / (4 * nl * nv);
    float3 diff = (1 - F) * (1 - metal) * base / PI;
    return diff + spec;
}

float spec_prob(float metal) { return lerp(0.3, 1.0, metal); }

float pdf_brdf(float3 n, float3 v, float3 l, float rough, float metal) {
    float nl = dot(n, l);
    if (nl <= 0) return 0;
    float3 h = normalize(v + l);
    float a = max(rough * rough, 2e-3);
    float ps = spec_prob(metal);
    float pdf_s = ggx_d(saturate(dot(n, h)), a * a) * saturate(dot(n, h)) / (4 * max(dot(v, h), 1e-4));
    return ps * pdf_s + (1 - ps) * nl / PI;
}

void basis(float3 n, out float3 t, out float3 b) {
    float s = n.z >= 0 ? 1.0 : -1.0;
    float a = -1.0 / (s + n.z);
    float c = n.x * n.y * a;
    t = float3(1 + s * n.x * n.x * a, s * c, -s * n.x);
    b = float3(c, s + n.y * n.y * a, -n.y);
}

float3 sample_brdf(float3 n, float3 v, float rough, float metal, inout uint s) {
    float3 t, b; basis(n, t, b);
    float u1 = rnd(s), u2 = rnd(s);
    if (rnd(s) < spec_prob(metal)) {
        float a = max(rough * rough, 2e-3);
        float phi = 2 * PI * u1;
        float ct = sqrt((1 - u2) / (1 + (a * a - 1) * u2));
        float st = sqrt(1 - ct * ct);
        float3 h = t * (st * cos(phi)) + b * (st * sin(phi)) + n * ct;
        return reflect(-v, h);
    }
    float r = sqrt(u1), phi = 2 * PI * u2;
    return normalize(t * (r * cos(phi)) + b * (r * sin(phi)) + n * sqrt(max(0, 1 - u1)));
}

// ---------------------------------------------------------------- lights (next-event estimation)
float3 direct_light(float3 p, float3 n, float3 ng, float3 v, float3 base, float rough, float metal, inout uint s) {
    if (frame.light_count == 0) return 0;
    uint li = min(uint(rnd(s) * frame.light_count), frame.light_count - 1);
    Light L = lights[li];
    float3 o = p + ng * 1e-3;
    float3 contrib = 0;
    uint type = uint(L.pos_type.w);
    if (type == 2) {                                  // directional: radiance.rgb = irradiance
        float3 l = -L.u.xyz;
        float nl = dot(n, l);
        if (nl > 0 && visible(o, l, 1e30)) contrib = L.radiance.rgb * eval_brdf(n, v, l, base, rough, metal) * nl;
    } else {
        float3 lp = L.pos_type.xyz;
        float area_term = 1;
        if (type == 1) {                              // rectangle: radiance.rgb = emitted radiance
            lp += L.u.xyz * (2 * rnd(s) - 1) + L.v.xyz * (2 * rnd(s) - 1);
        }
        float3 dl = lp - o;
        float d2 = max(dot(dl, dl), 1e-6);
        float dist = sqrt(d2);
        float3 l = dl / dist;
        float nl = dot(n, l);
        if (nl > 0) {
            if (type == 1) {
                float3 ln = normalize(cross(L.u.xyz, L.v.xyz));
                float cl = dot(ln, -l);
                area_term = cl > 0 ? cl * 4 * length(L.u.xyz) * length(L.v.xyz) : 0;
            }
            if (area_term > 0 && visible(o, l, dist * 0.999))
                contrib = L.radiance.rgb * area_term / d2 * eval_brdf(n, v, l, base, rough, metal) * nl;
        }
    }
    return contrib * float(frame.light_count);
}

// ---------------------------------------------------------------- main
[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 id: SV_DispatchThreadID) {
    if (id.x >= frame.width || id.y >= frame.height) return;
    uint idx = id.y * frame.width + id.x;
    uint s = hash(idx ^ hash(frame.sample_index * 0x9e3779b9u + frame.seed * 0x85ebca6bu));

    float2 uv = (float2(id.xy) + float2(rnd(s), rnd(s))) / float2(frame.width, frame.height) * 2 - 1;
    float3 o = frame.cam_pos.xyz;
    float3 d = normalize(frame.cam_fwd.xyz + uv.x * frame.cam_right.xyz - uv.y * frame.cam_up.xyz);

    float3 radiance = 0, throughput = 1;
    for (uint bounce = 0; bounce <= frame.max_bounces; bounce++) {
        Hit h = trace(o, d);
        if (!h.hit) { radiance += throughput * frame.world.rgb; break; }
        Material m = materials[h.mat];
        float3 base = m.base_rough.rgb;
        float rough = clamp(m.base_rough.w, 0.02, 1.0);
        float metal = saturate(m.emit_metal.w);
        float3 v = -d;
        radiance += throughput * m.emit_metal.rgb;
        if (bounce == frame.max_bounces) break;

        float3 ld = direct_light(h.p, h.n, h.ng, v, base, rough, metal, s);
        radiance += throughput * min(ld, float3(64.0));   // clamp rare fireflies

        float3 l = sample_brdf(h.n, v, rough, metal, s);
        float pdf = pdf_brdf(h.n, v, l, rough, metal);
        if (pdf <= 1e-6 || dot(l, h.ng) <= 0) break;
        throughput *= eval_brdf(h.n, v, l, base, rough, metal) * dot(h.n, l) / pdf;
        if (bounce >= 2) {                            // Russian roulette
            float pr = clamp(max(throughput.r, max(throughput.g, throughput.b)), 0.05, 0.95);
            if (rnd(s) > pr) break;
            throughput /= pr;
        }
        o = h.p + h.ng * 1e-3;
        d = l;
    }
    if (any(isnan(radiance)) || any(isinf(radiance))) radiance = 0;
    float4 prev = frame.sample_index == 0 ? float4(0) : accum[idx];
    accum[idx] = prev + float4(radiance, 1);
}
