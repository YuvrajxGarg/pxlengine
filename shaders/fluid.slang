// FLIP/PIC liquid on a MAC grid + surface-nets meshing. Fixed library: scenes configure it,
// the renderer runs it. Units: metres, seconds, density 1.
//
// Per substep: clear -> p2g -> normalize/classify -> forces/boundaries -> divergence ->
//              pressure (red-black Gauss-Seidel) x N -> project -> g2p + advect
// Per frame:   s_clear -> s_splat -> s_phi -> s_verts -> s_quads  (mesh into the global
//              vertex/index buffers at vbase/ibase; counts in counters[0..1])

struct Particle { float4 x; float4 v; };

struct FluidPush {
    uint nx; uint ny; uint nz; uint n_particles;
    float h; float dt; float flip; float inv_h;
    float3 origin; uint parity;
    float3 gravity; uint pad0;
    uint sx; uint sy; uint sz; uint max_verts;
    float sh; float radius; float iso; float pad1;
    uint vbase; uint ibase; uint max_tris; uint pad2;
};
[[vk::push_constant]] ConstantBuffer<FluidPush> pc;

[[vk::binding(30, 0)]] RWStructuredBuffer<Particle> particles;
[[vk::binding(31, 0)]] RWStructuredBuffer<float> vel;        // u faces, then v, then w
[[vk::binding(32, 0)]] RWStructuredBuffer<float> vel_old;
[[vk::binding(33, 0)]] RWStructuredBuffer<int> acc;          // fixed-point sum of weight * velocity
[[vk::binding(34, 0)]] RWStructuredBuffer<int> wacc;         // fixed-point sum of weights
[[vk::binding(35, 0)]] RWStructuredBuffer<uint> cell_count;
[[vk::binding(36, 0)]] RWStructuredBuffer<uint> cell_type;   // 0 air, 1 fluid, 2 solid
[[vk::binding(37, 0)]] RWStructuredBuffer<float> pressure;
[[vk::binding(38, 0)]] RWStructuredBuffer<float> divergence;
[[vk::binding(39, 0)]] RWStructuredBuffer<int> density;      // surface nodes, fixed point
[[vk::binding(40, 0)]] RWStructuredBuffer<float> phi;        // surface nodes: density - iso (> 0 inside)
[[vk::binding(41, 0)]] RWStructuredBuffer<uint> cell_vert;   // surface cells -> vertex index
[[vk::binding(42, 0)]] RWStructuredBuffer<uint> counters;    // [0] vertices, [1] triangles
[[vk::binding(43, 0)]] RWStructuredBuffer<float4> out_verts; // renderer vertex buffer (2 per vertex)
[[vk::binding(44, 0)]] RWStructuredBuffer<uint> out_indices;

static const float SW = 65536.0;     // fixed-point scale for P2G
static const float SD = 4096.0;      // fixed-point scale for surface density
static const uint AIR = 0, FLUID = 1, SOLID = 2;
static const uint NONE = 0xffffffffu;

uint n_u() { return (pc.nx + 1) * pc.ny * pc.nz; }
uint n_v() { return pc.nx * (pc.ny + 1) * pc.nz; }
uint n_w() { return pc.nx * pc.ny * (pc.nz + 1); }
uint n_faces() { return n_u() + n_v() + n_w(); }
uint n_cells() { return pc.nx * pc.ny * pc.nz; }

// Face dims and storage offset per component c (0 = u, 1 = v, 2 = w).
uint3 face_dims(uint c) { uint3 d = uint3(pc.nx, pc.ny, pc.nz); d[c] += 1; return d; }
uint face_base(uint c) { return c == 0 ? 0 : (c == 1 ? n_u() : n_u() + n_v()); }
uint face_index(uint c, int3 f) { uint3 d = face_dims(c); return face_base(c) + uint(f.x) + d.x * (uint(f.y) + d.y * uint(f.z)); }
uint cell_index(int3 c) { return uint(c.x) + pc.nx * (uint(c.y) + pc.ny * uint(c.z)); }

uint cell_t(int3 c) {
    if (any(c < 0) || c.x >= int(pc.nx) || c.y >= int(pc.ny) || c.z >= int(pc.nz)) return SOLID;
    return cell_type[cell_index(c)];
}

// Grid coordinates of point x for component c (faces sit at the cell centre on the other axes).
float3 face_coord(uint c, float3 x) {
    float3 g = (x - pc.origin) * pc.inv_h - 0.5;
    g[c] += 0.5;
    return g;
}

float sample_face(RWStructuredBuffer<float> buf, uint c, float3 x) {
    float3 g = face_coord(c, x);
    int3 d = int3(face_dims(c)) - 1;
    int3 b = int3(floor(g));
    float3 f = g - float3(b);
    float s = 0;
    for (int k = 0; k < 8; k++) {
        int3 o = int3(k & 1, (k >> 1) & 1, k >> 2);
        int3 q = clamp(b + o, int3(0, 0, 0), d);
        float3 w3 = lerp(1 - f, f, float3(o));
        s += w3.x * w3.y * w3.z * buf[face_index(c, q)];
    }
    return s;
}

float3 sample_vel(RWStructuredBuffer<float> buf, float3 x) {
    return float3(sample_face(buf, 0, x), sample_face(buf, 1, x), sample_face(buf, 2, x));
}

// ---------------------------------------------------------------- simulation
[shader("compute")] [numthreads(256, 1, 1)]
void clear_grid(uint3 id: SV_DispatchThreadID) {
    if (id.x < n_faces()) { acc[id.x] = 0; wacc[id.x] = 0; }
    if (id.x < n_cells()) cell_count[id.x] = 0;
}

[shader("compute")] [numthreads(256, 1, 1)]
void p2g(uint3 id: SV_DispatchThreadID) {
    if (id.x >= pc.n_particles) return;
    Particle p = particles[id.x];
    for (uint c = 0; c < 3; c++) {
        float3 g = face_coord(c, p.x.xyz);
        int3 d = int3(face_dims(c)) - 1;
        int3 b = int3(floor(g));
        float3 f = g - float3(b);
        for (int k = 0; k < 8; k++) {
            int3 o = int3(k & 1, (k >> 1) & 1, k >> 2);
            int3 q = b + o;
            if (any(q < 0) || any(q > d)) continue;
            float3 w3 = lerp(1 - f, f, float3(o));
            float w = w3.x * w3.y * w3.z;
            uint fi = face_index(c, q);
            InterlockedAdd(acc[fi], int(w * p.v[c] * SW));
            InterlockedAdd(wacc[fi], int(w * SW));
        }
    }
    int3 cell = clamp(int3(floor((p.x.xyz - pc.origin) * pc.inv_h)), int3(0, 0, 0), int3(pc.nx, pc.ny, pc.nz) - 1);
    InterlockedAdd(cell_count[cell_index(cell)], 1u);
}

[shader("compute")] [numthreads(256, 1, 1)]
void normalize_grid(uint3 id: SV_DispatchThreadID) {
    if (id.x < n_faces()) {
        int w = wacc[id.x];
        float v = w > 0 ? float(acc[id.x]) / float(w) : 0;
        vel[id.x] = v;
        vel_old[id.x] = v;
    }
    if (id.x < n_cells()) {
        uint i = id.x % pc.nx, j = (id.x / pc.nx) % pc.ny, k = id.x / (pc.nx * pc.ny);
        bool wall = i == 0 || j == 0 || k == 0 || i == pc.nx - 1 || j == pc.ny - 1 || k == pc.nz - 1;
        uint t = wall ? SOLID : (cell_count[id.x] > 0 ? FLUID : AIR);
        cell_type[id.x] = t;
        if (t != FLUID) pressure[id.x] = 0;
    }
}

// Decode face id -> (component, face coords).
void face_of(uint id, out uint c, out int3 f) {
    c = id < n_u() ? 0 : (id < n_u() + n_v() ? 1 : 2);
    uint3 d = face_dims(c);
    uint r = id - face_base(c);
    f = int3(r % d.x, (r / d.x) % d.y, r / (d.x * d.y));
}

[shader("compute")] [numthreads(256, 1, 1)]
void forces(uint3 id: SV_DispatchThreadID) {
    if (id.x >= n_faces()) return;
    uint c; int3 f;
    face_of(id.x, c, f);
    int3 lo = f; lo[c] -= 1;
    float v = vel[id.x] + pc.gravity[c] * pc.dt;
    if (cell_t(lo) == SOLID || cell_t(f) == SOLID) v = 0;
    vel[id.x] = v;
}

[shader("compute")] [numthreads(256, 1, 1)]
void compute_divergence(uint3 id: SV_DispatchThreadID) {
    if (id.x >= n_cells()) return;
    int3 c = int3(id.x % pc.nx, (id.x / pc.nx) % pc.ny, id.x / (pc.nx * pc.ny));
    float d = 0;
    if (cell_type[id.x] == FLUID) {
        for (uint a = 0; a < 3; a++) {
            int3 hi = c; hi[a] += 1;
            d += vel[face_index(a, hi)] - vel[face_index(a, c)];
        }
    }
    divergence[id.x] = d * pc.inv_h;
}

[shader("compute")] [numthreads(256, 1, 1)]
void pressure_rb(uint3 id: SV_DispatchThreadID) {
    if (id.x >= n_cells() || cell_type[id.x] != FLUID) return;
    int3 c = int3(id.x % pc.nx, (id.x / pc.nx) % pc.ny, id.x / (pc.nx * pc.ny));
    if (uint(c.x + c.y + c.z) % 2 != pc.parity) return;
    float sum = 0, cnt = 0;
    for (int n = 0; n < 6; n++) {
        int3 q = c;
        q[n >> 1] += ((n & 1) != 0) ? 1 : -1;
        uint t = cell_t(q);
        if (t == SOLID) continue;
        cnt += 1;
        if (t == FLUID) sum += pressure[cell_index(q)];
    }
    if (cnt > 0) pressure[id.x] = (sum - divergence[id.x] * pc.h * pc.h / pc.dt) / cnt;
}

[shader("compute")] [numthreads(256, 1, 1)]
void project(uint3 id: SV_DispatchThreadID) {
    if (id.x >= n_faces()) return;
    uint c; int3 f;
    face_of(id.x, c, f);
    int3 lo = f; lo[c] -= 1;
    uint ta = cell_t(lo), tb = cell_t(f);
    if (ta == SOLID || tb == SOLID) { vel[id.x] = 0; return; }
    if (ta == FLUID || tb == FLUID) {
        float pa = ta == FLUID ? pressure[cell_index(lo)] : 0;
        float pb = tb == FLUID ? pressure[cell_index(f)] : 0;
        vel[id.x] -= pc.dt * pc.inv_h * (pb - pa);
    }
}

[shader("compute")] [numthreads(256, 1, 1)]
void g2p_advect(uint3 id: SV_DispatchThreadID) {
    if (id.x >= pc.n_particles) return;
    Particle p = particles[id.x];
    float3 x = p.x.xyz;
    float3 pic = sample_vel(vel, x);
    float3 flip = p.v.xyz + pic - sample_vel(vel_old, x);
    float3 v = lerp(pic, flip, pc.flip);
    // RK2 through the divergence-free grid velocity.
    float3 mid = x + 0.5 * pc.dt * pic;
    x += pc.dt * sample_vel(vel, mid);
    // Stay inside the non-solid interior; kill velocity into walls.
    float3 lo = pc.origin + pc.h * 1.001;
    float3 hi = pc.origin + pc.h * (float3(pc.nx, pc.ny, pc.nz) - 1.001);
    for (uint a = 0; a < 3; a++) {
        if (x[a] < lo[a]) { x[a] = lo[a]; v[a] = max(v[a], 0); }
        if (x[a] > hi[a]) { x[a] = hi[a]; v[a] = min(v[a], 0); }
    }
    p.x.xyz = x;
    p.v.xyz = v;
    particles[id.x] = p;
}

// ---------------------------------------------------------------- surface (surface nets)
uint n_snodes() { return (pc.sx + 1) * (pc.sy + 1) * (pc.sz + 1); }
uint n_scells() { return pc.sx * pc.sy * pc.sz; }
uint snode(int3 n) { return uint(n.x) + (pc.sx + 1) * (uint(n.y) + (pc.sy + 1) * uint(n.z)); }
uint scell(int3 c) { return uint(c.x) + pc.sx * (uint(c.y) + pc.sy * uint(c.z)); }

[shader("compute")] [numthreads(256, 1, 1)]
void s_clear(uint3 id: SV_DispatchThreadID) {
    if (id.x < n_snodes()) density[id.x] = 0;
    if (id.x < n_scells()) cell_vert[id.x] = NONE;
    if (id.x < 2) counters[id.x] = 0;
}

[shader("compute")] [numthreads(256, 1, 1)]
void s_splat(uint3 id: SV_DispatchThreadID) {
    if (id.x >= pc.n_particles) return;
    float3 g = (particles[id.x].x.xyz - pc.origin) / pc.sh;
    float r = pc.radius / pc.sh;
    int3 lo = max(int3(ceil(g - r)), int3(0, 0, 0));
    int3 hi = min(int3(floor(g + r)), int3(pc.sx, pc.sy, pc.sz));
    for (int k = lo.z; k <= hi.z; k++)
    for (int j = lo.y; j <= hi.y; j++)
    for (int i = lo.x; i <= hi.x; i++) {
        float3 d = float3(i, j, k) - g;
        float q = 1 - dot(d, d) / (r * r);
        if (q > 0) InterlockedAdd(density[snode(int3(i, j, k))], int(q * q * q * SD));
    }
}

[shader("compute")] [numthreads(256, 1, 1)]
void s_phi(uint3 id: SV_DispatchThreadID) {
    if (id.x >= n_snodes()) return;
    int3 n = int3(id.x % (pc.sx + 1), (id.x / (pc.sx + 1)) % (pc.sy + 1), id.x / ((pc.sx + 1) * (pc.sy + 1)));
    // Light 7-point smoothing, then the iso offset; boundary nodes are forced outside so the
    // mesh closes against the container walls.
    float s = 2 * float(density[id.x]);
    float wsum = 2;
    for (int a = 0; a < 6; a++) {
        int3 q = n;
        q[a >> 1] += ((a & 1) != 0) ? 1 : -1;
        if (any(q < 0) || q.x > int(pc.sx) || q.y > int(pc.sy) || q.z > int(pc.sz)) continue;
        s += float(density[snode(q)]);
        wsum += 1;
    }
    bool edge = any(n == 0) || n.x == int(pc.sx) || n.y == int(pc.sy) || n.z == int(pc.sz);
    phi[id.x] = edge ? -1 : s / wsum / SD - pc.iso;
}

[shader("compute")] [numthreads(256, 1, 1)]
void s_verts(uint3 id: SV_DispatchThreadID) {
    if (id.x >= n_scells()) return;
    int3 c = int3(id.x % pc.sx, (id.x / pc.sx) % pc.sy, id.x / (pc.sx * pc.sy));
    float v[8];
    uint inside = 0;
    for (int k = 0; k < 8; k++) {
        v[k] = phi[snode(c + int3(k & 1, (k >> 1) & 1, k >> 2))];
        if (v[k] > 0) inside |= 1u << k;
    }
    if (inside == 0 || inside == 255) return;
    // Vertex = average of the edge crossings.
    float3 sum = 0;
    float cnt = 0;
    for (int e = 0; e < 12; e++) {
        int axis = e >> 2;
        int bits = e & 3;                       // the other two axes' corner bits
        int a0 = 0;
        int o1 = (axis + 1) % 3, o2 = (axis + 2) % 3;
        if ((bits & 1) != 0) a0 |= 1 << o1;
        if ((bits & 2) != 0) a0 |= 1 << o2;
        int a1 = a0 | (1 << axis);
        if ((v[a0] > 0) == (v[a1] > 0)) continue;
        float t = v[a0] / (v[a0] - v[a1]);
        float3 p0 = float3(a0 & 1, (a0 >> 1) & 1, a0 >> 2);
        float3 p1 = float3(a1 & 1, (a1 >> 1) & 1, a1 >> 2);
        sum += lerp(p0, p1, t);
        cnt += 1;
    }
    float3 local = sum / cnt;
    // Gradient of phi (trilinear) at the vertex -> outward normal = -grad.
    float3 grad = 0;
    for (int k = 0; k < 8; k++) {
        float3 o = float3(k & 1, (k >> 1) & 1, k >> 2);
        float3 w = lerp(1 - local, local, o);
        float3 s = o * 2 - 1;
        grad += v[k] * float3(s.x * w.y * w.z, s.y * w.x * w.z, s.z * w.x * w.y);
    }
    uint idx;
    InterlockedAdd(counters[0], 1u, idx);
    if (idx >= pc.max_verts) return;
    float3 pos = pc.origin + (float3(c) + local) * pc.sh;
    float3 n = -normalize(grad + float3(0, 1e-9, 0));
    out_verts[(pc.vbase + idx) * 2] = float4(pos, 0);
    out_verts[(pc.vbase + idx) * 2 + 1] = float4(n, 0);
    cell_vert[id.x] = idx;
}

[shader("compute")] [numthreads(256, 1, 1)]
void s_quads(uint3 id: SV_DispatchThreadID) {
    if (id.x >= n_snodes()) return;
    int3 n = int3(id.x % (pc.sx + 1), (id.x / (pc.sx + 1)) % (pc.sy + 1), id.x / ((pc.sx + 1) * (pc.sy + 1)));
    bool in0 = phi[id.x] > 0;
    for (int axis = 0; axis < 3; axis++) {
        int3 m = n;
        m[axis] += 1;
        if (m.x > int(pc.sx) || m.y > int(pc.sy) || m.z > int(pc.sz)) continue;
        if (in0 == (phi[snode(m)] > 0)) continue;
        int o1 = (axis + 1) % 3, o2 = (axis + 2) % 3;
        uint q[4];
        bool ok = true;
        // The four cells around this edge, in order around the axis.
        int2 offs[4] = { int2(-1, -1), int2(0, -1), int2(0, 0), int2(-1, 0) };
        for (int k = 0; k < 4; k++) {
            int3 c = n;
            c[o1] += offs[k].x;
            c[o2] += offs[k].y;
            if (any(c < 0) || c.x >= int(pc.sx) || c.y >= int(pc.sy) || c.z >= int(pc.sz)) { ok = false; break; }
            q[k] = cell_vert[scell(c)];
            if (q[k] == NONE) { ok = false; break; }
        }
        if (!ok) continue;
        uint t;
        InterlockedAdd(counters[1], 2u, t);
        if (t + 2 > pc.max_tris) continue;
        uint b = pc.ibase + t * 3;
        // Wind so the face normal points out of the liquid (from inside to outside along axis).
        if (in0) {
            out_indices[b] = q[0]; out_indices[b + 1] = q[1]; out_indices[b + 2] = q[2];
            out_indices[b + 3] = q[0]; out_indices[b + 4] = q[2]; out_indices[b + 5] = q[3];
        } else {
            out_indices[b] = q[0]; out_indices[b + 1] = q[2]; out_indices[b + 2] = q[1];
            out_indices[b + 3] = q[0]; out_indices[b + 4] = q[3]; out_indices[b + 5] = q[2];
        }
    }
}
