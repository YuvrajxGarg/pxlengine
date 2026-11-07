// Simulation infrastructure kernels (fixed; scene programs supply init/step).
//
// Neighbour grid per step (counting sort by hashed cell):
//   grid_clear -> grid_count -> scan_local -> scan_blocks -> scan_add -> grid_scatter
// leaves `offsets[h]` (first slot), `counts[h]` and `sorted[]` (agent ids grouped by cell).
// `emit` turns agents into TLAS instance records + renderer instance records.

struct Agent { float3 position; float size; float3 velocity; float age; float4 color; float4 data; };

struct SimPush {
    uint count; uint table_size; float cell_size; uint n_blocks;
    uint base_instance; uint material; uint vertex_offset; uint index_offset;
    uint blas_lo; uint blas_hi; float scale; float time_offset;
    uint object_id; uint orient; uint pad0; uint pad1;
};
[[vk::push_constant]] ConstantBuffer<SimPush> pc;

[[vk::binding(20, 0)]] StructuredBuffer<Agent> agents;
[[vk::binding(22, 0)]] RWStructuredBuffer<uint> offsets;
[[vk::binding(23, 0)]] RWStructuredBuffer<uint> counts;
[[vk::binding(24, 0)]] RWStructuredBuffer<uint> sorted;
[[vk::binding(25, 0)]] RWStructuredBuffer<uint> cursor;
[[vk::binding(26, 0)]] RWStructuredBuffer<uint> block_sums;
[[vk::binding(27, 0)]] RWStructuredBuffer<uint4> tlas_desc;   // 4 per instance (VkAccelerationStructureInstanceKHR)
[[vk::binding(28, 0)]] RWStructuredBuffer<uint4> inst_rec;    // 4 per instance (renderer Instance)

uint cell_hash(int3 c) {
    return ((uint(c.x) * 73856093u) ^ (uint(c.y) * 19349663u) ^ (uint(c.z) * 83492791u)) & (pc.table_size - 1);
}
uint agent_cell(uint id) { return cell_hash(int3(floor(agents[id].position / pc.cell_size))); }

[shader("compute")] [numthreads(256, 1, 1)]
void grid_clear(uint3 id: SV_DispatchThreadID) {
    if (id.x < pc.table_size) counts[id.x] = 0;
}

[shader("compute")] [numthreads(256, 1, 1)]
void grid_count(uint3 id: SV_DispatchThreadID) {
    if (id.x < pc.count) InterlockedAdd(counts[agent_cell(id.x)], 1u);
}

groupshared uint sh[256];

// Inclusive Hillis-Steele scan of sh[0..256) by 256 threads.
void scan_shared(uint t) {
    for (uint off = 1; off < 256; off <<= 1) {
        uint x = t >= off ? sh[t - off] : 0;
        GroupMemoryBarrierWithGroupSync();
        sh[t] += x;
        GroupMemoryBarrierWithGroupSync();
    }
}

// Each group: 1024 counts (4 per thread) -> local exclusive offsets + block total.
[shader("compute")] [numthreads(256, 1, 1)]
void scan_local(uint3 gid: SV_GroupID, uint3 tid: SV_GroupThreadID) {
    uint t = tid.x;
    uint base = gid.x * 1024 + t * 4;
    uint v[4];
    uint s = 0;
    for (uint k = 0; k < 4; k++) { v[k] = base + k < pc.table_size ? counts[base + k] : 0; s += v[k]; }
    sh[t] = s;
    GroupMemoryBarrierWithGroupSync();
    scan_shared(t);
    uint run = sh[t] - s;
    for (uint k = 0; k < 4; k++) {
        if (base + k < pc.table_size) offsets[base + k] = run;
        run += v[k];
    }
    if (t == 255) block_sums[gid.x] = sh[255];
}

// One group: exclusive scan of the block totals (any number of blocks).
[shader("compute")] [numthreads(256, 1, 1)]
void scan_blocks(uint3 tid: SV_GroupThreadID) {
    uint t = tid.x;
    uint per = (pc.n_blocks + 255) / 256;
    uint lo = t * per, hi = min(lo + per, pc.n_blocks);
    uint s = 0;
    for (uint i = lo; i < hi; i++) s += block_sums[i];
    sh[t] = s;
    GroupMemoryBarrierWithGroupSync();
    scan_shared(t);
    uint run = sh[t] - s;
    for (uint i = lo; i < hi; i++) { uint x = block_sums[i]; block_sums[i] = run; run += x; }
}

[shader("compute")] [numthreads(256, 1, 1)]
void scan_add(uint3 gid: SV_GroupID, uint3 tid: SV_GroupThreadID) {
    uint base = gid.x * 1024 + tid.x * 4;
    uint add = block_sums[gid.x];
    for (uint k = 0; k < 4; k++) {
        uint i = base + k;
        if (i < pc.table_size) { offsets[i] += add; cursor[i] = offsets[i]; }
    }
}

[shader("compute")] [numthreads(256, 1, 1)]
void grid_scatter(uint3 id: SV_DispatchThreadID) {
    if (id.x >= pc.count) return;
    uint slot;
    InterlockedAdd(cursor[agent_cell(id.x)], 1u, slot);
    sorted[slot] = id.x;
}

uint pack_rgba8(float4 c) {
    uint4 q = uint4(saturate(c) * 255.0 + 0.5);
    return q.x | (q.y << 8) | (q.z << 16) | (q.w << 24);
}

[shader("compute")] [numthreads(256, 1, 1)]
void emit(uint3 id: SV_DispatchThreadID) {
    if (id.x >= pc.count) return;
    Agent a = agents[id.x];
    float3 pos = a.position + a.velocity * pc.time_offset;
    float s = max(a.size, 0) * pc.scale;
    float3 f = float3(0, 0, 1);
    if (pc.orient != 0 && dot(a.velocity, a.velocity) > 1e-12) f = normalize(a.velocity);
    float3 up0 = abs(f.y) > 0.99 ? float3(1, 0, 0) : float3(0, 1, 0);
    float3 r = normalize(cross(up0, f));
    float3 u = cross(f, r);
    uint i = pc.base_instance + id.x;
    // Object -> world: columns r, u, f (x right, y up, z forward), uniform scale, translation.
    tlas_desc[i * 4 + 0] = asuint(float4(r.x * s, u.x * s, f.x * s, pos.x));
    tlas_desc[i * 4 + 1] = asuint(float4(r.y * s, u.y * s, f.y * s, pos.y));
    tlas_desc[i * 4 + 2] = asuint(float4(r.z * s, u.z * s, f.z * s, pos.z));
    // custom index = instance index, mask 0xff; flags = TRIANGLE_FACING_CULL_DISABLE; BLAS address
    tlas_desc[i * 4 + 3] = uint4(i | (0xffu << 24), 1u << 24, pc.blas_lo, pc.blas_hi);
    // Renderer instance: meta (vertex offset, index offset, material, no emissive), then the
    // normal matrix rows (rotation; scale is uniform), object id in n0.w, tint in n1.w.
    uint tint = a.color.a > 0 ? pack_rgba8(a.color) : 0u;
    inst_rec[i * 4 + 0] = uint4(pc.vertex_offset, pc.index_offset, pc.material, 0xffffffffu);
    inst_rec[i * 4 + 1] = uint4(asuint(r.x), asuint(u.x), asuint(f.x), pc.object_id);
    inst_rec[i * 4 + 2] = uint4(asuint(r.y), asuint(u.y), asuint(f.y), tint);
    inst_rec[i * 4 + 3] = uint4(asuint(r.z), asuint(u.z), asuint(f.z), 0u);
}
