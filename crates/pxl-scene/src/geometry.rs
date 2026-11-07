//! Built-in ("std") geometry. Meshes are indexed triangle lists in object space.

use crate::Geometry;
use std::f32::consts::{PI, TAU};

#[derive(Debug, Clone, Default)]
pub struct MeshData {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub uvs: Vec<[f32; 2]>,
    pub indices: Vec<u32>,
}

impl MeshData {
    fn push(&mut self, p: [f32; 3], n: [f32; 3], uv: [f32; 2]) -> u32 {
        self.positions.push(p);
        self.normals.push(n);
        self.uvs.push(uv);
        (self.positions.len() - 1) as u32
    }

    /// Indices for a (rows+1) x (cols+1) vertex lattice starting at `base`.
    fn lattice(&mut self, base: u32, rows: u32, cols: u32) {
        let stride = cols + 1;
        for i in 0..rows {
            for j in 0..cols {
                let a = base + i * stride + j;
                let b = a + stride;
                self.indices.extend_from_slice(&[a, b, a + 1, a + 1, b, b + 1]);
            }
        }
    }

    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }

    /// Area-weighted vertex normals (for meshes that ship without normals).
    pub fn compute_normals(&mut self) {
        let mut n = vec![[0.0f32; 3]; self.positions.len()];
        for t in self.indices.chunks_exact(3) {
            let [a, b, c] = [t[0], t[1], t[2]].map(|i| glam::Vec3::from(self.positions[i as usize]));
            let f = (b - a).cross(c - a);
            for &i in t {
                let v = &mut n[i as usize];
                *v = (glam::Vec3::from(*v) + f).to_array();
            }
        }
        self.normals = n.into_iter().map(|v| glam::Vec3::from(v).normalize_or(glam::Vec3::Y).to_array()).collect();
    }
}

pub fn build(g: &Geometry) -> MeshData {
    match *g {
        Geometry::Sphere { radius, segments } => sphere(radius, segments.clamp(3, 1024)),
        Geometry::Grid { size } => grid(size),
        Geometry::Box { size } => cuboid(size),
        Geometry::Cylinder { radius, height, segments } => cylinder(radius, height, segments.clamp(3, 1024)),
        Geometry::Torus { radius, tube, segments } => torus(radius, tube, segments.clamp(3, 1024)),
        Geometry::Bird {} => bird(),
    }
}

/// Low-poly bird: diamond body, swept wings, tail fan. Unshared vertices -> flat shading.
fn bird() -> MeshData {
    let nose = [0.0, 0.0, 0.42];
    let tail = [0.0, 0.0, -0.3];
    let top = [0.0, 0.055, 0.04];
    let bottom = [0.0, -0.045, 0.06];
    let l = [-0.055, 0.0, 0.04];
    let r = [0.055, 0.0, 0.04];
    let mut tris: Vec<[[f32; 3]; 3]> = vec![
        [nose, l, top], [nose, top, r], [nose, r, bottom], [nose, bottom, l],
        [tail, top, l], [tail, r, top], [tail, bottom, r], [tail, l, bottom],
    ];
    for s in [-1.0f32, 1.0] {
        // wing: root at the shoulder, swept tip, slight dihedral
        let root_f = [0.05 * s, 0.01, 0.14];
        let root_b = [0.05 * s, 0.01, -0.08];
        let mid = [0.3 * s, 0.05, 0.02];
        let tip = [0.5 * s, 0.07, -0.16];
        tris.push([root_f, mid, root_b]);
        tris.push([mid, tip, root_b]);
        // tail feathers
        tris.push([[0.0, 0.0, -0.22], [0.11 * s, 0.0, -0.42], [0.0, 0.0, -0.4]]);
    }
    let mut m = MeshData::default();
    for t in tris {
        let base = m.positions.len() as u32;
        for p in t {
            m.push(p, [0.0, 1.0, 0.0], [p[0] + 0.5, p[2] + 0.5]);
        }
        m.indices.extend_from_slice(&[base, base + 1, base + 2]);
    }
    m.compute_normals();
    m
}

fn sphere(r: f32, seg: u32) -> MeshData {
    let rings = (seg / 2).max(2);
    let mut m = MeshData::default();
    for i in 0..=rings {
        let th = PI * i as f32 / rings as f32;
        for j in 0..=seg {
            let ph = TAU * j as f32 / seg as f32;
            let n = [th.sin() * ph.cos(), th.cos(), th.sin() * ph.sin()];
            m.push(n.map(|c| c * r), n, [j as f32 / seg as f32, i as f32 / rings as f32]);
        }
    }
    m.lattice(0, rings, seg);
    m
}

/// Flat grid on the XZ plane, facing +Y.
fn grid([w, d]: [f32; 2]) -> MeshData {
    let mut m = MeshData::default();
    let (hw, hd) = (w * 0.5, d * 0.5);
    for (x, z, u, v) in [(-hw, -hd, 0.0, 0.0), (hw, -hd, 1.0, 0.0), (-hw, hd, 0.0, 1.0), (hw, hd, 1.0, 1.0)] {
        m.push([x, 0.0, z], [0.0, 1.0, 0.0], [u, v]);
    }
    m.indices.extend_from_slice(&[0, 2, 1, 1, 2, 3]);
    m
}

fn cuboid([sx, sy, sz]: [f32; 3]) -> MeshData {
    let h = [sx * 0.5, sy * 0.5, sz * 0.5];
    let mut m = MeshData::default();
    // (normal axis, sign)
    for axis in 0..3 {
        for sign in [-1.0f32, 1.0] {
            let (u, v) = ((axis + 1) % 3, (axis + 2) % 3);
            let mut n = [0.0; 3];
            n[axis] = sign;
            let base = m.positions.len() as u32;
            for (a, b) in [(-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0)] {
                let mut p = [0.0; 3];
                p[axis] = sign * h[axis];
                p[u] = a * h[u];
                p[v] = b * h[v];
                m.push(p, n, [(a + 1.0) * 0.5, (b + 1.0) * 0.5]);
            }
            m.lattice(base, 1, 1);
        }
    }
    m
}

fn cylinder(r: f32, height: f32, seg: u32) -> MeshData {
    let mut m = MeshData::default();
    let hy = height * 0.5;
    for i in 0..=1 {
        let y = if i == 0 { -hy } else { hy };
        for j in 0..=seg {
            let a = TAU * j as f32 / seg as f32;
            m.push([r * a.cos(), y, r * a.sin()], [a.cos(), 0.0, a.sin()], [j as f32 / seg as f32, i as f32]);
        }
    }
    m.lattice(0, 1, seg);
    for (y, ny) in [(-hy, -1.0f32), (hy, 1.0)] {
        let c = m.push([0.0, y, 0.0], [0.0, ny, 0.0], [0.5, 0.5]);
        let base = m.positions.len() as u32;
        for j in 0..=seg {
            let a = TAU * j as f32 / seg as f32;
            m.push([r * a.cos(), y, r * a.sin()], [0.0, ny, 0.0], [0.5 + 0.5 * a.cos(), 0.5 + 0.5 * a.sin()]);
        }
        for j in 0..seg {
            m.indices.extend_from_slice(&[c, base + j, base + j + 1]);
        }
    }
    m
}

fn torus(r: f32, tube: f32, seg: u32) -> MeshData {
    let mut m = MeshData::default();
    let rings = (seg / 2).max(3);
    for i in 0..=seg {
        let u = TAU * i as f32 / seg as f32;
        for j in 0..=rings {
            let v = TAU * j as f32 / rings as f32;
            let n = [v.cos() * u.cos(), v.sin(), v.cos() * u.sin()];
            let p = [(r + tube * v.cos()) * u.cos(), tube * v.sin(), (r + tube * v.cos()) * u.sin()];
            m.push(p, n, [i as f32 / seg as f32, j as f32 / rings as f32]);
        }
    }
    m.lattice(0, seg, rings);
    m
}
