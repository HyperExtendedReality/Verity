//! Vertex-coloured triangle meshes of what a city builds: roads, kerbs,
//! sidewalks, junctions, rivers and buildings.

use glam::{Vec2, Vec3};
use verity_nav::Path;

use crate::{
    network::{Network, KERB},
    Kind,
};

const ASPHALT: [f32; 3] = [0.06, 0.06, 0.065];
const HIGHWAY: [f32; 3] = [0.08, 0.08, 0.085];
const PAVEMENT: [f32; 3] = [0.42, 0.41, 0.39];
const CONCRETE: [f32; 3] = [0.55, 0.55, 0.53];
const DIVIDER: [f32; 3] = [0.75, 0.6, 0.1];
const PAINT: [f32; 3] = [0.8, 0.8, 0.78];
const WATER: [f32; 3] = [0.03, 0.14, 0.2];
const BANK: [f32; 3] = [0.16, 0.13, 0.09];

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Mesh {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub colors: Vec<[f32; 3]>,
    /// Counter-clockwise seen from the front.
    pub indices: Vec<u32>,
}

impl Mesh {
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }

    fn vertex(&mut self, p: Vec3, normal: Vec3, color: [f32; 3]) -> u32 {
        self.positions.push(p.to_array());
        self.normals.push(normal.to_array());
        self.colors.push(color);
        self.positions.len() as u32 - 1
    }

    /// A flat quad with corners counter-clockwise seen from `normal`'s side.
    pub fn quad(&mut self, corners: [Vec3; 4], normal: Vec3, color: [f32; 3]) {
        let v = corners.map(|c| self.vertex(c, normal, color));
        self.indices.extend([v[0], v[1], v[2], v[0], v[2], v[3]]);
    }

    /// A flat rectangle around `centre` with half-extent vectors `a` and `b`,
    /// facing `facing`'s side.
    pub fn panel(&mut self, centre: Vec3, a: Vec3, b: Vec3, facing: Vec3, color: [f32; 3]) {
        let (a, b) = if a.cross(b).dot(facing) >= 0.0 { (a, b) } else { (b, a) };
        let c = centre;
        self.quad([c - a - b, c + a - b, c + a + b, c - a + b], facing.normalize_or_zero(), color);
    }

    /// A box around `centre` with half-extent vectors `x`, `y`, `z`; its top
    /// face gets `top`.
    pub fn cuboid(&mut self, centre: Vec3, [x, y, z]: [Vec3; 3], color: [f32; 3], top: [f32; 3]) {
        for (n, u, v) in [(x, y, z), (-x, z, y), (y, z, x), (-y, x, z), (z, x, y), (-z, y, x)] {
            let normal = n.normalize_or_zero();
            // Keep each face counter-clockwise from outside whatever the axes' handedness.
            let (u, v) = if u.cross(v).dot(n) >= 0.0 { (u, v) } else { (v, u) };
            let c = centre + n;
            let shade = if normal.z > 0.5 { top } else { color };
            self.quad([c - u - v, c + u - v, c + u + v, c - u + v], normal, shade);
        }
    }

    /// An upward surface between two polylines, `left` on the left walking along them.
    pub fn strip(&mut self, left: &[Vec3], right: &[Vec3], color: [f32; 3]) {
        let n = left.len().min(right.len());
        for i in 1..n {
            self.quad([right[i - 1], right[i], left[i], left[i - 1]], Vec3::Z, color);
        }
    }

    /// A vertical wall under a polyline, from its points down by `height`,
    /// facing `face`'s side (left when positive).
    pub fn curtain(&mut self, top: &[Vec3], height: f32, left: bool, color: [f32; 3]) {
        for i in 1..top.len() {
            let (a, b) = (top[i - 1], top[i]);
            let side = crate::spline::left(b - a) * if left { 1.0 } else { -1.0 };
            let down = Vec3::Z * height;
            let corners = if left { [b - down, a - down, a, b] } else { [a - down, b - down, b, a] };
            self.quad(corners, side, color);
        }
    }

    pub fn append(&mut self, other: &Mesh) {
        let base = self.positions.len() as u32;
        self.positions.extend_from_slice(&other.positions);
        self.normals.extend_from_slice(&other.normals);
        self.colors.extend_from_slice(&other.colors);
        self.indices.extend(other.indices.iter().map(|i| i + base));
    }

    /// Minimum and maximum corners.
    pub fn bounds(&self) -> (Vec3, Vec3) {
        (self.positions.iter()).fold((Vec3::MAX, Vec3::MIN), |(lo, hi), p| {
            (lo.min(Vec3::from_array(*p)), hi.max(Vec3::from_array(*p)))
        })
    }
}

fn lift(path: &Path, z: f32) -> Vec<Vec3> {
    path.points.iter().map(|p| *p + Vec3::Z * z).collect()
}

/// Short painted dashes along a path.
fn dashes(mesh: &mut Mesh, path: &Path, width: f32, dash: f32, gap: f32, color: [f32; 3]) {
    let mut s = gap * 0.5;
    while s + dash <= path.length() {
        let pieces: Vec<(Vec3, Vec3)> = (0..=2).map(|k| path.at(s + dash * k as f32 / 2.0)).collect();
        let left: Vec<Vec3> = pieces.iter().map(|(p, d)| *p + crate::spline::left(*d) * width * 0.5).collect();
        let right: Vec<Vec3> = pieces.iter().map(|(p, d)| *p - crate::spline::left(*d) * width * 0.5).collect();
        mesh.strip(&left, &right, color);
        s += dash + gap;
    }
}

/// The surface of edge `e`: carriageway, markings, kerbs and sidewalks, or water.
pub fn edge(network: &Network, e: usize) -> Mesh {
    let mut mesh = Mesh::default();
    let edge = &network.edges[e];
    let (w, centre) = (edge.width * 0.5, &edge.centre);
    match edge.kind {
        Kind::River => {
            let bank = 3.0;
            mesh.strip(&lift(&centre.offset(w), -0.2), &lift(&centre.offset(-w), -0.2), WATER);
            mesh.strip(&lift(&centre.offset(w + bank), 0.05), &lift(&centre.offset(w), -0.2), BANK);
            mesh.strip(&lift(&centre.offset(-w), -0.2), &lift(&centre.offset(-w - bank), 0.05), BANK);
        }
        Kind::Road | Kind::Highway => {
            let surface = if edge.kind == Kind::Highway { HIGHWAY } else { ASPHALT };
            mesh.strip(&lift(&centre.offset(w), 0.03), &lift(&centre.offset(-w), 0.03), surface);
            let [ta, tb] = network.trims(e);
            let [nf, nb] = edge.lanes.map(f32::from);
            let lane = edge.width / (nf + nb).max(1.0);
            let line = |offset: f32| lift(&centre.offset(offset).trimmed(ta, tb), 0.04);
            if nf > 0.0 && nb > 0.0 {
                let divide = Path::new(line(-w + nf * lane));
                let (l, r) = (divide.offset(0.12), divide.offset(-0.12));
                mesh.strip(&l.points, &r.points, DIVIDER);
            }
            for k in (1..(nf + nb) as u32).filter(|&k| k as f32 != nf) {
                dashes(&mut mesh, &Path::new(line(-w + k as f32 * lane)), 0.15, 3.0, 6.0, PAINT);
            }
            if edge.kind == Kind::Road {
                for sign in [1.0, -1.0] {
                    let inner = lift(&centre.offset(w * sign).trimmed(ta, tb), 0.03 + KERB);
                    let outer = lift(&centre.offset((w + network.sidewalk) * sign).trimmed(ta, tb), 0.03 + KERB);
                    let (left, right) = if sign > 0.0 { (&outer, &inner) } else { (&inner, &outer) };
                    mesh.strip(left, right, PAVEMENT);
                    // The kerb faces the road.
                    mesh.curtain(&inner, KERB, sign < 0.0, CONCRETE);
                }
            } else {
                // Barriers along both edges.
                for sign in [1.0, -1.0] {
                    let top = lift(&centre.offset((w - 0.3) * sign).trimmed(ta, tb), 0.9);
                    mesh.curtain(&top, 0.9, sign < 0.0, CONCRETE);
                    mesh.curtain(&top, 0.9, sign > 0.0, CONCRETE);
                }
            }
        }
    }
    mesh
}

/// A junction's paving: a disc where lanes stop short of it.
pub fn junction(network: &Network, n: usize) -> Mesh {
    let mut mesh = Mesh::default();
    let j = &network.junctions[n];
    if j.radius <= 0.0 || j.edges.len() < 2 {
        return mesh;
    }
    let centre = j.position + Vec3::Z * 0.035;
    let ring: Vec<Vec3> = (0..=24)
        .map(|k| centre + Vec2::from_angle(k as f32 / 24.0 * std::f32::consts::TAU).extend(0.0) * j.radius)
        .collect();
    let c = mesh.vertex(centre, Vec3::Z, ASPHALT);
    for k in 0..24 {
        let a = mesh.vertex(ring[k], Vec3::Z, ASPHALT);
        let b = mesh.vertex(ring[k + 1], Vec3::Z, ASPHALT);
        mesh.indices.extend([c, a, b]);
    }
    mesh
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cuboid_faces_point_outward() {
        let mut mesh = Mesh::default();
        // A left-handed set of axes must still face out.
        mesh.cuboid(Vec3::new(1.0, 2.0, 3.0), [Vec3::Y, Vec3::X * 2.0, Vec3::Z], [1.0; 3], [0.5; 3]);
        assert_eq!(mesh.indices.len(), 36);
        for t in mesh.indices.chunks(3) {
            let [a, b, c] = [t[0], t[1], t[2]].map(|i| Vec3::from_array(mesh.positions[i as usize]));
            let face = (b - a).cross(c - a);
            let out = (a + b + c) / 3.0 - Vec3::new(1.0, 2.0, 3.0);
            assert!(face.dot(out) > 0.0);
        }
    }
}
