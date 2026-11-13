//! Walkable surfaces as triangles with adjacency. Paths are found over the
//! triangles with A*, then pulled tight with the funnel algorithm.

use std::collections::{BinaryHeap, HashMap};

use glam::{Vec2, Vec3};

use crate::NONE;

#[derive(Clone, Debug, Default)]
pub struct NavMesh {
    pub vertices: Vec<Vec3>,
    /// Counter-clockwise seen from above.
    pub triangles: Vec<[u32; 3]>,
    /// The triangle across edge `i` (vertex `i` to `i + 1`), or [`NONE`].
    pub neighbors: Vec<[u32; 3]>,
    centroids: Vec<Vec3>,
    cell: f32,
    /// Ground-plane cell keys and the triangles overlapping them, sorted.
    cells: Vec<(u64, u32)>,
}

fn key(x: i32, y: i32) -> u64 {
    (x as u32 as u64) << 32 | y as u32 as u64
}

fn cross(a: Vec2, b: Vec2, c: Vec2) -> f32 {
    (b - a).perp_dot(c - a)
}

/// The point of triangle `t` nearest `p` in the ground plane.
fn closest_2d(p: Vec2, t: [Vec2; 3]) -> Vec2 {
    let inside = (0..3).all(|i| cross(t[i], t[(i + 1) % 3], p) >= 0.0);
    if inside {
        return p;
    }
    (0..3)
        .map(|i| {
            let (a, b) = (t[i], t[(i + 1) % 3]);
            let ab = b - a;
            a + ab * ((p - a).dot(ab) / ab.length_squared().max(1e-12)).clamp(0.0, 1.0)
        })
        .min_by(|a, b| a.distance_squared(p).total_cmp(&b.distance_squared(p)))
        .unwrap()
}

#[derive(PartialEq)]
struct Open(f32, u32);
impl Eq for Open {}
impl Ord for Open {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other.0.total_cmp(&self.0).then(self.1.cmp(&other.1))
    }
}
impl PartialOrd for Open {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl NavMesh {
    /// From a triangle soup: vertices closer than `weld` merge, and triangles
    /// steeper than `max_slope` (radians) or facing down are left out. Pieces
    /// connect only where they share welded edges.
    pub fn from_triangles(positions: &[Vec3], indices: &[u32], max_slope: f32, weld: f32) -> Self {
        let weld = weld.max(1e-4);
        let mut welded: HashMap<[i64; 3], u32> = HashMap::new();
        let mut vertices = Vec::new();
        let mut index = |p: Vec3| -> u32 {
            let q = (p / weld).round().as_i64vec3().to_array();
            *welded.entry(q).or_insert_with(|| {
                vertices.push(p);
                vertices.len() as u32 - 1
            })
        };
        let min_up = max_slope.cos();
        let mut triangles = Vec::new();
        for t in indices.as_chunks::<3>().0 {
            let Some(p) = t.iter().map(|&i| positions.get(i as usize).copied()).collect::<Option<Vec<_>>>() else {
                continue;
            };
            if !p.iter().all(|v| v.is_finite()) {
                continue;
            }
            let normal = (p[1] - p[0]).cross(p[2] - p[0]).normalize_or_zero();
            if normal.z < min_up {
                continue;
            }
            let t = [index(p[0]), index(p[1]), index(p[2])];
            if t[0] != t[1] && t[1] != t[2] && t[0] != t[2] {
                triangles.push(t);
            }
        }
        Self::assemble(vertices, triangles)
    }

    /// From ground sampled on a grid: `sample` gives the height at a point, or
    /// `None` where agents cannot stand (inflate obstacles by agent radius).
    /// Cells whose corners differ by more than `max_step` are left out.
    pub fn from_heights(
        origin: Vec2,
        cell: f32,
        size: [u32; 2],
        max_step: f32,
        sample: impl Fn(Vec2) -> Option<f32>,
    ) -> Self {
        let [w, h] = size.map(|s| s.min(4096) as usize);
        let corner = |x: usize, y: usize| origin + Vec2::new(x as f32, y as f32) * cell;
        let heights: Vec<Option<f32>> = (0..(w + 1) * (h + 1))
            .map(|i| sample(corner(i % (w + 1), i / (w + 1))).filter(|z| z.is_finite()))
            .collect();
        let mut ids = vec![NONE; heights.len()];
        let mut vertices = Vec::new();
        let mut triangles = Vec::new();
        for y in 0..h {
            for x in 0..w {
                let c = [(x, y), (x + 1, y), (x + 1, y + 1), (x, y + 1)];
                let i = c.map(|(x, y)| y * (w + 1) + x);
                let Some(z) = i.iter().map(|&i| heights[i]).collect::<Option<Vec<f32>>>() else {
                    continue;
                };
                let (lo, hi) = z.iter().fold((f32::MAX, f32::MIN), |(l, u), &z| (l.min(z), u.max(z)));
                if hi - lo > max_step {
                    continue;
                }
                let v = i.map(|i| {
                    if ids[i] == NONE {
                        let (x, y) = (i % (w + 1), i / (w + 1));
                        ids[i] = vertices.len() as u32;
                        vertices.push(corner(x, y).extend(heights[i].unwrap()));
                    }
                    ids[i]
                });
                triangles.extend([[v[0], v[1], v[2]], [v[0], v[2], v[3]]]);
            }
        }
        Self::assemble(vertices, triangles)
    }

    fn assemble(vertices: Vec<Vec3>, triangles: Vec<[u32; 3]>) -> Self {
        let mut edges: HashMap<(u32, u32), (u32, usize)> = HashMap::new();
        let mut neighbors = vec![[NONE; 3]; triangles.len()];
        for (t, tri) in triangles.iter().enumerate() {
            for i in 0..3 {
                let (a, b) = (tri[i], tri[(i + 1) % 3]);
                if let Some(&(other, j)) = edges.get(&(b, a))
                    && neighbors[other as usize][j] == NONE
                {
                    neighbors[other as usize][j] = t as u32;
                    neighbors[t][i] = other;
                } else {
                    edges.insert((a, b), (t as u32, i));
                }
            }
        }
        let centroids: Vec<Vec3> = (triangles.iter())
            .map(|t| t.iter().map(|&i| vertices[i as usize]).sum::<Vec3>() / 3.0)
            .collect();
        // Cells about two triangles across keep lookups short.
        let span: f32 = (triangles.iter())
            .map(|t| vertices[t[0] as usize].truncate().distance(vertices[t[1] as usize].truncate()))
            .sum::<f32>()
            / triangles.len().max(1) as f32;
        let cell = (span * 2.0).max(0.5);
        let mut cells = Vec::new();
        for (t, tri) in triangles.iter().enumerate() {
            let p = tri.map(|i| vertices[i as usize].truncate());
            let lo = (p[0].min(p[1]).min(p[2]) / cell).floor().as_ivec2();
            let hi = (p[0].max(p[1]).max(p[2]) / cell).floor().as_ivec2();
            for y in lo.y..=hi.y.min(lo.y + 256) {
                for x in lo.x..=hi.x.min(lo.x + 256) {
                    cells.push((key(x, y), t as u32));
                }
            }
        }
        cells.sort_unstable();
        Self {
            vertices,
            triangles,
            neighbors,
            centroids,
            cell,
            cells,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.triangles.is_empty()
    }

    fn corners(&self, t: u32) -> [Vec3; 3] {
        self.triangles[t as usize].map(|i| self.vertices[i as usize])
    }

    fn in_cell(&self, x: i32, y: i32) -> &[(u64, u32)] {
        let k = key(x, y);
        let lo = self.cells.partition_point(|c| c.0 < k);
        let hi = self.cells.partition_point(|c| c.0 <= k);
        &self.cells[lo..hi]
    }

    /// Height of triangle `t`'s plane at `p`.
    fn plane_height(&self, t: u32, p: Vec2) -> f32 {
        let [a, b, c] = self.corners(t);
        let area = cross(a.truncate(), b.truncate(), c.truncate());
        if area.abs() < 1e-12 {
            return a.z;
        }
        let u = cross(b.truncate(), c.truncate(), p) / area;
        let v = cross(c.truncate(), a.truncate(), p) / area;
        a.z * u + b.z * v + c.z * (1.0 - u - v)
    }

    /// The triangle under or over `p` whose surface is vertically nearest it.
    pub fn locate(&self, p: Vec3) -> Option<u32> {
        if !p.is_finite() || self.is_empty() {
            return None;
        }
        let c = (p.truncate() / self.cell).floor().as_ivec2();
        self.in_cell(c.x, c.y)
            .iter()
            .filter(|&&(_, t)| {
                let [a, b, c] = self.corners(t).map(|v| v.truncate());
                let q = p.truncate();
                cross(a, b, q) >= -1e-5 && cross(b, c, q) >= -1e-5 && cross(c, a, q) >= -1e-5
            })
            .map(|&(_, t)| (t, (self.plane_height(t, p.truncate()) - p.z).abs()))
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(t, _)| t)
    }

    /// Height of the surface at `p`, if `p` is over the mesh.
    pub fn height(&self, p: Vec3) -> Option<f32> {
        self.locate(p).map(|t| self.plane_height(t, p.truncate()))
    }

    /// The point of the mesh nearest `p` in the ground plane within `radius`,
    /// on its surface, and its triangle.
    pub fn nearest(&self, p: Vec3, radius: f32) -> Option<(u32, Vec3)> {
        if let Some(t) = self.locate(p) {
            return Some((t, p.truncate().extend(self.plane_height(t, p.truncate()))));
        }
        if !p.is_finite() || radius.is_nan() || radius <= 0.0 || self.is_empty() {
            return None;
        }
        let q = p.truncate();
        let lo = ((q - radius) / self.cell).floor().as_ivec2();
        let hi = ((q + radius) / self.cell).floor().as_ivec2();
        if (hi - lo).max_element() > 256 {
            return None;
        }
        let mut best: Option<(u32, Vec2, f32)> = None;
        for y in lo.y..=hi.y {
            for x in lo.x..=hi.x {
                for &(_, t) in self.in_cell(x, y) {
                    let at = closest_2d(q, self.corners(t).map(|v| v.truncate()));
                    // Prefer surfaces at the point's level over ones above or below.
                    let score = at.distance_squared(q) + (self.plane_height(t, at) - p.z).powi(2) * 0.25;
                    if best.is_none_or(|b| score < b.2) && at.distance(q) <= radius {
                        best = Some((t, at, score));
                    }
                }
            }
        }
        best.map(|(t, at, _)| (t, at.extend(self.plane_height(t, at))))
    }

    /// Triangles from `start` to `goal` across shared edges, by A* over centroids.
    pub fn corridor(&self, start: u32, goal: u32, target: Vec3) -> Option<Vec<u32>> {
        let n = self.triangles.len();
        if start as usize >= n || goal as usize >= n {
            return None;
        }
        let mut cost = vec![f32::INFINITY; n];
        let mut came = vec![NONE; n];
        let mut open = BinaryHeap::new();
        cost[start as usize] = 0.0;
        open.push(Open(0.0, start));
        while let Some(Open(_, t)) = open.pop() {
            if t == goal {
                let mut corridor = vec![goal];
                while let Some(&last) = corridor.last().filter(|&&l| l != start) {
                    corridor.push(came[last as usize]);
                }
                corridor.reverse();
                return Some(corridor);
            }
            for &next in self.neighbors[t as usize].iter().filter(|&&n| n != NONE) {
                let through = cost[t as usize]
                    + self.centroids[t as usize].distance(self.centroids[next as usize]);
                if through < cost[next as usize] {
                    cost[next as usize] = through;
                    came[next as usize] = t;
                    open.push(Open(through + self.centroids[next as usize].distance(target), next));
                }
            }
        }
        None
    }

    /// A shortest-looking path over the mesh from `from` to `to`, each snapped
    /// to the mesh within `snap`. Its first point is the start.
    pub fn find_path(&self, from: Vec3, to: Vec3, snap: f32) -> Option<Vec<Vec3>> {
        let (start, from) = self.nearest(from, snap)?;
        let (goal, to) = self.nearest(to, snap)?;
        let corridor = self.corridor(start, goal, to)?;
        // Portals between consecutive triangles, as (left, right) seen walking.
        let mut portals = vec![(from, from)];
        for pair in corridor.windows(2) {
            let tri = self.triangles[pair[0] as usize];
            let edge = (0..3).find(|&i| self.neighbors[pair[0] as usize][i] == pair[1])?;
            let (a, b) = (self.vertices[tri[edge] as usize], self.vertices[tri[(edge + 1) % 3] as usize]);
            portals.push((b, a));
        }
        portals.push((to, to));
        // The funnel is tight within the corridor, but A* may pick a narrow
        // staircase among equal ones: drop corners a straight line can skip.
        let pulled = funnel(&portals);
        let mut path = vec![pulled[0]];
        let mut i = 0;
        while i + 1 < pulled.len() {
            let far = (i + 1..pulled.len())
                .rev()
                .find(|&j| j == i + 1 || self.line_of_sight(pulled[i], pulled[j]))
                .unwrap();
            path.push(pulled[far]);
            i = far;
        }
        Some(path)
    }

    /// Whether the straight line from `a` to `b` stays on the mesh, walked
    /// across shared edges from the triangle under `a`.
    pub fn line_of_sight(&self, a: Vec3, b: Vec3) -> bool {
        let Some(mut t) = self.locate(a) else {
            return false;
        };
        let (a2, b2) = (a.truncate(), b.truncate());
        for _ in 0..self.triangles.len() {
            let p = self.corners(t).map(|v| v.truncate());
            if (0..3).all(|i| cross(p[i], p[(i + 1) % 3], b2) >= -1e-5) {
                return true;
            }
            // The edge `b` lies beyond that the segment passes through.
            let exit = (0..3).find(|&i| {
                let (u, v) = (p[i], p[(i + 1) % 3]);
                cross(u, v, b2) < 0.0 && cross(a2, b2, u) * cross(a2, b2, v) <= 0.0
            });
            match exit.map(|i| self.neighbors[t as usize][i]) {
                Some(next) if next != NONE => t = next,
                _ => return false,
            }
        }
        false
    }
}

/// Pull a path tight through portals given as (left, right) points.
fn funnel(portals: &[(Vec3, Vec3)]) -> Vec<Vec3> {
    let f = |p: Vec3| p.truncate();
    let mut path = vec![portals[0].0];
    let (mut apex, mut left, mut right) = (portals[0].0, portals[0].0, portals[0].1);
    let (mut left_i, mut right_i) = (0, 0);
    let mut i = 1;
    while i < portals.len() {
        let (l, r) = portals[i];
        // Tighten the right side, unless it would cross over the left, which
        // then becomes a corner of the path and the funnel's new apex.
        if cross(f(apex), f(right), f(r)) >= 0.0 {
            // On the left ray is not across it: grid meshes are full of collinear points.
            if apex == right || cross(f(apex), f(left), f(r)) <= 0.0 {
                right = r;
                right_i = i;
            } else {
                path.push(left);
                (apex, right, right_i) = (left, left, left_i);
                i = left_i + 1;
                continue;
            }
        }
        if cross(f(apex), f(left), f(l)) <= 0.0 {
            if apex == left || cross(f(apex), f(right), f(l)) >= 0.0 {
                left = l;
                left_i = i;
            } else {
                path.push(right);
                (apex, left, left_i) = (right, right, right_i);
                i = right_i + 1;
                continue;
            }
        }
        i += 1;
    }
    let end = portals[portals.len() - 1].0;
    if path.last() != Some(&end) {
        path.push(end);
    }
    path
}

#[cfg(test)]
mod tests {
    use super::*;

    fn on_mesh(mesh: &NavMesh, path: &[Vec3]) -> bool {
        path.windows(2).all(|pair| {
            (0..=20).all(|k| mesh.locate(pair[0].lerp(pair[1], k as f32 / 20.0)).is_some())
        })
    }

    #[test]
    fn an_l_shaped_corridor_path_bends_at_the_corner() {
        // Walkable: y < 5, or x > 15, of a 20 m square.
        let mesh = NavMesh::from_heights(Vec2::ZERO, 1.0, [20, 20], 0.5, |p| {
            (p.y <= 5.0 || p.x >= 15.0).then_some(0.0)
        });
        // From a vertex, and from inside a triangle.
        for start in [Vec3::new(2.3, 1.6, 0.0), Vec3::new(2.0, 2.0, 0.0)] {
            let path = mesh.find_path(start, Vec3::new(17.0, 18.0, 0.0), 1.0).unwrap();
            assert!(on_mesh(&mesh, &path), "{path:?}");
            assert_eq!(path.len(), 3, "{path:?}");
            assert!(path[1].truncate().distance(Vec2::new(15.0, 5.0)) < 0.01, "{path:?}");
        }
        // An open floor gives a straight line.
        let open = NavMesh::from_heights(Vec2::ZERO, 1.0, [20, 20], 0.5, |_| Some(1.0));
        let path = open.find_path(Vec3::new(1.0, 1.0, 1.0), Vec3::new(18.0, 13.0, 1.0), 1.0).unwrap();
        assert_eq!(path.len(), 2);
        assert_eq!(open.height(Vec3::new(4.0, 4.0, 0.0)), Some(1.0));
    }

    #[test]
    fn a_wall_with_a_gap_is_walked_around() {
        let mesh = NavMesh::from_heights(Vec2::ZERO, 1.0, [30, 30], 0.5, |p| {
            (!(p.x >= 14.0 && p.x <= 16.0 && p.y <= 25.0)).then_some(0.0)
        });
        let path = mesh.find_path(Vec3::new(5.0, 5.0, 0.0), Vec3::new(25.0, 5.0, 0.0), 1.0).unwrap();
        assert!(on_mesh(&mesh, &path));
        let length: f32 = path.windows(2).map(|p| p[0].distance(p[1])).sum();
        assert!(length > 40.0, "{length}");
        // A closed wall has no way through.
        let shut = NavMesh::from_heights(Vec2::ZERO, 1.0, [30, 30], 0.5, |p| {
            (!(14.0..=16.0).contains(&p.x)).then_some(0.0)
        });
        assert!(shut.find_path(Vec3::new(5.0, 5.0, 0.0), Vec3::new(25.0, 5.0, 0.0), 1.0).is_none());
    }

    #[test]
    fn triangle_soup_welds_and_drops_steep_faces() {
        // Two floor quads sharing an edge with seams under the weld size, and a wall.
        let p = [
            Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 0.0), Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(1.0001, 0.0, 0.0), Vec3::new(2.0, 0.0, 0.0), Vec3::new(2.0, 1.0, 0.0), Vec3::new(1.0, 1.0001, 0.0),
            Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 2.0), Vec3::new(0.0, 1.0, 2.0),
        ];
        let i = [0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7, 8, 9, 10];
        let mesh = NavMesh::from_triangles(&p, &i, 0.8, 0.01);
        assert_eq!(mesh.triangles.len(), 4);
        let path = mesh.find_path(Vec3::new(0.2, 0.5, 0.0), Vec3::new(1.8, 0.5, 0.0), 0.5).unwrap();
        assert_eq!(path.len(), 2);
        assert!(mesh.nearest(Vec3::new(3.0, 0.5, 0.0), 1.5).unwrap().1.distance(Vec3::new(2.0, 0.5, 0.0)) < 1e-4);
    }
}
