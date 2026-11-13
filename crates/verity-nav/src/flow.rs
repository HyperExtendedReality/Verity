//! A distance field over a grid toward one or more goals. Any number of
//! agents read their heading from it, so one search moves a whole horde.

use std::collections::BinaryHeap;

use glam::{IVec2, Vec2, Vec3};

use crate::NavMesh;

const STEPS: [(IVec2, f32); 8] = [
    (IVec2::new(1, 0), 1.0),
    (IVec2::new(-1, 0), 1.0),
    (IVec2::new(0, 1), 1.0),
    (IVec2::new(0, -1), 1.0),
    (IVec2::new(1, 1), std::f32::consts::SQRT_2),
    (IVec2::new(-1, 1), std::f32::consts::SQRT_2),
    (IVec2::new(1, -1), std::f32::consts::SQRT_2),
    (IVec2::new(-1, -1), std::f32::consts::SQRT_2),
];

#[derive(Clone, Debug)]
pub struct FlowField {
    pub origin: Vec2,
    pub cell: f32,
    pub size: [u32; 2],
    walkable: Vec<bool>,
    /// Metres to the nearest goal, through walkable cells.
    distance: Vec<f32>,
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

impl FlowField {
    /// A grid of `size` cells from `origin`; `walkable` is asked about each
    /// cell's centre. No goal yet: every heading is zero until [`Self::solve`].
    pub fn new(origin: Vec2, cell: f32, size: [u32; 2], walkable: impl Fn(Vec2) -> bool) -> Self {
        let size = size.map(|s| s.clamp(1, 8192));
        let n = (size[0] * size[1]) as usize;
        let mut field = Self {
            origin,
            cell: cell.max(0.01),
            size,
            walkable: Vec::with_capacity(n),
            distance: vec![f32::INFINITY; n],
        };
        for i in 0..n as u32 {
            let c = IVec2::new((i % size[0]) as i32, (i / size[0]) as i32);
            field.walkable.push(walkable(field.centre(c)));
        }
        field
    }

    /// Covering `mesh`, walkable wherever a cell's centre is over it.
    pub fn from_navmesh(mesh: &NavMesh, cell: f32) -> Self {
        let (lo, hi) = (mesh.vertices.iter()).fold((Vec2::MAX, Vec2::MIN), |(lo, hi), v| {
            (lo.min(v.truncate()), hi.max(v.truncate()))
        });
        if lo.x > hi.x {
            return Self::new(Vec2::ZERO, cell, [1, 1], |_| false);
        }
        let size = ((hi - lo) / cell).ceil().as_uvec2().max(glam::UVec2::ONE);
        let heights: Vec<f32> = mesh.vertices.iter().map(|v| v.z).collect();
        let mid = heights.iter().sum::<f32>() / heights.len() as f32;
        Self::new(lo, cell, size.to_array(), |p| mesh.locate(p.extend(mid)).is_some())
    }

    fn centre(&self, c: IVec2) -> Vec2 {
        self.origin + (c.as_vec2() + 0.5) * self.cell
    }

    fn cell_of(&self, p: Vec2) -> IVec2 {
        let c = ((p - self.origin) / self.cell).floor();
        if c.is_finite() {
            c.clamp(Vec2::splat(-1.0), Vec2::splat(9000.0)).as_ivec2()
        } else {
            IVec2::splat(-1)
        }
    }

    fn index(&self, c: IVec2) -> Option<usize> {
        let [w, h] = self.size.map(|s| s as i32);
        (c.x >= 0 && c.y >= 0 && c.x < w && c.y < h).then(|| (c.y * w + c.x) as usize)
    }

    fn open(&self, c: IVec2) -> bool {
        self.index(c).is_some_and(|i| self.walkable[i])
    }

    /// A diagonal step only where both cells beside it are open, so no
    /// heading cuts a blocked corner.
    fn can_step(&self, from: IVec2, step: IVec2) -> bool {
        self.open(from + step)
            && (step.x == 0 || step.y == 0 || (self.open(from + IVec2::new(step.x, 0)) && self.open(from + IVec2::new(0, step.y))))
    }

    /// Recompute distances to the nearest of `goals`.
    pub fn solve(&mut self, goals: &[Vec3]) {
        self.distance.fill(f32::INFINITY);
        let mut open = BinaryHeap::new();
        for g in goals {
            let c = self.cell_of(g.truncate());
            if let Some(i) = self.index(c).filter(|&i| self.walkable[i]) {
                self.distance[i] = 0.0;
                open.push(Open(0.0, i as u32));
            }
        }
        let w = self.size[0];
        while let Some(Open(d, i)) = open.pop() {
            if d > self.distance[i as usize] {
                continue;
            }
            let c = IVec2::new((i % w) as i32, (i / w) as i32);
            for (step, length) in STEPS {
                if !self.can_step(c, step) {
                    continue;
                }
                let j = self.index(c + step).unwrap();
                let through = d + length * self.cell;
                if through < self.distance[j] {
                    self.distance[j] = through;
                    open.push(Open(through, j as u32));
                }
            }
        }
    }

    /// Metres from `p` to the nearest goal; infinite off the field or cut off.
    pub fn distance(&self, p: Vec3) -> f32 {
        self.index(self.cell_of(p.truncate()))
            .map_or(f32::INFINITY, |i| self.distance[i])
    }

    /// Unit heading toward the goal from `p`: to the neighbouring cell nearest
    /// it. A point that strayed into a blocked cell heads back out the same way.
    /// Zero at a goal, off the field or where no goal can be reached.
    pub fn direction(&self, p: Vec3) -> Vec2 {
        let c = self.cell_of(p.truncate());
        let Some(i) = self.index(c) else {
            return Vec2::ZERO;
        };
        let here = if self.walkable[i] { self.distance[i] } else { f32::INFINITY };
        let blocked = !self.walkable[i];
        let best = STEPS
            .iter()
            .filter(|(step, _)| if blocked { self.open(c + *step) } else { self.can_step(c, *step) })
            .map(|&(step, _)| (step, self.distance[self.index(c + step).unwrap()]))
            .filter(|&(_, d)| d < here)
            .min_by(|a, b| a.1.total_cmp(&b.1));
        match best {
            Some((step, _)) => (self.centre(c + step) - p.truncate()).normalize_or_zero(),
            None => Vec2::ZERO,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_reachable_start_follows_the_field_to_the_goal() {
        // A wall at x = 15 with a gap at the top, and a pocket sealed by a
        // blocked band around it.
        let wall = |p: Vec2| (p.x - 15.0).abs() < 1.0 && p.y < 25.0;
        let band = |p: Vec2| p.x < 4.0 && p.y > 26.0 && !(p.x < 3.0 && p.y > 27.0);
        let mut field = FlowField::new(Vec2::ZERO, 1.0, [30, 30], |p| !wall(p) && !band(p));
        let goal = Vec3::new(25.5, 5.5, 0.0);
        field.solve(&[goal]);
        assert!(field.distance(Vec3::new(1.5, 28.5, 0.0)).is_infinite());
        for start in [(2.0, 2.0), (10.0, 20.0), (13.2, 3.0), (20.0, 28.0), (29.5, 0.5)] {
            let mut p = Vec3::new(start.0, start.1, 0.0);
            assert!(field.distance(p).is_finite(), "{p}");
            for _ in 0..3000 {
                let d = field.direction(p);
                if d == Vec2::ZERO {
                    break;
                }
                p += d.extend(0.0) * 0.25;
            }
            assert!(p.distance(goal) < 1.0, "from {start:?} ended at {p}");
        }
        assert!(field.distance(goal) == 0.0);
        assert_eq!(field.direction(Vec3::new(-50.0, 0.0, 0.0)), Vec2::ZERO);
    }

    #[test]
    fn a_navmesh_rasterizes_to_a_field() {
        let mesh = NavMesh::from_heights(Vec2::ZERO, 1.0, [10, 4], 0.5, |p| (p.x < 4.0 || p.x > 6.0 || p.y < 1.0).then_some(2.0));
        let mut field = FlowField::from_navmesh(&mesh, 0.5);
        field.solve(&[Vec3::new(9.0, 3.0, 2.0)]);
        assert!(field.distance(Vec3::new(1.0, 3.0, 2.0)) > 10.0);
        assert!(field.distance(Vec3::new(5.0, 3.0, 2.0)).is_infinite());
    }
}
