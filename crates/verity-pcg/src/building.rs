//! Buildings on lots: how tall (taller toward the town centre), their shell
//! with window bands, and, on demand, an interior of floors, rooms and doors.

use glam::{Vec2, Vec3};

use crate::{hash, unit, Lot, Mesh, Rules};

const GLASS: [f32; 3] = [0.06, 0.08, 0.11];
const FLOOR: [f32; 3] = [0.35, 0.3, 0.25];
const PARTITION: [f32; 3] = [0.75, 0.73, 0.68];
const PALETTE: [[f32; 3]; 6] = [
    [0.55, 0.52, 0.48],
    [0.45, 0.22, 0.15],
    [0.62, 0.58, 0.5],
    [0.3, 0.33, 0.36],
    [0.7, 0.66, 0.58],
    [0.38, 0.3, 0.24],
];
/// Smallest side a room is split down to.
const ROOM: f32 = 4.0;

#[derive(Clone, Debug, PartialEq)]
pub struct Building {
    pub lot: Lot,
    pub floors: u32,
    pub color: [f32; 3],
}

/// A building's own axes: base centre, unit along the street, unit inward,
/// half its size along each, and its height.
#[derive(Clone, Copy, Debug)]
pub struct Frame {
    pub base: Vec3,
    pub along: Vec3,
    pub inward: Vec3,
    pub half: Vec2,
    pub height: f32,
}

impl Frame {
    fn at(&self, u: f32, v: f32, z: f32) -> Vec3 {
        self.base + self.along * u + self.inward * v + Vec3::Z * z
    }

    /// A box from local centre and half sizes.
    fn cuboid(&self, mesh: &mut Mesh, (u, v, z): (f32, f32, f32), (hu, hv, hz): (f32, f32, f32), color: [f32; 3]) {
        mesh.cuboid(self.at(u, v, z), [self.along * hu, self.inward * hv, Vec3::Z * hz], color, color);
    }
}

impl Building {
    pub fn plan(lot: Lot, rules: &Rules, town: Vec2) -> Self {
        let low = rules.floors[0].clamp(1, 200);
        let high = rules.floors[1].clamp(low, 200);
        let d = lot.centre.truncate().distance(town);
        let density = (-(d / rules.downtown.max(1.0)).powi(2)).exp();
        let r = unit(hash(lot.seed, 1));
        let floors = low + ((high - low) as f32 * density * (0.4 + 0.6 * r)).round() as u32;
        let base = PALETTE[hash(lot.seed, 2) as usize % PALETTE.len()];
        let tint = 0.85 + unit(hash(lot.seed, 3)) * 0.3;
        Self {
            lot,
            floors,
            color: base.map(|c| (c * tint).min(1.0)),
        }
    }

    pub fn frame(&self, rules: &Rules) -> Frame {
        Frame {
            base: self.lot.centre,
            along: self.lot.front.extend(0.0),
            inward: self.lot.inward().extend(0.0),
            half: (self.lot.half - Vec2::splat(rules.setback.max(0.0))).max(Vec2::splat(2.0)),
            height: self.floors as f32 * rules.floor_height.clamp(2.0, 10.0),
        }
    }

    /// Whether `p` is inside the building.
    pub fn contains(&self, rules: &Rules, p: Vec3) -> bool {
        let f = self.frame(rules);
        let d = p - f.base;
        d.dot(f.along).abs() <= f.half.x && d.dot(f.inward).abs() <= f.half.y && (-1.0..=f.height).contains(&d.z)
    }

    /// The closed outside: walls reaching a metre below the lot to meet
    /// sloping ground, a roof, and a band of glass per floor.
    pub fn shell(&self, rules: &Rules, mesh: &mut Mesh) {
        let f = self.frame(rules);
        let roof = self.color.map(|c| c * 0.6);
        mesh.cuboid(
            f.at(0.0, 0.0, f.height * 0.5 - 0.5),
            [f.along * f.half.x, f.inward * f.half.y, Vec3::Z * (f.height * 0.5 + 0.5)],
            self.color,
            roof,
        );
        let storey = rules.floor_height.clamp(2.0, 10.0);
        for floor in 0..self.floors {
            let bottom = floor as f32 * storey + if floor == 0 { 0.4 } else { 0.9 };
            let top = (floor + 1) as f32 * storey - 0.6;
            let (mid, tall) = ((bottom + top) * 0.5, (top - bottom) * 0.5);
            for (normal, along, reach) in [
                (f.inward, f.along, f.half.y),
                (-f.inward, f.along, f.half.y),
                (f.along, f.inward, f.half.x),
                (-f.along, f.inward, f.half.x),
            ] {
                let width = if normal.dot(f.along).abs() > 0.5 { f.half.y } else { f.half.x };
                let centre = f.base + normal * (reach + 0.02) + Vec3::Z * mid;
                mesh.panel(centre, along * (width - 0.6).max(0.2), Vec3::Z * tall, normal, GLASS);
            }
        }
    }

    /// Floors, rooms split with door gaps, and outer walls with window
    /// openings, to walk through in place of the shell.
    pub fn interior(&self, rules: &Rules, mesh: &mut Mesh) {
        let f = self.frame(rules);
        let storey = f.height / self.floors.max(1) as f32;
        let wall = 0.08;
        for floor in 0..=self.floors {
            let z = floor as f32 * storey;
            f.cuboid(mesh, (0.0, 0.0, z - 0.1), (f.half.x, f.half.y, 0.1), FLOOR);
        }
        for floor in 0..self.floors {
            let z = floor as f32 * storey;
            let seed = hash(self.lot.seed, floor + 10);
            let mut walls = Vec::new();
            split((-f.half, f.half), seed, 0, &mut walls);
            let height = storey - 0.2;
            for (a, b, door) in walls {
                // Two pieces either side of a 1 m door.
                let length = a.distance(b);
                let dir = (b - a) / length.max(1e-3);
                let gap = (door * (length - 2.0) + 1.0).clamp(0.5, (length - 0.5).max(0.5));
                for (s0, s1) in [(0.0, gap - 0.5), (gap + 0.5, length)] {
                    if s1 - s0 < 0.1 {
                        continue;
                    }
                    let mid = a + dir * (s0 + s1) * 0.5;
                    let half = if dir.x.abs() > 0.5 { (0.5 * (s1 - s0), wall) } else { (wall, 0.5 * (s1 - s0)) };
                    f.cuboid(mesh, (mid.x, mid.y, z + height * 0.5), (half.0, half.1, height * 0.5), PARTITION);
                }
            }
            // Outer walls: sill, lintel, and piers between windows.
            let (sill, lintel) = (0.9, storey - 0.6);
            for (axis, sign) in [(0, 1.0), (0, -1.0), (1, 1.0), (1, -1.0)] {
                let (reach, run) = if axis == 0 { (f.half.y, f.half.x) } else { (f.half.x, f.half.y) };
                let place = |s: f32, half_run: f32, zc: f32, hz: f32, mesh: &mut Mesh| {
                    let (centre, half) = if axis == 0 {
                        ((s, sign * reach, zc), (half_run, wall, hz))
                    } else {
                        ((sign * reach, s, zc), (wall, half_run, hz))
                    };
                    f.cuboid(mesh, centre, half, self.color);
                };
                place(0.0, run, z + sill * 0.5, sill * 0.5, mesh);
                place(0.0, run, z + (lintel + storey) * 0.5, (storey - lintel) * 0.5, mesh);
                let bays = ((run * 2.0) / 3.0).floor().max(1.0);
                let bay = run * 2.0 / bays;
                let pier = (bay - 1.6).max(0.3) * 0.5;
                for k in 0..=bays as u32 {
                    let s = (-run + k as f32 * bay).clamp(-run + pier, run - pier);
                    place(s, pier, z + (sill + lintel) * 0.5, (lintel - sill) * 0.5, mesh);
                }
            }
        }
    }
}

/// Walls splitting a rectangle into rooms, each as its ends and where along
/// it its door is, from 0 to 1.
fn split((lo, hi): (Vec2, Vec2), seed: u32, depth: u32, walls: &mut Vec<(Vec2, Vec2, f32)>) {
    let size = hi - lo;
    if depth >= 4 || size.max_element() < ROOM * 2.0 {
        return;
    }
    let cut = 0.35 + unit(hash(seed, depth)) * 0.3;
    let door = unit(hash(seed, depth + 100));
    let (a, b) = if size.x >= size.y {
        let x = lo.x + size.x * cut;
        walls.push((Vec2::new(x, lo.y), Vec2::new(x, hi.y), door));
        ((lo, Vec2::new(x, hi.y)), (Vec2::new(x, lo.y), hi))
    } else {
        let y = lo.y + size.y * cut;
        walls.push((Vec2::new(lo.x, y), Vec2::new(hi.x, y), door));
        ((lo, Vec2::new(hi.x, y)), (Vec2::new(lo.x, y), hi))
    };
    split(a, hash(seed, 1), depth + 1, walls);
    split(b, hash(seed, 2), depth + 1, walls);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lot(seed: u32, at: Vec2) -> Lot {
        Lot { centre: at.extend(0.0), front: Vec2::X, half: Vec2::new(10.0, 12.0), block: 0, seed }
    }

    #[test]
    fn buildings_grow_taller_downtown_and_have_rooms() {
        let rules = Rules::default();
        let mean = |at: Vec2| (0..50).map(|s| Building::plan(lot(s, at), &rules, Vec2::ZERO).floors).sum::<u32>() as f32 / 50.0;
        assert!(mean(Vec2::ZERO) > mean(Vec2::splat(1000.0)) + 3.0);
        let building = Building::plan(lot(4, Vec2::ZERO), &rules, Vec2::ZERO);
        assert!(building.floors >= rules.floors[0] && building.floors <= rules.floors[1]);
        let mut walls = Vec::new();
        split((Vec2::splat(-8.5), Vec2::new(8.5, 10.5)), 3, 0, &mut walls);
        assert!(walls.len() >= 3);
        let (mut shell, mut inside) = (Mesh::default(), Mesh::default());
        building.shell(&rules, &mut shell);
        building.interior(&rules, &mut inside);
        assert!(!shell.is_empty() && inside.positions.len() > shell.positions.len());
        assert!(building.contains(&rules, Vec3::new(0.0, 0.0, 2.0)));
        assert!(!building.contains(&rules, Vec3::new(0.0, 30.0, 2.0)));
    }
}
