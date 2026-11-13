//! Neighbour queries over many moving points: a hash of ground-plane cells,
//! rebuilt from scratch each tick by a counting sort, so it needs no
//! per-point bookkeeping and no allocation once warm.

use glam::{IVec2, Vec3};

#[derive(Clone, Debug)]
pub struct SpatialHash {
    cell: f32,
    mask: u32,
    /// Where each bucket's points start in `items`; one more entry than buckets.
    start: Vec<u32>,
    items: Vec<u32>,
    cells: Vec<IVec2>,
}

impl SpatialHash {
    /// `cell` is best around the usual query radius.
    pub fn new(cell: f32) -> Self {
        assert!(cell.is_finite() && cell > 0.0, "cell size must be positive");
        Self {
            cell,
            mask: 0,
            start: vec![0, 0],
            items: Vec::new(),
            cells: Vec::new(),
        }
    }

    fn cell_of(&self, p: Vec3) -> IVec2 {
        // Non-finite points land in one far cell instead of poisoning the table.
        let c = (p.truncate() / self.cell).floor();
        if c.is_finite() {
            c.clamp(glam::Vec2::splat(-1e9), glam::Vec2::splat(1e9)).as_ivec2()
        } else {
            IVec2::splat(i32::MIN)
        }
    }

    fn bucket(&self, c: IVec2) -> usize {
        let h = (c.x as u32).wrapping_mul(0x8DA6_B343) ^ (c.y as u32).wrapping_mul(0xD816_3841);
        (h ^ h >> 15) as usize & self.mask as usize
    }

    pub fn rebuild(&mut self, positions: &[Vec3]) {
        let buckets = (positions.len() * 2).next_power_of_two().max(64);
        self.mask = buckets as u32 - 1;
        let mut cells = std::mem::take(&mut self.cells);
        cells.clear();
        cells.extend(positions.iter().map(|&p| self.cell_of(p)));
        self.cells = cells;
        self.start.clear();
        self.start.resize(buckets + 1, 0);
        for i in 0..positions.len() {
            let b = self.bucket(self.cells[i]);
            self.start[b + 1] += 1;
        }
        for b in 0..buckets {
            self.start[b + 1] += self.start[b];
        }
        self.items.clear();
        self.items.resize(positions.len(), 0);
        let mut fill = self.start.clone();
        for i in 0..positions.len() {
            let b = self.bucket(self.cells[i]);
            self.items[fill[b] as usize] = i as u32;
            fill[b] += 1;
        }
    }

    /// Calls `found` once for each point within `radius` of `p`, given the
    /// positions the hash was last rebuilt with.
    pub fn for_each_near(&self, positions: &[Vec3], p: Vec3, radius: f32, mut found: impl FnMut(u32)) {
        if self.items.is_empty() || !p.is_finite() || radius.is_nan() || radius < 0.0 {
            return;
        }
        let (lo, hi) = (self.cell_of(p - radius), self.cell_of(p + radius));
        // A huge radius visits every point once rather than every cell.
        if (hi - lo).max_element() > 64 {
            for (i, q) in positions.iter().enumerate() {
                if q.distance_squared(p) <= radius * radius {
                    found(i as u32);
                }
            }
            return;
        }
        for y in lo.y..=hi.y {
            for x in lo.x..=hi.x {
                let c = IVec2::new(x, y);
                let b = self.bucket(c);
                for &i in &self.items[self.start[b] as usize..self.start[b + 1] as usize] {
                    // Other cells share the bucket; each point counts in its own.
                    if self.cells[i as usize] == c
                        && positions[i as usize].distance_squared(p) <= radius * radius
                    {
                        found(i);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_exactly_the_brute_force_neighbours() {
        let positions: Vec<Vec3> = (0..3000u32)
            .map(|i| {
                let h = |k: u32| (i.wrapping_mul(2654435761).wrapping_add(k * 97)).rotate_left(k) as f32 / u32::MAX as f32;
                Vec3::new(h(1) * 200.0 - 100.0, h(2) * 200.0 - 100.0, h(3) * 4.0)
            })
            .collect();
        let mut hash = SpatialHash::new(3.0);
        hash.rebuild(&positions);
        for &(p, r) in &[(Vec3::ZERO, 5.0), (Vec3::new(40.0, -12.0, 1.0), 9.5), (Vec3::new(99.0, 99.0, 0.0), 2.0)] {
            let mut found = Vec::new();
            hash.for_each_near(&positions, p, r, |i| found.push(i));
            found.sort();
            let expected: Vec<u32> = (0..positions.len() as u32)
                .filter(|&i| positions[i as usize].distance(p) <= r)
                .collect();
            assert_eq!(found, expected);
        }
        hash.rebuild(&[]);
        hash.for_each_near(&[], Vec3::ZERO, 10.0, |_| panic!("empty"));
    }
}
