//! City blocks are the faces of the planar road graph; lots are rectangles
//! marched along each block's street frontage.

use glam::{Vec2, Vec3};
use verity_nav::{spline, Path};

use crate::{hash, network::Network, City, Kind};

#[derive(Clone, Debug, PartialEq)]
pub struct Block {
    /// Counter-clockwise, inside the sidewalks.
    pub outline: Vec<Vec2>,
    /// The edges around it, each with whether it runs `a` to `b` that way.
    pub sides: Vec<(u32, bool)>,
}

/// A rectangle facing a street.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Lot {
    /// Centre, at the height of its frontage.
    pub centre: Vec3,
    /// Unit direction along the street.
    pub front: Vec2,
    /// Half its frontage and half its depth.
    pub half: Vec2,
    pub block: u32,
    pub seed: u32,
}

impl Lot {
    /// Unit direction from the street into the lot.
    pub fn inward(&self) -> Vec2 {
        self.front.perp()
    }

    pub fn corners(&self) -> [Vec2; 4] {
        let (c, x, y) = (self.centre.truncate(), self.front * self.half.x, self.inward() * self.half.y);
        [c - x - y, c + x - y, c + x + y, c - x + y]
    }
}

fn area(outline: &[Vec2]) -> f32 {
    (0..outline.len())
        .map(|i| outline[i].perp_dot(outline[(i + 1) % outline.len()]))
        .sum::<f32>()
        * 0.5
}

fn inside(outline: &[Vec2], p: Vec2) -> bool {
    let mut odd = false;
    for i in 0..outline.len() {
        let (a, b) = (outline[i], outline[(i + 1) % outline.len()]);
        if (a.y > p.y) != (b.y > p.y) && p.x < a.x + (p.y - a.y) / (b.y - a.y) * (b.x - a.x) {
            odd = !odd;
        }
    }
    odd
}

/// Whether two convex polygons overlap, by separating axes.
fn overlap(a: &[Vec2; 4], b: &[Vec2; 4]) -> bool {
    for poly in [a, b] {
        for i in 0..4 {
            let axis = (poly[(i + 1) % 4] - poly[i]).perp();
            let span = |p: &[Vec2; 4]| {
                p.iter().map(|v| v.dot(axis)).fold((f32::MAX, f32::MIN), |(lo, hi), d| (lo.min(d), hi.max(d)))
            };
            let ((alo, ahi), (blo, bhi)) = (span(a), span(b));
            if ahi <= blo + 1e-3 || bhi <= alo + 1e-3 {
                return false;
            }
        }
    }
    true
}

/// How far into a block from an edge's centre line the lots start.
fn kerb_line(network: &Network, edge: u32) -> f32 {
    let e = &network.edges[edge as usize];
    e.width * 0.5 + if e.kind == Kind::Road { network.sidewalk } else { 1.0 }
}

/// An edge's centre line in a direction, offset to its left by `by`.
fn side(network: &Network, edge: u32, forward: bool, by: f32) -> Path {
    let centre = &network.edges[edge as usize].centre;
    let path = if forward { centre.clone() } else { centre.reversed() };
    path.offset(by)
}

/// Faces of the drivable edges with positive area.
pub fn blocks(network: &Network) -> Vec<Block> {
    let n = network.edges.len();
    // Half-edge 2e runs a to b, 2e + 1 back.
    let dest = |h: usize| network.edges[h / 2].ends[1 - h % 2];
    let next = |h: usize| -> usize {
        let at = dest(h);
        let around = &network.junctions[at as usize].edges;
        let i = around.iter().position(|&e| e as usize == h / 2).unwrap_or(0);
        let e = around[(i + around.len() - 1) % around.len()] as usize;
        e * 2 + usize::from(network.edges[e].ends[0] != at)
    };
    let mut seen = vec![false; n * 2];
    let mut found = Vec::new();
    for start in 0..n * 2 {
        if seen[start] || !network.edges[start / 2].kind.drives() {
            continue;
        }
        let mut cycle = Vec::new();
        let mut h = start;
        while !seen[h] && cycle.len() <= n * 2 {
            seen[h] = true;
            cycle.push(h);
            h = next(h);
        }
        if h != start {
            continue;
        }
        let mut outline = Vec::new();
        for &h in &cycle {
            let e = (h / 2) as u32;
            let path = side(network, e, h % 2 == 0, kerb_line(network, e));
            outline.extend(path.points[..path.points.len() - 1].iter().map(|p| p.truncate()));
        }
        if outline.len() >= 3 && area(&outline) > 1.0 {
            let sides = cycle.iter().map(|&h| ((h / 2) as u32, h % 2 == 0)).collect();
            found.push(Block { outline, sides });
        }
    }
    found
}

/// Lots along the roads around each block, clear of rivers and of each other.
pub fn lots(city: &City, network: &Network, blocks: &[Block]) -> Vec<Lot> {
    let rules = &city.rules;
    let (width, depth) = (rules.lot_width.clamp(4.0, 200.0), rules.lot_depth.clamp(4.0, 200.0));
    let rivers: Vec<(&Path, f32)> = (network.edges.iter())
        .filter(|e| e.kind == Kind::River)
        .map(|e| (&e.centre, e.width * 0.5 + 2.0))
        .collect();
    let mut all = Vec::new();
    for (b, block) in blocks.iter().enumerate() {
        let mut placed: Vec<[Vec2; 4]> = Vec::new();
        for &(e, forward) in &block.sides {
            if network.edges[e as usize].kind != Kind::Road {
                continue;
            }
            let [ta, tb] = network.trims(e as usize);
            let (start, end) = if forward { (ta, tb) } else { (tb, ta) };
            let frontage = side(network, e, forward, kerb_line(network, e)).trimmed(start, end);
            let count = (frontage.length() / width).floor() as usize;
            let spare = frontage.length() - count as f32 * width;
            for k in 0..count {
                let s = spare * 0.5 + (k as f32 + 0.5) * width;
                let (p, heading) = frontage.at(s);
                let front = heading.truncate().normalize_or_zero();
                let lot = Lot {
                    // Just behind the kerb line, which bounds the block.
                    centre: p + (spline::left(heading) * (depth * 0.5 + 0.25)),
                    front,
                    half: Vec2::new(width * 0.5, depth * 0.5),
                    block: b as u32,
                    seed: 0,
                };
                let corners = lot.corners();
                let fits = corners.iter().all(|&c| inside(&block.outline, c))
                    && !placed.iter().any(|other| overlap(other, &corners))
                    && rivers.iter().all(|(river, clear)| {
                        corners.iter().chain([&lot.centre.truncate()]).all(|c| river.nearest(c.extend(0.0)).1 > *clear)
                    });
                if fits {
                    placed.push(corners);
                    all.push(lot);
                }
            }
        }
    }
    for (i, lot) in all.iter_mut().enumerate() {
        lot.seed = hash(city.seed, i as u32);
    }
    all
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{flat, tests::grid, Edge, Model};

    #[test]
    fn a_four_by_four_grid_has_nine_blocks_full_of_lots() {
        let city = grid(4, 80.0);
        let model = Model::build(&city, &flat);
        assert_eq!(model.blocks.len(), 9);
        for block in &model.blocks {
            assert_eq!(block.sides.len(), 4);
            assert!(area(&block.outline) > 60.0 * 60.0 * 0.9);
        }
        // Frontage room for two lots per side, four sides, nine blocks.
        assert!(model.buildings.len() >= 9 * 4 * 2, "{}", model.buildings.len());
        for pair in model.buildings.windows(2).filter(|p| p[0].lot.block == p[1].lot.block) {
            assert!(!overlap(&pair[0].lot.corners(), &pair[1].lot.corners()));
        }
    }

    #[test]
    fn the_same_seed_builds_the_same_town_and_rivers_keep_clear() {
        let mut city = grid(3, 100.0);
        let (a, b) = (city.add_node(Vec3::new(-50.0, -150.0, 0.0)), city.add_node(Vec3::new(-50.0, 150.0, 0.0)));
        city.edges.push(Edge { a, b, controls: Vec::new(), kind: Kind::River, lanes: [0, 0], width: 12.0 });
        let one = Model::build(&city, &flat);
        let two = Model::build(&city, &flat);
        assert_eq!(one.buildings, two.buildings);
        assert!(!one.buildings.is_empty());
        for building in &one.buildings {
            for c in building.lot.corners() {
                assert!((c.x + 50.0).abs() > 8.0, "a lot stands in the river at {c}");
            }
        }
        city.seed = 9;
        assert_ne!(Model::build(&city, &flat).buildings, one.buildings);
    }
}
