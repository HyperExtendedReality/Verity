//! What a city's edges become: lanes joined through junctions into a
//! [`LaneGraph`], signals on busy junctions, and sidewalks with corners and
//! crosswalks as a walk [`Graph`] whose crosswalks the signals gate.

use std::collections::HashMap;

use glam::Vec3;
use verity_agents::{LaneGraph, Signal};
use verity_nav::{spline, Graph, Path, NONE};

use crate::{hash, unit, City, Height, Kind};

/// Seconds of caution after green, and of all-red after that.
const CAUTION: f32 = 3.5;
const ALL_RED: f32 = 1.5;
/// Pedestrian graph nodes along a sidewalk are at most this far apart.
const WALK_SPACING: f32 = 8.0;
/// Height of the sidewalk above the road.
pub const KERB: f32 = 0.15;

/// An edge of the recipe as built.
#[derive(Clone, Debug)]
pub struct EdgeGeom {
    /// Its index in [`City::edges`].
    pub recipe: u32,
    pub kind: Kind,
    pub width: f32,
    pub lanes: [u8; 2],
    /// Its junctions, at `a` and `b`.
    pub ends: [u32; 2],
    pub centre: Path,
}

#[derive(Clone, Debug, Default)]
pub struct Junction {
    pub position: Vec3,
    /// How far lanes and sidewalks stop short of it.
    pub radius: f32,
    /// Drivable edges meeting here, counter-clockwise by the way they leave.
    pub edges: Vec<u32>,
    /// Gate ids of its approaches, in `edges` order, if it has signals.
    pub approaches: Vec<u32>,
}

#[derive(Clone, Debug, Default)]
pub struct Network {
    pub edges: Vec<EdgeGeom>,
    /// One per recipe node.
    pub junctions: Vec<Junction>,
    pub lanes: LaneGraph,
    /// The edge each lane runs along, or `NONE` through a junction.
    pub lane_edge: Vec<u32>,
    pub walks: Graph,
    pub sidewalk: f32,
    pub green: f32,
    /// Gates: the junction and edge of each signalled approach.
    pub gates: Vec<(u32, u32)>,
}

impl Network {
    pub fn build(city: &City, height: Height) -> Self {
        let rules = &city.rules;
        let sidewalk = rules.sidewalk.clamp(0.5, 20.0);
        let edges: Vec<EdgeGeom> = (0..city.edges.len())
            .filter(|&e| city.valid(&city.edges[e]))
            .map(|e| {
                let r = &city.edges[e];
                EdgeGeom {
                    recipe: e as u32,
                    kind: r.kind,
                    width: r.width,
                    lanes: r.lanes,
                    ends: [r.a, r.b],
                    centre: city.centre(e, height),
                }
            })
            .filter(|e| e.centre.length() > 1.0)
            .collect();
        let mut junctions: Vec<Junction> = (0..city.nodes.len() as u32)
            .map(|n| Junction {
                position: city.node(n),
                ..Junction::default()
            })
            .collect();
        for (i, e) in edges.iter().enumerate() {
            if e.kind.drives() {
                for &n in &e.ends {
                    junctions[n as usize].edges.push(i as u32);
                }
            }
        }
        let mut network = Self {
            sidewalk,
            green: rules.green.clamp(2.0, 120.0),
            ..Self::default()
        };
        for (n, junction) in junctions.iter_mut().enumerate() {
            let leaving = |e: u32| leaving(&edges[e as usize], n as u32);
            junction.edges.sort_by(|&a, &b| {
                let (a, b) = (leaving(a), leaving(b));
                a.y.atan2(a.x).total_cmp(&b.y.atan2(b.x))
            });
            // The centre sits on the drivable ends, which may be draped.
            if let Some(&e) = junction.edges.first() {
                let geom = &edges[e as usize];
                let end = if geom.ends[0] == n as u32 { 0.0 } else { geom.centre.length() };
                junction.position = geom.centre.at(end).0;
            }
            if junction.edges.len() >= 2 {
                junction.radius = (junction.edges.iter())
                    .map(|&e| {
                        let e = &edges[e as usize];
                        e.width * 0.5 + if e.kind == Kind::Road { sidewalk } else { 1.0 }
                    })
                    .fold(0.0, f32::max)
                    + 1.0;
            }
            if junction.edges.len() >= 3 {
                for &e in &junction.edges {
                    junction.approaches.push(network.gates.len() as u32);
                    network.gates.push((n as u32, e));
                }
            }
        }
        network.edges = edges;
        network.junctions = junctions;
        network.build_lanes();
        network.build_walks();
        network
    }

    /// How far edge `e` stops short of its junctions at `a` and `b`, scaled
    /// down to leave a short edge some length.
    pub fn trims(&self, e: usize) -> [f32; 2] {
        let edge = &self.edges[e];
        let [ra, rb] = edge.ends.map(|n| self.junctions[n as usize].radius);
        let room = (edge.centre.length() - 1.0).max(0.0);
        let scale = if ra + rb > room { room / (ra + rb).max(1e-3) } else { 1.0 };
        [ra * scale, rb * scale]
    }

    /// The gate of the approach to junction `n` along edge `e`, if signalled.
    fn approach(&self, n: u32, e: u32) -> u32 {
        let junction = &self.junctions[n as usize];
        (junction.edges.iter().position(|&x| x == e))
            .and_then(|i| junction.approaches.get(i).copied())
            .unwrap_or(NONE)
    }

    fn build_lanes(&mut self) {
        let mut lanes: Vec<(Path, f32, u32)> = Vec::new();
        let mut lane_edge = Vec::new();
        // Per junction: lanes arriving and leaving as (edge, lane index, lane id).
        let mut arriving: HashMap<u32, Vec<(u32, u32, u32)>> = HashMap::new();
        let mut departing: HashMap<u32, Vec<(u32, u32, u32)>> = HashMap::new();
        for (e, edge) in self.edges.iter().enumerate() {
            if !edge.kind.drives() {
                continue;
            }
            let [nf, nb] = edge.lanes.map(u32::from);
            let lane_width = edge.width / (nf + nb).max(1) as f32;
            let [ta, tb] = self.trims(e);
            for (dir, count) in [(0, nf), (1, nb)] {
                for k in 0..count {
                    // Forward lanes keep right of the divide, k = 0 nearest it.
                    let offset = if dir == 0 {
                        -edge.width * 0.5 + (nf - k) as f32 * lane_width - lane_width * 0.5
                    } else {
                        -edge.width * 0.5 + (nf + k) as f32 * lane_width + lane_width * 0.5
                    };
                    let mut path = edge.centre.offset(offset).trimmed(ta, tb);
                    if dir == 1 {
                        path = path.reversed();
                    }
                    if path.length() < 0.5 {
                        continue;
                    }
                    let (from, to) = if dir == 0 { (edge.ends[0], edge.ends[1]) } else { (edge.ends[1], edge.ends[0]) };
                    let id = lanes.len() as u32;
                    lanes.push((path, edge.kind.speed(), self.approach(to, e as u32)));
                    lane_edge.push(e as u32);
                    arriving.entry(to).or_default().push((e as u32, k, id));
                    departing.entry(from).or_default().push((e as u32, k, id));
                }
            }
        }
        let mut links = Vec::new();
        for (n, ins) in &arriving {
            let outs = departing.get(n).map(Vec::as_slice).unwrap_or(&[]);
            let dead_end = self.junctions[*n as usize].edges.len() <= 1;
            for &(ei, k, li) in ins {
                let mut targets: Vec<u32> = outs.iter().map(|o| o.0).filter(|&eo| eo != ei || dead_end).collect();
                targets.sort_unstable();
                targets.dedup();
                for eo in targets {
                    let mut choices: Vec<&(u32, u32, u32)> = outs.iter().filter(|o| o.0 == eo).collect();
                    choices.sort_by_key(|o| o.1);
                    let lo = choices[(k as usize).min(choices.len() - 1)].2;
                    let (end, arrive) = (lanes[li as usize].0.at(f32::MAX).0, lanes[li as usize].0.ends().1);
                    let (start, leave) = (lanes[lo as usize].0.at(0.0).0, lanes[lo as usize].0.ends().0);
                    let speed = lanes[li as usize].1.min(lanes[lo as usize].1) * 0.6;
                    let id = lanes.len() as u32;
                    lanes.push((Path::new(spline::bend(end, arrive, start, leave, 8)), speed.max(3.0), NONE));
                    lane_edge.push(NONE);
                    links.extend([(li, id), (id, lo)]);
                }
            }
        }
        links.sort_unstable();
        self.lanes = LaneGraph::new(lanes, links);
        self.lane_edge = lane_edge;
    }

    fn build_walks(&mut self) {
        let mut positions: Vec<Vec3> = Vec::new();
        let mut links: Vec<(u32, u32, f32, u32)> = Vec::new();
        let mut join = |positions: &[Vec3], a: u32, b: u32, gate: u32| {
            let d = positions[a as usize].distance(positions[b as usize]);
            links.extend([(a, b, d, gate), (b, a, d, gate)]);
        };
        // The end nodes of each road's sidewalks: [edge][side][end].
        let mut ends: HashMap<(u32, u32, u32), u32> = HashMap::new();
        for (e, edge) in self.edges.iter().enumerate() {
            if edge.kind != Kind::Road {
                continue;
            }
            let [ta, tb] = self.trims(e);
            let offset = edge.width * 0.5 + self.sidewalk * 0.5;
            for (side, sign) in [(0u32, 1.0), (1, -1.0)] {
                let path = edge.centre.offset(offset * sign).trimmed(ta, tb);
                let steps = (path.length() / WALK_SPACING).ceil().max(1.0) as usize;
                let first = positions.len() as u32;
                for k in 0..=steps {
                    positions.push(path.at(path.length() * k as f32 / steps as f32).0 + Vec3::Z * KERB);
                }
                for k in 0..steps as u32 {
                    join(&positions, first + k, first + k + 1, NONE);
                }
                ends.insert((e as u32, side, 0), first);
                ends.insert((e as u32, side, 1), first + steps as u32);
            }
        }
        for (n, junction) in self.junctions.iter().enumerate() {
            let n = n as u32;
            let roads: Vec<u32> = (junction.edges.iter().copied())
                .filter(|&e| self.edges[e as usize].kind == Kind::Road)
                .collect();
            let end_of = |e: u32| u32::from(self.edges[e as usize].ends[0] != n);
            // Crosswalks across each road's end, except where a road just bends.
            if junction.edges.len() != 2 {
                for &e in &roads {
                    let end = end_of(e);
                    let (Some(&l), Some(&r)) = (ends.get(&(e, 0, end)), ends.get(&(e, 1, end))) else {
                        continue;
                    };
                    let gate = self.approach(n, e);
                    join(&positions, l, r, gate);
                }
            }
            // Corners between roads next to each other around the junction.
            if roads.len() >= 2 {
                for i in 0..roads.len() {
                    let (e1, e2) = (roads[i], roads[(i + 1) % roads.len()]);
                    // The sidewalk left of e1 as it leaves, and right of e2.
                    let s1 = end_of(e1);
                    let s2 = 1 - end_of(e2);
                    let (Some(&a), Some(&b)) = (ends.get(&(e1, s1, end_of(e1))), ends.get(&(e2, s2, end_of(e2)))) else {
                        continue;
                    };
                    let (pa, pb) = (positions[a as usize], positions[b as usize]);
                    let (da, db) = (-leaving(&self.edges[e1 as usize], n), leaving(&self.edges[e2 as usize], n));
                    let curve = spline::bend(pa, da, pb, db, 4);
                    let mut previous = a;
                    for &p in &curve[1..curve.len() - 1] {
                        positions.push(p);
                        let id = positions.len() as u32 - 1;
                        join(&positions, previous, id, NONE);
                        previous = id;
                    }
                    join(&positions, previous, b, NONE);
                }
            }
        }
        self.walks = Graph::new(positions, links);
    }

    /// Signals at time `t`: what each gate tells vehicles, and whether
    /// pedestrians may cross at it (while its vehicles are held).
    pub fn signals(&self, t: f32, cars: &mut Vec<Signal>, walk: &mut Vec<bool>) {
        cars.clear();
        walk.clear();
        let slot = self.green + CAUTION + ALL_RED;
        for &(n, e) in &self.gates {
            let junction = &self.junctions[n as usize];
            let count = junction.approaches.len().max(1) as f32;
            // Junctions start their cycles at different times.
            let phase = (t + unit(hash(n, 77)) * slot * count).rem_euclid(slot * count);
            let current = (phase / slot) as usize;
            let local = phase - current as f32 * slot;
            let mine = junction.edges.iter().position(|&x| x == e) == Some(current);
            let signal = match (mine, local) {
                (true, l) if l < self.green => Signal::Open,
                (true, l) if l < self.green + CAUTION => Signal::Caution,
                _ => Signal::Closed,
            };
            cars.push(signal);
            walk.push(signal == Signal::Closed);
        }
    }
}

/// Unit heading of edge `e` as it leaves junction `n`.
pub fn leaving(edge: &EdgeGeom, n: u32) -> Vec3 {
    let (start, end) = edge.centre.ends();
    if edge.ends[0] == n { start } else { -end }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{flat, tests::grid};

    #[test]
    fn every_lane_leads_somewhere_and_junctions_are_signalled() {
        let city = grid(4, 80.0);
        let net = Network::build(&city, &flat);
        assert_eq!(net.edges.len(), 24);
        for lane in 0..net.lanes.len() as u32 {
            assert!(!net.lanes.next.get(lane).is_empty(), "lane {lane} is a dead end");
        }
        // Inner junctions have four approaches, edge middles three.
        let signalled = net.junctions.iter().filter(|j| !j.approaches.is_empty()).count();
        assert_eq!(signalled, 4 + 8);
        let (mut cars, mut walk) = (Vec::new(), Vec::new());
        for t in 0..200 {
            net.signals(t as f32 * 0.5, &mut cars, &mut walk);
            assert_eq!(cars.len(), net.gates.len());
            for junction in &net.junctions {
                let open = junction.approaches.iter().filter(|&&g| cars[g as usize] != Signal::Closed).count();
                assert!(open <= 1, "two approaches open at once");
            }
        }
    }

    #[test]
    fn sidewalks_connect_around_the_town() {
        let city = grid(3, 80.0);
        let net = Network::build(&city, &flat);
        let walks = &net.walks;
        assert!(walks.len() > 50);
        // Every node reaches every other.
        let reach = walks.distances_to(&[0]);
        assert!(reach.iter().all(|d| d.is_finite()), "the walk graph is split");
        assert!(walks.gates.iter().any(|&g| g != NONE), "no crosswalk is signalled");
    }
}
