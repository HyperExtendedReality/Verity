//! Free-moving agents. Behaviour belongs to groups, so ten thousand zombies
//! share one goal, one speed and one flow field.
//!
//! Each step reads only the previous state while it decides, in parallel,
//! so results do not depend on thread count or scheduling.

use std::sync::Arc;

use glam::{Vec2, Vec3};
use rayon::prelude::*;
use verity_nav::{FlowField, Graph, NavMesh, SpatialHash, NONE};

use crate::{hash, unit};

/// Distance an agent walks per animation cycle, in metres.
const STRIDE: f32 = 1.4;
/// Far agents decide once per this many steps.
const FAR_EVERY: u64 = 4;
/// Seconds ahead agents look for collisions.
const HORIZON: f32 = 3.0;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Goal {
    Stand,
    /// Roam, turning to a new heading every few seconds.
    Wander,
    /// Walk to each agent's target along a route: over the navmesh if there
    /// is one, else along the graph, else straight.
    Seek,
    /// Follow flow field `n` of [`Crowd::fields`] toward its goals.
    Flow(u32),
    /// Run from a point while within `radius` of it.
    Flee { from: Vec3, radius: f32 },
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Group {
    pub goal: Goal,
    /// Preferred speed, metres per second.
    pub speed: f32,
    /// Most change of velocity per second.
    pub acceleration: f32,
    /// How hard agents keep apart and avoid collisions.
    pub separation: f32,
    /// Pull toward, and matching the heading of, nearby agents of this group.
    pub cohesion: f32,
    pub alignment: f32,
    /// How far agents see others, in metres.
    pub sight: f32,
    /// Half-width agents keep within along graph routes (a sidewalk); 0 for none.
    pub corridor: f32,
}

impl Default for Group {
    fn default() -> Self {
        Self {
            goal: Goal::Stand,
            speed: 1.4,
            acceleration: 4.0,
            separation: 1.0,
            cohesion: 0.0,
            alignment: 0.0,
            sight: 3.0,
            corridor: 0.0,
        }
    }
}

/// Height of the ground under a point of the plane.
pub type GroundFn = dyn Fn(Vec2) -> Option<f32> + Send + Sync;

/// What one agent decided this step.
#[derive(Clone, Copy)]
struct Decision {
    velocity: Vec3,
    route: [u32; 3],
    arrived: bool,
}

pub struct Crowd {
    pub position: Vec<Vec3>,
    pub velocity: Vec<Vec3>,
    pub radius: Vec<f32>,
    pub group: Vec<u16>,
    /// Where [`Goal::Seek`] agents are going.
    pub target: Vec<Vec3>,
    /// Walk cycle, advancing by one per stride walked.
    pub phase: Vec<f32>,
    /// Start, next and end of each agent's waypoints in `points`.
    route: Vec<[u32; 3]>,
    needs_route: Vec<bool>,
    points: Vec<Vec3>,
    /// The gate passed to reach each point, or `NONE`.
    point_gates: Vec<u32>,
    pub groups: Vec<Group>,
    pub fields: Vec<FlowField>,
    navmesh: Option<Arc<NavMesh>>,
    graph: Option<(Arc<Graph>, SpatialHash)>,
    /// Height of the ground, when there is no navmesh.
    pub ground: Option<Arc<GroundFn>>,
    /// Whether each gate of the graph is open.
    pub gates: Vec<bool>,
    /// Points agents are watched from (cameras, players); far agents decide
    /// less often. Empty: every agent is near.
    pub focus: Vec<Vec3>,
    pub near: f32,
    /// Routes planned per step at most.
    pub plans_per_step: usize,
    /// Seek agents that reached their target in the last step.
    pub arrived: Vec<u32>,
    pub time: f32,
    tick: u64,
    hash: SpatialHash,
}

impl Default for Crowd {
    fn default() -> Self {
        Self::new()
    }
}

impl Crowd {
    pub fn new() -> Self {
        Self {
            position: Vec::new(),
            velocity: Vec::new(),
            radius: Vec::new(),
            group: Vec::new(),
            target: Vec::new(),
            phase: Vec::new(),
            route: Vec::new(),
            needs_route: Vec::new(),
            points: Vec::new(),
            point_gates: Vec::new(),
            groups: Vec::new(),
            fields: Vec::new(),
            navmesh: None,
            graph: None,
            ground: None,
            gates: Vec::new(),
            focus: Vec::new(),
            near: 60.0,
            plans_per_step: 64,
            arrived: Vec::new(),
            time: 0.0,
            tick: 0,
            hash: SpatialHash::new(2.0),
        }
    }

    pub fn len(&self) -> usize {
        self.position.len()
    }

    pub fn is_empty(&self) -> bool {
        self.position.is_empty()
    }

    pub fn add_group(&mut self, group: Group) -> u16 {
        self.groups.push(group);
        self.groups.len() as u16 - 1
    }

    pub fn spawn(&mut self, group: u16, position: Vec3, radius: f32) -> u32 {
        self.position.push(position);
        self.velocity.push(Vec3::ZERO);
        self.radius.push(radius.max(0.01));
        self.group.push(group);
        self.target.push(position);
        self.phase.push(unit(hash(self.position.len() as u32, 0x5eed)));
        self.route.push([0; 3]);
        self.needs_route.push(false);
        self.position.len() as u32 - 1
    }

    /// Remove agent `i`; the last agent takes its index.
    pub fn remove(&mut self, i: u32) {
        let i = i as usize;
        if i >= self.len() {
            return;
        }
        self.position.swap_remove(i);
        self.velocity.swap_remove(i);
        self.radius.swap_remove(i);
        self.group.swap_remove(i);
        self.target.swap_remove(i);
        self.phase.swap_remove(i);
        self.route.swap_remove(i);
        self.needs_route.swap_remove(i);
    }

    /// Send a [`Goal::Seek`] agent somewhere new; its route is planned soon.
    pub fn set_target(&mut self, i: u32, target: Vec3) {
        if let Some(t) = self.target.get_mut(i as usize) {
            *t = target;
            self.route[i as usize] = [0; 3];
            self.needs_route[i as usize] = true;
        }
    }

    /// The waypoints agent `i` has yet to reach.
    pub fn route(&self, i: u32) -> &[Vec3] {
        let [_, at, end] = self.route[i as usize];
        &self.points[at as usize..end as usize]
    }

    pub fn navmesh(&self) -> Option<&Arc<NavMesh>> {
        self.navmesh.as_ref()
    }

    /// Walk on `mesh`: agents stay on it and route over it.
    pub fn set_navmesh(&mut self, mesh: Option<Arc<NavMesh>>) {
        self.navmesh = mesh;
        self.replan_all();
    }

    /// Route over `graph` when there is no navmesh, as along sidewalks.
    pub fn set_graph(&mut self, graph: Option<Arc<Graph>>) {
        self.graph = graph.map(|g| {
            let mut index = SpatialHash::new(8.0);
            index.rebuild(&g.positions);
            (g, index)
        });
        self.replan_all();
    }

    pub fn graph(&self) -> Option<&Arc<Graph>> {
        self.graph.as_ref().map(|(g, _)| g)
    }

    fn replan_all(&mut self) {
        for i in 0..self.len() {
            if self.route[i][2] > 0 || self.needs_route[i] {
                self.route[i] = [0; 3];
                self.needs_route[i] = true;
            }
        }
    }

    /// Advance every agent by `dt` seconds.
    pub fn step(&mut self, dt: f32) {
        if dt.is_nan() || dt <= 0.0 || self.is_empty() {
            return;
        }
        self.hash.rebuild(&self.position);
        let decisions: Vec<Decision> = (0..self.len())
            .into_par_iter()
            .map(|i| {
                if self.thinks(i) {
                    self.decide(i, dt)
                } else {
                    Decision {
                        velocity: self.velocity[i],
                        route: self.route[i],
                        arrived: false,
                    }
                }
            })
            .collect();
        self.arrived.clear();
        for (i, d) in decisions.iter().enumerate() {
            self.velocity[i] = d.velocity;
            self.route[i] = d.route;
            if d.arrived {
                self.arrived.push(i as u32);
            }
        }
        let moved: Vec<Vec3> = (0..self.len())
            .into_par_iter()
            .map(|i| self.settle(i, self.position[i] + self.velocity[i] * dt))
            .collect();
        // Push overlapping agents apart, each by its share, from where all moved to.
        self.hash.rebuild(&moved);
        let relaxed: Vec<Vec3> = (0..self.len())
            .into_par_iter()
            .map(|i| self.settle(i, moved[i] + self.overlap(i, &moved)))
            .collect();
        for (i, p) in relaxed.into_iter().enumerate() {
            let walked = (p - self.position[i]).truncate().length();
            self.phase[i] = (self.phase[i] + walked / STRIDE).fract();
            self.position[i] = p;
        }
        self.plan();
        self.compact();
        self.time += dt;
        self.tick += 1;
    }

    fn thinks(&self, i: usize) -> bool {
        let near = self.focus.is_empty()
            || (self.focus.iter()).any(|f| f.distance_squared(self.position[i]) <= self.near * self.near);
        near || (i as u64 + self.tick).is_multiple_of(FAR_EVERY)
    }

    fn decide(&self, i: usize, dt: f32) -> Decision {
        let group = self.groups.get(self.group[i] as usize).copied().unwrap_or_default();
        let (p, v, r) = (self.position[i], self.velocity[i], self.radius[i]);
        let mut route = self.route[i];
        let mut arrived = false;
        let arrive = |to: Vec3, slow: f32| -> Vec2 {
            let d = (to - p).truncate();
            d.normalize_or_zero() * group.speed * (d.length() / slow).min(1.0)
        };
        let desired = match group.goal {
            Goal::Stand => Vec2::ZERO,
            Goal::Wander => {
                // A new heading every three seconds or so, staggered per agent.
                let slot = (self.time / 3.0 + unit(hash(i as u32, 1)) * 3.0) as u32;
                let angle = unit(hash(i as u32, slot)) * std::f32::consts::TAU;
                Vec2::from_angle(angle) * group.speed * 0.5
            }
            Goal::Seek if self.needs_route[i] => Vec2::ZERO,
            Goal::Seek => {
                let [_, at, end] = route;
                if at >= end {
                    Vec2::ZERO
                } else {
                    let w = self.points[at as usize];
                    let reach = (r * 2.0).max(0.6);
                    let close = (w - p).truncate().length() <= reach;
                    if close && at + 1 >= end {
                        route[1] = end;
                        arrived = true;
                        Vec2::ZERO
                    } else if close {
                        let gate = self.point_gates[at as usize + 1];
                        if gate == NONE || self.gates.get(gate as usize).copied().unwrap_or(true) {
                            route[1] += 1;
                            arrive(self.points[at as usize + 1], 0.5)
                        } else {
                            // Wait at the gate.
                            arrive(w, 1.0)
                        }
                    } else if at + 1 >= end {
                        arrive(w, 1.5)
                    } else {
                        arrive(w, 0.1)
                    }
                }
            }
            Goal::Flow(n) => match self.fields.get(n as usize) {
                Some(field) if field.distance(p) > r * 2.0 => field.direction(p) * group.speed,
                _ => Vec2::ZERO,
            },
            Goal::Flee { from, radius } => {
                let away = (p - from).truncate();
                if away.length() < radius {
                    away.normalize_or(Vec2::from_angle(unit(hash(i as u32, 2)) * 6.3)) * group.speed
                } else {
                    Vec2::ZERO
                }
            }
        };
        let mut push = Vec2::ZERO;
        let (mut centre, mut heading, mut flockmates) = (Vec2::ZERO, Vec2::ZERO, 0u32);
        self.hash.for_each_near(&self.position, p, group.sight, |j| {
            let j = j as usize;
            if j == i {
                return;
            }
            let (q, w, rj) = (self.position[j], self.velocity[j], self.radius[j]);
            let x = (q - p).truncate();
            let d = x.length();
            let room = r + rj + 0.3;
            if d < room {
                let away = if d > 1e-5 { -x / d } else { Vec2::from_angle(unit(hash(i as u32, j as u32)) * 6.3) };
                push += away * (room - d) / room * group.separation * group.speed * 3.0;
            } else {
                // Time to collision: steer from where the two would touch.
                let rel = (v - w).truncate();
                let (a, b, c) = (rel.dot(rel), x.dot(rel), d * d - (r + rj) * (r + rj));
                let disc = b * b - a * c;
                if a > 1e-6 && b > 0.0 && disc > 0.0 {
                    let tau = (b - disc.sqrt()) / a;
                    if tau < HORIZON {
                        let apart = ((p + v * tau) - (q + w * tau)).truncate().normalize_or(-x / d);
                        push += apart * (HORIZON - tau) / (tau + 0.25) * group.separation * 0.5;
                    }
                }
            }
            if self.group[j] == self.group[i] {
                centre += q.truncate();
                heading += w.truncate();
                flockmates += 1;
            }
        });
        if flockmates > 0 {
            let n = flockmates as f32;
            push += (centre / n - p.truncate()) * group.cohesion;
            push += (heading / n - v.truncate()) * group.alignment;
        }
        let want = desired + push;
        let change = ((want - v.truncate()) * 4.0).clamp_length_max(group.acceleration) * dt;
        let velocity = (v.truncate() + change).clamp_length_max(group.speed * 1.5);
        Decision {
            velocity: velocity.extend(0.0),
            route,
            arrived,
        }
    }

    /// Movement apart from overlapping neighbours, half the overlap each.
    fn overlap(&self, i: usize, at: &[Vec3]) -> Vec3 {
        let (p, r) = (at[i], self.radius[i]);
        let mut shift = Vec2::ZERO;
        self.hash.for_each_near(at, p, r * 2.0 + 1.0, |j| {
            let j = j as usize;
            let x = (p - at[j]).truncate();
            let (d, room) = (x.length(), r + self.radius[j]);
            if j != i && d < room {
                let away = if d > 1e-5 { x / d } else { Vec2::from_angle(unit(hash(i.min(j) as u32, i.max(j) as u32)) * 6.3) * if i < j { 1.0 } else { -1.0 } };
                shift += away * (room - d) * 0.5;
            }
        });
        shift.clamp_length_max(r).extend(0.0)
    }

    /// Keep a moved agent on walkable ground: on the navmesh, within its
    /// graph corridor, and at the ground's height.
    fn settle(&self, i: usize, mut p: Vec3) -> Vec3 {
        let old = self.position[i];
        if !p.is_finite() {
            return old;
        }
        let group = self.groups.get(self.group[i] as usize).copied().unwrap_or_default();
        let [start, at, end] = self.route[i];
        // The segment being walked; after arriving, the last one.
        let to = at.min(end.saturating_sub(1)) as usize;
        if group.corridor > 0.0 && to > start as usize && self.navmesh.is_none() {
            let (a3, b3) = (self.points[to - 1], self.points[to]);
            let (a, ab) = (a3.truncate(), (b3 - a3).truncate());
            let t = ((p.truncate() - a).dot(ab) / ab.length_squared().max(1e-6)).clamp(0.0, 1.0);
            let on = a + ab * t;
            let side = (p.truncate() - on).clamp_length_max(group.corridor);
            p = (on + side).extend(a3.z + (b3.z - a3.z) * t);
        }
        if let Some(mesh) = &self.navmesh {
            return match mesh.nearest(p.truncate().extend(old.z), self.radius[i] * 4.0 + 1.0) {
                Some((_, on)) => on,
                None => old,
            };
        }
        if let Some(ground) = &self.ground
            && let Some(z) = ground(p.truncate())
        {
            p.z = z;
        }
        p
    }

    fn plan(&mut self) {
        let mut budget = self.plans_per_step;
        for i in 0..self.len() {
            if budget == 0 {
                break;
            }
            if !self.needs_route[i] {
                continue;
            }
            budget -= 1;
            self.needs_route[i] = false;
            let (from, to) = (self.position[i], self.target[i]);
            let waypoints: Vec<(Vec3, u32)> = if let Some(mesh) = &self.navmesh {
                match mesh.find_path(from, to, 5.0) {
                    Some(path) => path.into_iter().skip(1).map(|p| (p, NONE)).collect(),
                    None => Vec::new(),
                }
            } else if let Some((graph, index)) = &self.graph {
                let nearest = |p: Vec3| {
                    let mut best = (NONE, f32::INFINITY);
                    for reach in [10.0, 40.0, 160.0, 640.0] {
                        index.for_each_near(&graph.positions, p, reach, |n| {
                            let d = graph.positions[n as usize].distance_squared(p);
                            if d < best.1 {
                                best = (n, d);
                            }
                        });
                        if best.0 != NONE {
                            break;
                        }
                    }
                    best.0
                };
                let (a, b) = (nearest(from), nearest(to));
                match (a != NONE && b != NONE).then(|| graph.astar(a, b)).flatten() {
                    Some(nodes) => {
                        let gate = |pair: &[u32]| graph.edge(pair[0], pair[1]).map_or(NONE, |e| graph.gates[e as usize]);
                        std::iter::once((graph.positions[a as usize], NONE))
                            .chain(nodes.windows(2).map(|pair| (graph.positions[pair[1] as usize], gate(pair))))
                            .collect()
                    }
                    None => Vec::new(),
                }
            } else {
                vec![(to, NONE)]
            };
            let start = self.points.len() as u32;
            for (p, gate) in waypoints {
                self.points.push(p);
                self.point_gates.push(gate);
            }
            let end = self.points.len() as u32;
            self.route[i] = [start, start, end];
            if start == end {
                // Unreachable: report arrival so the owner can pick again.
                self.arrived.push(i as u32);
            }
        }
    }

    /// Drop waypoints no route uses once they outnumber the live ones.
    fn compact(&mut self) {
        let live: usize = self.route.iter().map(|r| (r[2] - r[0]) as usize).sum();
        if self.points.len() < 4096 || self.points.len() < live * 4 {
            return;
        }
        let (mut points, mut gates) = (Vec::with_capacity(live * 2), Vec::with_capacity(live * 2));
        for r in &mut self.route {
            let start = points.len() as u32;
            points.extend_from_slice(&self.points[r[0] as usize..r[2] as usize]);
            gates.extend_from_slice(&self.point_gates[r[0] as usize..r[2] as usize]);
            *r = [start, start + (r[1] - r[0]), start + (r[2] - r[0])];
        }
        self.points = points;
        self.point_gates = gates;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn floor(size: u32, open: impl Fn(Vec2) -> bool) -> Arc<NavMesh> {
        Arc::new(NavMesh::from_heights(Vec2::ZERO, 1.0, [size, size], 0.5, |p| open(p).then_some(0.0)))
    }

    #[test]
    fn crossing_streams_pass_without_overlapping() {
        let mut crowd = Crowd::new();
        crowd.set_navmesh(Some(floor(60, |_| true)));
        let seek = crowd.add_group(Group { goal: Goal::Seek, ..Group::default() });
        for k in 0..60 {
            let (row, col) = ((k / 6) as f32, (k % 6) as f32);
            let a = crowd.spawn(seek, Vec3::new(5.0 + col, 15.0 + row * 1.5, 0.0), 0.3);
            crowd.set_target(a, Vec3::new(55.0 - col, 15.0 + row * 1.5, 0.0));
            let b = crowd.spawn(seek, Vec3::new(55.0 - col, 15.5 + row * 1.5, 0.0), 0.3);
            crowd.set_target(b, Vec3::new(5.0 + col, 15.5 + row * 1.5, 0.0));
        }
        let mut done = 0;
        for _ in 0..60 * 90 {
            crowd.step(1.0 / 60.0);
            done += crowd.arrived.len();
            for i in 0..crowd.len() {
                for j in i + 1..crowd.len() {
                    let d = crowd.position[i].distance(crowd.position[j]);
                    assert!(d >= 0.8 * 0.6, "agents {i} and {j} overlap: {d}");
                }
            }
        }
        assert!(done >= crowd.len() * 9 / 10, "only {done} of {} arrived", crowd.len());
    }

    #[test]
    fn a_horde_pours_through_a_gap() {
        // A wall across x = 30 with a 4 m gap.
        let mesh = floor(60, |p| !((29.0..=31.0).contains(&p.x) && !(28.0..=32.0).contains(&p.y)));
        let mut crowd = Crowd::new();
        let mut field = FlowField::from_navmesh(&mesh, 1.0);
        let goal = Vec3::new(50.0, 50.0, 0.0);
        field.solve(&[goal]);
        crowd.fields.push(field);
        crowd.set_navmesh(Some(mesh));
        let horde = crowd.add_group(Group { goal: Goal::Flow(0), speed: 2.0, ..Group::default() });
        for k in 0..200 {
            crowd.spawn(horde, Vec3::new(3.0 + (k % 20) as f32 * 1.0, 3.0 + (k / 20) as f32 * 1.0, 0.0), 0.3);
        }
        for _ in 0..60 * 120 {
            crowd.step(1.0 / 60.0);
        }
        let there = crowd.position.iter().filter(|p| p.x > 32.0 && p.distance(goal) < 15.0).count();
        assert!(there >= 180, "{there} of 200 got through");
        assert!(crowd.position.iter().all(|p| crowd.navmesh().unwrap().locate(*p).is_some()));
    }

    #[test]
    fn a_flock_stays_together_and_runs_from_danger() {
        let mut crowd = Crowd::new();
        let flock = crowd.add_group(Group { goal: Goal::Wander, cohesion: 0.6, alignment: 0.6, sight: 8.0, speed: 2.0, ..Group::default() });
        for k in 0..40 {
            crowd.spawn(flock, Vec3::new((k % 8) as f32 * 1.5, (k / 8) as f32 * 1.5, 0.0), 0.4);
        }
        for _ in 0..60 * 30 {
            crowd.step(1.0 / 60.0);
        }
        let centre = crowd.position.iter().sum::<Vec3>() / crowd.len() as f32;
        let spread = crowd.position.iter().map(|p| p.distance(centre)).fold(0.0, f32::max);
        assert!(spread < 25.0, "flock spread to {spread}");
        crowd.groups[flock as usize].goal = Goal::Flee { from: centre, radius: 40.0 };
        crowd.groups[flock as usize].cohesion = 0.0;
        for _ in 0..60 * 10 {
            crowd.step(1.0 / 60.0);
        }
        let mean = crowd.position.iter().map(|p| p.distance(centre)).sum::<f32>() / crowd.len() as f32;
        assert!(mean > 15.0, "fled only {mean}");
    }

    #[test]
    fn graph_routes_wait_at_closed_gates_and_stay_in_their_corridor() {
        // A path 0 -> 1 -> 2 along x, gate 0 between 1 and 2.
        let positions = vec![Vec3::ZERO, Vec3::new(20.0, 0.0, 0.0), Vec3::new(40.0, 0.0, 0.0)];
        let graph = Graph::new(positions, [(0, 1, 20.0, NONE), (1, 2, 20.0, 0), (1, 0, 20.0, NONE), (2, 1, 20.0, 0)]);
        let mut crowd = Crowd::new();
        crowd.set_graph(Some(Arc::new(graph)));
        crowd.gates = vec![false];
        let walkers = crowd.add_group(Group { goal: Goal::Seek, corridor: 1.0, ..Group::default() });
        for k in 0..10 {
            let a = crowd.spawn(walkers, Vec3::new(-2.0, k as f32 * 0.2 - 1.0, 0.0), 0.25);
            crowd.set_target(a, Vec3::new(40.0, 0.0, 0.0));
        }
        for _ in 0..60 * 30 {
            crowd.step(1.0 / 60.0);
            for (i, p) in crowd.position.iter().enumerate() {
                assert!(p.x < 21.5, "agent {i} passed the closed gate at {p}");
            }
        }
        crowd.gates[0] = true;
        let mut arrived = 0;
        for _ in 0..60 * 40 {
            crowd.step(1.0 / 60.0);
            arrived += crowd.arrived.len();
            for p in &crowd.position {
                if p.x > 1.0 {
                    assert!(p.y.abs() <= 1.0 + 1e-3, "left the corridor at {p}");
                }
            }
        }
        assert_eq!(arrived, 10);
    }

    #[test]
    fn stepping_is_deterministic_and_thread_independent() {
        let run = |threads: usize| {
            rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap().install(|| {
                let mut crowd = Crowd::new();
                let g = crowd.add_group(Group { goal: Goal::Wander, cohesion: 0.3, ..Group::default() });
                for k in 0..500 {
                    crowd.spawn(g, Vec3::new((k % 25) as f32, (k / 25) as f32, 0.0), 0.3);
                }
                crowd.focus = vec![Vec3::ZERO];
                crowd.near = 10.0;
                for _ in 0..200 {
                    crowd.step(1.0 / 30.0);
                }
                crowd.position
            })
        };
        assert_eq!(run(1), run(4));
    }
}
