//! Followers of a lane graph: road traffic, trains, racing lines. Each
//! vehicle is a distance along a lane. Sorting by lane and distance makes a
//! vehicle's leader its neighbour in the sorted order, so a step is a sort
//! and two parallel passes, without any spatial search.

use glam::Vec3;
use rayon::prelude::*;
use verity_nav::{Csr, Path, NONE};

use crate::{hash, unit};

/// Intelligent Driver Model: comfortable braking, time headway, standstill gap.
const COMFORT: f32 = 2.5;
const HEADWAY: f32 = 1.2;
const STANDSTILL: f32 = 2.0;
/// Hardest a vehicle brakes, and what a caution signal asks it to stop within.
const HARDEST: f32 = 9.0;
const CAUTION: f32 = 4.5;
/// Far vehicles decide once per this many steps.
const FAR_EVERY: u64 = 4;
/// How far short of a closed gate vehicles stop.
const STOP_SHORT: f32 = 0.5;

/// What a gate at the end of a lane tells vehicles.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Signal {
    #[default]
    Open,
    /// Stop if it can be done comfortably, else go.
    Caution,
    Closed,
}

#[derive(Clone, Debug, Default)]
pub struct LaneGraph {
    pub paths: Vec<Path>,
    /// Speed limit of each lane, metres per second.
    pub speed: Vec<f32>,
    /// The gate at each lane's end, or `NONE`.
    pub gate: Vec<u32>,
    pub next: Csr,
    /// Centre and radius of each lane in the ground plane, for distance checks.
    bounds: Vec<(Vec3, f32)>,
}

impl LaneGraph {
    /// Lanes as (path, speed limit, gate), and links from lane to lane.
    pub fn new(lanes: Vec<(Path, f32, u32)>, links: impl IntoIterator<Item = (u32, u32)>) -> Self {
        let n = lanes.len() as u32;
        let next = Csr::new(n as usize, links.into_iter().filter(|&(a, b)| a < n && b < n));
        let bounds = (lanes.iter())
            .map(|(p, ..)| {
                let (c, r) = p.circle();
                (c.extend(p.points.first().map_or(0.0, |p| p.z)), r)
            })
            .collect();
        let (paths, rest): (Vec<Path>, Vec<(f32, u32)>) = lanes.into_iter().map(|(p, s, g)| (p, (s, g))).unzip();
        Self {
            paths,
            speed: rest.iter().map(|r| r.0.max(0.1)).collect(),
            gate: rest.iter().map(|r| r.1).collect(),
            next,
            bounds,
        }
    }

    pub fn len(&self) -> usize {
        self.paths.len()
    }

    pub fn is_empty(&self) -> bool {
        self.paths.is_empty()
    }
}

pub struct Traffic {
    pub lanes: LaneGraph,
    pub lane: Vec<u32>,
    /// Distance of the vehicle's front along its lane.
    pub s: Vec<f32>,
    pub speed: Vec<f32>,
    /// The lane it takes after this one, or `NONE`.
    pub next: Vec<u32>,
    pub length: Vec<f32>,
    /// Share of the speed limit this driver likes, around 1.
    pub eagerness: Vec<f32>,
    /// State of each gate.
    pub gates: Vec<Signal>,
    /// Points vehicles are watched from; far vehicles decide less often.
    pub focus: Vec<Vec3>,
    pub near: f32,
    pub seed: u32,
    tick: u64,
    order: Vec<u32>,
    /// Vehicle with the smallest `s` on each lane, or `NONE`.
    first: Vec<u32>,
}

/// One vehicle's state after a step.
#[derive(Clone, Copy)]
struct Moved {
    lane: u32,
    s: f32,
    speed: f32,
    next: u32,
}

impl Traffic {
    pub fn new(lanes: LaneGraph, seed: u32) -> Self {
        let gates = lanes.gate.iter().filter(|&&g| g != NONE).map(|&g| g as usize + 1).max().unwrap_or(0);
        Self {
            first: vec![NONE; lanes.len()],
            lanes,
            lane: Vec::new(),
            s: Vec::new(),
            speed: Vec::new(),
            next: Vec::new(),
            length: Vec::new(),
            eagerness: Vec::new(),
            gates: vec![Signal::Open; gates],
            focus: Vec::new(),
            near: 150.0,
            seed,
            tick: 0,
            order: Vec::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.lane.len()
    }

    pub fn is_empty(&self) -> bool {
        self.lane.is_empty()
    }

    fn choose(&self, vehicle: u32, lane: u32, salt: u64) -> u32 {
        let options = self.lanes.next.get(lane);
        if options.is_empty() {
            return NONE;
        }
        let h = hash(hash(vehicle, lane), self.seed ^ salt as u32);
        options[h as usize % options.len()]
    }

    /// Add a vehicle of `length` with its front `s` along `lane`.
    pub fn spawn(&mut self, lane: u32, s: f32, length: f32) -> u32 {
        let i = self.len() as u32;
        let s = s.clamp(0.0, self.lanes.paths.get(lane as usize).map_or(0.0, Path::length));
        self.lane.push(lane);
        self.s.push(s);
        self.speed.push(0.0);
        self.next.push(self.choose(i, lane, 0));
        self.length.push(length.max(1.0));
        self.eagerness.push(0.85 + unit(hash(i, self.seed)) * 0.3);
        i
    }

    /// Fill lanes without gates evenly to `per_km` vehicles per kilometre,
    /// each `length(i)` long. Lanes with gates (approaches) start empty.
    pub fn populate(&mut self, per_km: f32, length: impl Fn(u32) -> f32) {
        for lane in 0..self.lanes.len() as u32 {
            let path_len = self.lanes.paths[lane as usize].length();
            let count = (path_len * per_km.max(0.0) / 1000.0).floor() as usize;
            if self.lanes.gate[lane as usize] != NONE || self.lanes.next.get(lane).is_empty() {
                continue;
            }
            let spacing = path_len / count.max(1) as f32;
            for k in 0..count {
                let l = length(self.len() as u32);
                if spacing < l + STANDSTILL {
                    break;
                }
                self.spawn(lane, path_len - (k as f32 + 0.5) * spacing, l);
            }
        }
    }

    /// Front position and unit heading of vehicle `i`.
    pub fn pose(&self, i: u32) -> (Vec3, Vec3) {
        self.lanes.paths[self.lane[i as usize] as usize].at(self.s[i as usize])
    }

    /// Advance every vehicle by `dt` seconds.
    pub fn step(&mut self, dt: f32) {
        if dt.is_nan() || dt <= 0.0 || self.is_empty() {
            return;
        }
        let key = |i: u32| (self.lane[i as usize] as u64) << 32 | self.s[i as usize].max(0.0).to_bits() as u64;
        let mut order: Vec<u32> = std::mem::take(&mut self.order);
        order.clear();
        order.extend(0..self.len() as u32);
        order.par_sort_unstable_by_key(|&i| key(i));
        self.first.fill(NONE);
        for &i in order.iter().rev() {
            self.first[self.lane[i as usize] as usize] = i;
        }
        // Each vehicle's leader on its lane: the next in sorted order.
        let mut leader = vec![NONE; self.len()];
        for pair in order.windows(2) {
            if self.lane[pair[0] as usize] == self.lane[pair[1] as usize] {
                leader[pair[0] as usize] = pair[1];
            }
        }
        self.order = order;
        let near: Vec<bool> = (self.lanes.bounds.iter())
            .map(|&(c, r)| {
                self.focus.is_empty()
                    || (self.focus.iter()).any(|f| f.truncate().distance(c.truncate()) <= self.near + r)
            })
            .collect();
        let moved: Vec<Moved> = (0..self.len() as u32)
            .into_par_iter()
            .map(|i| self.drive(i, leader[i as usize], &near, dt))
            .collect();
        for (i, m) in moved.into_iter().enumerate() {
            self.lane[i] = m.lane;
            self.s[i] = m.s;
            self.speed[i] = m.speed;
            self.next[i] = m.next;
        }
        self.tick += 1;
    }

    fn drive(&self, i: u32, leader: u32, near: &[bool], dt: f32) -> Moved {
        let k = i as usize;
        let (lane, s, v) = (self.lane[k], self.s[k], self.speed[k]);
        let mut next = self.next[k];
        let here = Moved { lane, s, speed: v, next };
        let lane_len = self.lanes.paths[lane as usize].length();
        // Far vehicles keep their speed between decisions, but never their gap.
        let decides = near[lane as usize] || (k as u64 + self.tick).is_multiple_of(FAR_EVERY);
        // Bumper gap to the vehicle ahead, here or on the next lane, and its speed.
        let ahead = if leader != NONE {
            let l = leader as usize;
            Some((self.s[l] - self.length[l] - s, self.speed[l]))
        } else if next != NONE && self.first[next as usize] != NONE {
            let l = self.first[next as usize] as usize;
            Some((lane_len - s + self.s[l] - self.length[l], self.speed[l]))
        } else {
            None
        };
        let gate = self.lanes.gate[lane as usize];
        let signal = (gate != NONE).then(|| self.gates.get(gate as usize).copied().unwrap_or_default());
        let to_line = lane_len - s - STOP_SHORT;
        let stops = match signal {
            Some(Signal::Closed) => true,
            Some(Signal::Caution) => to_line >= v * v / (2.0 * CAUTION),
            _ => next == NONE,
        };
        let obstacle = match (ahead, stops && to_line > -STOP_SHORT) {
            (Some((gap, w)), true) if gap < to_line => Some((gap, w)),
            (_, true) => Some((to_line, 0.0)),
            (a, false) => a,
        };
        let limit = self.lanes.speed[lane as usize] * self.eagerness[k];
        let accel = (2.0 - (self.length[k] - 4.5).max(0.0) * 0.08).max(0.8);
        let a = if decides {
            let free = 1.0 - (v / limit).powi(4);
            let interaction = obstacle.map_or(0.0, |(gap, w)| {
                let wanted = STANDSTILL + v * HEADWAY + v * (v - w) / (2.0 * (accel * COMFORT).sqrt());
                (wanted.max(0.0) / gap.max(0.1)).powi(2)
            });
            (accel * (free - interaction)).clamp(-HARDEST, accel)
        } else {
            0.0
        };
        let speed = (v + a * dt).max(0.0);
        let mut travel = speed * dt;
        // Never into the vehicle ahead's last place, nor over a line it stops at.
        if let Some((gap, _)) = ahead {
            travel = travel.min((gap - 0.1).max(0.0));
        }
        if signal == Some(Signal::Closed) && s <= lane_len - STOP_SHORT {
            travel = travel.min(to_line.max(0.0));
        }
        if next == NONE {
            travel = travel.min((lane_len - s).max(0.0));
        }
        let (mut lane, mut s) = (lane, s + travel);
        let mut speed = if travel < speed * dt { travel / dt } else { speed };
        while s > self.lanes.paths[lane as usize].length() && next != NONE {
            s -= self.lanes.paths[lane as usize].length();
            lane = next;
            next = self.choose(i, lane, self.tick);
        }
        if next == NONE && s >= self.lanes.paths[lane as usize].length() {
            speed = 0.0;
        }
        if !s.is_finite() || !speed.is_finite() {
            return here;
        }
        Moved { lane, s, speed, next }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use verity_nav::spline::bend;

    /// A 200 m two-way road with U-turns at both ends; lane 0's end is gate 0.
    fn road() -> LaneGraph {
        let (a, b) = (Vec3::new(0.0, 0.0, 0.0), Vec3::new(200.0, 0.0, 0.0));
        let up = Vec3::new(0.0, 5.0, 0.0);
        LaneGraph::new(
            vec![
                (Path::new([a, b]), 14.0, 0),
                (Path::new([b + up, a + up]), 14.0, NONE),
                (Path::new(bend(b, Vec3::X, b + up, -Vec3::X, 8)), 5.0, NONE),
                (Path::new(bend(a + up, -Vec3::X, a, Vec3::X, 8)), 5.0, NONE),
            ],
            [(0, 2), (2, 1), (1, 3), (3, 0)],
        )
    }

    fn check(traffic: &Traffic) {
        let mut by_lane: Vec<(u32, f32, f32)> =
            (0..traffic.len()).map(|i| (traffic.lane[i], traffic.s[i], traffic.length[i])).collect();
        by_lane.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.total_cmp(&b.1)));
        for pair in by_lane.windows(2) {
            if pair[0].0 == pair[1].0 {
                assert!(pair[1].1 - pair[1].2 >= pair[0].1 - 1e-3, "overlap on lane {}: {pair:?}", pair[0].0);
            }
        }
    }

    #[test]
    fn vehicles_never_pass_their_leader_nor_a_closed_gate() {
        let mut traffic = Traffic::new(road(), 7);
        for k in 0..20 {
            let lane = k % 2;
            traffic.spawn(lane, 190.0 - (k / 2) as f32 * 18.0, if k % 5 == 0 { 12.0 } else { 4.5 });
        }
        let mut passed = 0;
        for t in 0..10_000u32 {
            traffic.gates[0] = match t % 1200 {
                0..600 => Signal::Open,
                600..800 => Signal::Caution,
                _ => Signal::Closed,
            };
            let before: Vec<u32> = traffic.lane.clone();
            traffic.step(1.0 / 60.0);
            for i in 0..traffic.len() {
                if before[i] == 0 && traffic.lane[i] != 0 {
                    passed += 1;
                    assert_ne!(traffic.gates[0], Signal::Closed, "vehicle {i} crossed a closed gate at tick {t}");
                }
            }
            check(&traffic);
        }
        assert!(passed > 20, "traffic hardly moved: {passed}");
        assert!(traffic.speed.iter().all(|&v| v <= 14.0 * 1.15 + 1e-3));
    }

    #[test]
    fn stepping_is_deterministic_and_thread_independent() {
        let run = |threads: usize| {
            rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap().install(|| {
                let mut traffic = Traffic::new(road(), 3);
                traffic.populate(60.0, |i| 4.0 + (i % 3) as f32);
                traffic.focus = vec![Vec3::ZERO];
                traffic.near = 50.0;
                for t in 0..3000 {
                    traffic.gates[0] = if t % 900 < 450 { Signal::Open } else { Signal::Closed };
                    traffic.step(1.0 / 60.0);
                }
                (traffic.lane, traffic.s)
            })
        };
        let (a, b) = (run(1), run(4));
        assert_eq!(a, b);
        assert!(!a.0.is_empty());
    }
}
