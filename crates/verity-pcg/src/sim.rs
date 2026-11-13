//! A city's traffic and pedestrians: the engine's generic lane traffic and
//! crowd, run over the network, with junction signals driving both.

use std::sync::Arc;

use glam::Vec3;
use verity_agents::{Crowd, Goal, Group, Signal, Traffic};

use crate::{hash, unit, City, Network};

/// Vehicle lengths: cars, vans and buses.
pub fn vehicle_length(i: u32) -> f32 {
    match hash(i, 0xCA25) % 100 {
        0..80 => 4.5,
        80..92 => 6.5,
        _ => 12.0,
    }
}

pub struct Simulation {
    pub traffic: Traffic,
    pub crowd: Crowd,
    pub time: f32,
    seed: u32,
    walk: Vec<bool>,
}

impl Simulation {
    pub fn new(city: &City, network: &Network) -> Self {
        let rules = &city.rules;
        let mut traffic = Traffic::new(network.lanes.clone(), city.seed);
        traffic.populate(rules.vehicles_per_km.clamp(0.0, 200.0), vehicle_length);
        let mut crowd = Crowd::new();
        let corridor = (network.sidewalk * 0.5 - 0.3).max(0.2);
        let groups = [1.1, 1.35, 1.6].map(|speed| {
            crowd.add_group(Group { goal: Goal::Seek, speed, corridor, sight: 2.5, ..Group::default() })
        });
        let walks = &network.walks;
        let nodes = walks.len() as u32;
        if nodes > 0 {
            let length: f32 = walks.costs.iter().sum::<f32>() * 0.5;
            let count = (length / 1000.0 * rules.walkers_per_km.clamp(0.0, 1000.0)) as u32;
            for k in 0..count {
                let h = hash(city.seed ^ 0x9ED, k);
                let at = walks.positions[(h % nodes) as usize];
                let jitter = Vec3::new(unit(hash(h, 1)) - 0.5, unit(hash(h, 2)) - 0.5, 0.0);
                let i = crowd.spawn(groups[(h >> 8) as usize % 3], at + jitter, 0.28);
                crowd.set_target(i, walks.positions[(hash(h, 3) % nodes) as usize]);
            }
        }
        crowd.set_graph(Some(Arc::new(walks.clone())));
        crowd.plans_per_step = 256;
        let mut sim = Self { traffic, crowd, time: 0.0, seed: city.seed, walk: Vec::new() };
        sim.signal(network);
        sim
    }

    fn signal(&mut self, network: &Network) {
        let mut cars = std::mem::take(&mut self.traffic.gates);
        network.signals(self.time, &mut cars, &mut self.walk);
        self.traffic.gates = cars;
        self.crowd.gates.clone_from(&self.walk);
    }

    /// Where the agents are watched from; far ones decide less often.
    pub fn set_focus(&mut self, focus: &[Vec3]) {
        self.traffic.focus = focus.to_vec();
        self.crowd.focus = focus.to_vec();
    }

    pub fn step(&mut self, network: &Network, dt: f32) {
        self.signal(network);
        self.traffic.step(dt);
        self.crowd.step(dt);
        // Pedestrians who got there set off somewhere else.
        let nodes = self.crowd.graph().map_or(0, |g| g.len() as u32);
        if nodes > 0 {
            let arrived = std::mem::take(&mut self.crowd.arrived);
            for (k, &i) in arrived.iter().enumerate() {
                let node = hash(hash(i, self.seed), (self.time * 10.0) as u32 ^ k as u32) % nodes;
                let target = self.crowd.graph().unwrap().positions[node as usize];
                self.crowd.set_target(i, target);
            }
        }
        self.time += dt;
    }

    /// Whether a lane's gate is held closed, for drawing signals.
    pub fn signal_of(&self, gate: u32) -> Signal {
        self.traffic.gates.get(gate as usize).copied().unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{flat, tests::grid, Model};

    #[test]
    fn a_town_runs_its_traffic_and_people() {
        let mut city = grid(4, 90.0);
        city.rules.walkers_per_km = 60.0;
        let model = Model::build(&city, &flat);
        let mut sim = Simulation::new(&city, &model.network);
        assert!(sim.traffic.len() > 20 && sim.crowd.len() > 20);
        let start: Vec<Vec3> = sim.crowd.position.clone();
        for _ in 0..30 * 60 {
            let before = sim.traffic.lane.clone();
            sim.step(&model.network, 1.0 / 30.0);
            for i in 0..sim.traffic.len() {
                let lane = before[i] as usize;
                let gate = model.network.lanes.gate[lane];
                if sim.traffic.lane[i] as usize != lane && gate != verity_nav::NONE {
                    assert_ne!(sim.traffic.gates[gate as usize], Signal::Closed, "ran a red light");
                }
            }
        }
        let walked = sim.crowd.position.iter().zip(&start).filter(|(a, b)| a.distance(**b) > 5.0).count();
        assert!(walked > sim.crowd.len() / 2, "only {walked} pedestrians moved");
        // Everyone stays near a sidewalk or crosswalk.
        let walks = &model.network.walks;
        for p in &sim.crowd.position {
            let near = (0..walks.targets.len()).any(|e| {
                let (a, b) = (walks.positions[walks.sources[e] as usize], walks.positions[walks.targets[e] as usize]);
                verity_nav::Path::new([a, b]).nearest(*p).1 < 2.5
            });
            assert!(near, "a pedestrian wandered off to {p}");
        }
    }
}
