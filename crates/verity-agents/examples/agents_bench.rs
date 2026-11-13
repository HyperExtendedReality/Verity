//! Time per step of 100k crowd agents and 100k vehicles.
//! `cargo run -p verity-agents --example agents_bench --release`

use std::time::Instant;

use glam::Vec3;
use verity_agents::{Crowd, Goal, Group, LaneGraph, Signal, Traffic};
use verity_nav::{Path, NONE};

fn main() {
    let mut crowd = Crowd::new();
    let wander = crowd.add_group(Group { goal: Goal::Wander, cohesion: 0.2, ..Group::default() });
    for k in 0..100_000u32 {
        // A 1 km square, about 3 m apart.
        let (x, y) = ((k % 316) as f32 * 3.16, (k / 316) as f32 * 3.16);
        crowd.spawn(wander, Vec3::new(x, y, 0.0), 0.3);
    }
    crowd.focus = vec![Vec3::new(500.0, 500.0, 0.0)];
    crowd.near = 100.0;

    // 1000 square loops of four 500 m lanes, the first lane of each gated.
    let mut lanes = Vec::new();
    let mut links = Vec::new();
    for l in 0..1000u32 {
        let o = Vec3::new((l % 40) as f32 * 600.0, (l / 40) as f32 * 600.0, 0.0);
        let c = [Vec3::ZERO, Vec3::X * 500.0, Vec3::new(500.0, 500.0, 0.0), Vec3::Y * 500.0].map(|p| p + o);
        for side in 0..4u32 {
            let gate = if side == 0 { l } else { NONE };
            lanes.push((Path::new([c[side as usize], c[(side as usize + 1) % 4]]), 14.0, gate));
            links.push((l * 4 + side, l * 4 + (side + 1) % 4));
        }
    }
    let mut traffic = Traffic::new(LaneGraph::new(lanes, links), 1);
    traffic.populate(66.0, |i| if i % 10 == 0 { 12.0 } else { 4.5 });
    traffic.focus = vec![Vec3::new(500.0, 500.0, 0.0)];

    println!("{} agents, {} vehicles", crowd.len(), traffic.len());
    let dt = 1.0 / 30.0;
    for _ in 0..10 {
        crowd.step(dt);
        traffic.step(dt);
    }
    let steps = 100;
    let (mut crowd_time, mut traffic_time) = (0.0, 0.0);
    for t in 0..steps {
        let signal = if t % 60 < 30 { Signal::Open } else { Signal::Closed };
        traffic.gates.fill(signal);
        let start = Instant::now();
        crowd.step(dt);
        crowd_time += start.elapsed().as_secs_f64();
        let start = Instant::now();
        traffic.step(dt);
        traffic_time += start.elapsed().as_secs_f64();
    }
    println!(
        "crowd {:.2} ms/step, traffic {:.2} ms/step on {} threads",
        crowd_time * 1000.0 / steps as f64,
        traffic_time * 1000.0 / steps as f64,
        rayon::current_num_threads()
    );
}
