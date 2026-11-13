//! Things that move, for any game, kept as structure-of-arrays so a world
//! can hold hundreds of thousands of them.
//!
//! - [`Crowd`]: free-moving agents (people, zombies, animals) whose behaviour
//!   comes from their [`Group`]: stand, wander, seek a target along a navmesh
//!   or graph route, follow a flow field, flee; plus flocking and local
//!   avoidance.
//! - [`Traffic`]: followers of a [`LaneGraph`] (cars, trains, racing lines),
//!   driven by the Intelligent Driver Model and stopped by gates.

pub mod crowd;
pub mod lanes;

pub use crowd::{Crowd, Goal, Group};
pub use lanes::{LaneGraph, Signal, Traffic};

/// A well-mixed 32-bit hash of two values, for seeded, thread-independent choices.
pub fn hash(a: u32, b: u32) -> u32 {
    let mut x = a.wrapping_mul(0x9E37_79B1) ^ b.wrapping_add(0x7F4A_7C15).rotate_left(16);
    x ^= x >> 16;
    x = x.wrapping_mul(0x85EB_CA6B);
    x ^= x >> 13;
    x = x.wrapping_mul(0xC2B2_AE35);
    x ^ (x >> 16)
}

/// A hash as a number in `0..1`.
pub fn unit(h: u32) -> f32 {
    (h >> 8) as f32 / (1 << 24) as f32
}
