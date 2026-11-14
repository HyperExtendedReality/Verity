//! A game with one component and one system, run in the editor:
//! `cargo run -p verity-editor --example spinner`, add a Spin component to an
//! entity, and press Play.

use glam::{Quat, Vec3};
use serde::{Deserialize, Serialize};
use verity_core::{Component, FieldMut};
use verity_runtime::{Game, Runtime};

/// Turns its entity about an axis.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
struct Spin {
    axis: [f32; 3],
    /// Degrees per second.
    speed: f32,
}

impl Default for Spin {
    fn default() -> Self {
        Self {
            axis: [0.0, 0.0, 1.0],
            speed: 90.0,
        }
    }
}

impl Component for Spin {
    const NAME: &'static str = "Spin";

    fn fields(&mut self) -> Vec<(&'static str, FieldMut<'_>)> {
        vec![
            ("Axis", FieldMut::Vec3(&mut self.axis, -1.0..=1.0)),
            ("Speed", FieldMut::Float(&mut self.speed, -1440.0..=1440.0)),
        ]
    }
}

fn spin(runtime: &mut Runtime, dt: f32) {
    runtime.world.each_mut(|_, spin: &mut Spin, transform| {
        let Some(axis) = Vec3::from(spin.axis).try_normalize() else {
            return;
        };
        let turn = Quat::from_axis_angle(axis, (spin.speed * dt).to_radians());
        let rotation = turn * Quat::from_array(transform.rotation);
        transform.rotation = rotation.normalize().to_array();
    });
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    verity_editor::run(Game::new().component::<Spin>().system(spin))
}
