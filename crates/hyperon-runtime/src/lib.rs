//! Hyperon runtime world and fixed-step Box3D simulation.

use std::time::Duration;

use hyperon_box3d::{Box3dSettings, PhysicsWorld};
use hyperon_core::{Scene, World};

/// Owns the authoritative runtime world and its native physics simulation.
///
/// A runtime must not reuse its physics world with another [`World`]: entity
/// handles are scoped to the world that issued them.
pub struct Runtime {
    pub world: World,
    pub physics: PhysicsWorld,
    pub physics_settings: Box3dSettings,
    fixed_timestep: Duration,
    accumulated: Duration,
}

impl Runtime {
    /// Creates an empty world running physics at approximately 60 Hz.
    pub fn new() -> Self {
        Self {
            world: World::new(),
            physics: PhysicsWorld::default(),
            physics_settings: Box3dSettings::default(),
            fixed_timestep: Duration::from_nanos(16_666_667),
            accumulated: Duration::ZERO,
        }
    }

    /// Creates a runtime from a scene. Physics bodies are explicitly attached
    /// to the new world's handles using `physics.insert`.
    pub fn from_scene(scene: Scene) -> Self {
        Self {
            world: scene.into(),
            ..Self::new()
        }
    }

    /// Returns the simulation interval used by [`Self::advance`].
    pub fn fixed_timestep(&self) -> Duration {
        self.fixed_timestep
    }

    /// Changes the interval without discarding previously accumulated time.
    /// A zero interval cannot advance a simulation.
    pub fn set_fixed_timestep(&mut self, timestep: Duration) -> Result<(), &'static str> {
        if timestep.is_zero() || !timestep.as_secs_f32().is_finite() {
            return Err("physics timestep must be finite and positive");
        }
        self.fixed_timestep = timestep;
        Ok(())
    }

    /// Runs all complete fixed ticks in `elapsed`, returning the tick count.
    /// Remaining fractional time carries into the next call.
    pub fn advance(&mut self, elapsed: Duration) -> u64 {
        self.accumulated = self
            .accumulated
            .checked_add(elapsed)
            .expect("elapsed time overflow");
        let mut ticks = 0u64;
        while self.accumulated >= self.fixed_timestep {
            self.physics.step(
                &mut self.world,
                &self.physics_settings,
                self.fixed_timestep.as_secs_f32(),
            );
            self.accumulated -= self.fixed_timestep;
            ticks = ticks.checked_add(1).expect("physics tick count overflow");
        }
        ticks
    }
}

impl Default for Runtime {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hyperon_box3d::{Collider, PhysicsBody, RigidBody};
    use hyperon_core::{Entity, Material, Transform};

    #[test]
    fn fixed_step_carries_fractional_time_and_moves_dynamic_body() {
        let mut runtime = Runtime::new();
        runtime
            .set_fixed_timestep(Duration::from_millis(10))
            .unwrap();
        let entity = runtime.world.spawn(Entity {
            name: "falling".into(),
            transform: Transform {
                translation: [0.0, 5.0, 0.0],
                ..Transform::default()
            },
            material: Material::default(),
        });
        runtime
            .physics
            .insert(
                &runtime.world,
                entity,
                PhysicsBody::new(RigidBody::Dynamic, Collider::Sphere { radius: 0.5 }),
            )
            .unwrap();
        assert_eq!(runtime.advance(Duration::from_millis(9)), 0);
        assert_eq!(
            runtime.world.get(entity).unwrap().transform.translation[1],
            5.0
        );
        assert_eq!(runtime.advance(Duration::from_millis(21)), 3);
        assert!(runtime.world.get(entity).unwrap().transform.translation[1] < 5.0);
    }
}
