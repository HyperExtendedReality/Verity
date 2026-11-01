//! Hyperon runtime world and fixed-step Box3D simulation.

use std::time::Duration;

use hyperon_box3d::{Box3dSettings, PhysicsWorld, RigidBodyComponent};
use hyperon_core::{ComponentRegistry, Scene, World};

/// Every component type the runtime simulates.
pub fn registry() -> ComponentRegistry {
    let mut registry = ComponentRegistry::default();
    registry.register::<RigidBodyComponent>();
    registry
}

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

    /// Creates a runtime from a scene, simulating its Rigid Body components.
    /// Fails on unknown components or invalid values.
    pub fn from_scene(scene: &Scene) -> Result<Self, String> {
        let mut runtime = Self {
            world: World::from_scene(scene, &registry())?,
            ..Self::new()
        };
        // Scenes are Z-up.
        runtime.physics_settings.gravity = glam::Vec3::new(0.0, 0.0, -9.81);
        let bodies = runtime
            .world
            .query::<RigidBodyComponent>()
            .map(|(entity, body)| {
                let name = runtime.world.get(entity).map_or("", |e| e.name);
                body.to_body()
                    .map(|body| (entity, body))
                    .map_err(|e| format!("{name}: Rigid Body: {e}"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        for (entity, body) in bodies {
            runtime
                .physics
                .insert(&runtime.world, entity, body)
                .expect("entities come from this world");
        }
        Ok(runtime)
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
            ..Entity::default()
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

    #[test]
    fn scene_rigid_bodies_are_simulated() {
        let mut scene = Scene::default();
        scene.entities[0].transform.translation = [0.0, 0.0, 5.0];
        scene.entities[0].components.insert(
            "Rigid Body".into(),
            serde_json::to_value(RigidBodyComponent::default()).unwrap(),
        );
        let mut runtime = Runtime::from_scene(&scene).unwrap();
        let [falling, resting] = [0, 1].map(|i| runtime.world.handles()[i]);
        assert!(runtime.physics.contains(falling));
        assert!(!runtime.physics.contains(resting));
        runtime.advance(Duration::from_millis(100));
        let fallen = runtime.world.get(falling).unwrap().transform.translation;
        assert!(fallen[2] < 5.0 && fallen[1] == 0.0, "{fallen:?}");

        scene.entities[0].components.insert(
            "Rigid Body".into(),
            serde_json::json!({ "shape": "Sphere", "radius": -1.0 }),
        );
        let error = Runtime::from_scene(&scene).err().unwrap();
        assert!(error.contains("Copper") && error.contains("radius"), "{error}");
    }
}
