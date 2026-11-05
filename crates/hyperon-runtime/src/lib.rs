//! Hyperon runtime world and fixed-step physics simulation.

use std::time::Duration;

use hyperon_box3d::{Box3dSettings, PhysicsWorld, RigidBodyComponent};
use hyperon_core::{ComponentRegistry, PhysicsEngine, Scene, World};

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
    pub hyperon_physics: Option<hyperon_physics::PhysicsWorld>,
    pub hyperon_settings: hyperon_physics::PhysicsSettings,
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
            hyperon_physics: None,
            hyperon_settings: hyperon_physics::PhysicsSettings::default(),
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
        let physics = scene.physics;
        if !physics.gravity.iter().all(|v| v.is_finite()) {
            return Err("Physics gravity must be finite".into());
        }
        if physics.iterations == 0 {
            return Err("Physics iterations must be positive".into());
        }
        if physics.substeps == 0 || physics.substeps > i32::MAX as u32 {
            return Err("Physics substeps must fit a positive integer".into());
        }
        let gravity = glam::Vec3::from_array(physics.gravity);
        runtime.physics_settings.gravity = gravity;
        runtime.physics_settings.substeps = physics.substeps;
        runtime.hyperon_settings.gravity = gravity;
        runtime.hyperon_settings.iterations = physics.iterations;
        if physics.engine == PhysicsEngine::Hyperon {
            runtime.hyperon_physics = Some(hyperon_physics::PhysicsWorld::default());
        }
        let bodies = runtime
            .world
            .query::<RigidBodyComponent>()
            .map(|(entity, body)| {
                let object = runtime.world.get(entity).expect("entities come from this world");
                let name = object.name;
                let position = glam::Vec3::from_array(object.transform.translation);
                let rotation = glam::Quat::from_array(object.transform.rotation);
                if !position.is_finite()
                    || !rotation.is_finite()
                    || (rotation.length_squared() - 1.0).abs() >= 0.001
                {
                    return Err(format!("{name}: Rigid Body: position must be finite and rotation must be a unit quaternion"));
                }
                body.to_body()
                    .map(|body| (entity, body))
                    .map_err(|e| format!("{name}: Rigid Body: {e}"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        for (entity, body) in bodies {
            if let Some(physics) = &mut runtime.hyperon_physics {
                let name = runtime.world.get(entity).map_or("", |e| e.name);
                let collider = match body.collider {
                    hyperon_box3d::Collider::Cuboid { half_extents } => {
                        hyperon_physics::Collider::Cuboid { half_extents }
                    }
                    hyperon_box3d::Collider::Sphere { radius } => {
                        hyperon_physics::Collider::Sphere { radius }
                    }
                    _ => {
                        return Err(format!(
                            "{name}: Rigid Body: Hyperon physics does not support this collider"
                        ));
                    }
                };
                if body.material.restitution != 0.0 {
                    return Err(format!(
                        "{name}: Rigid Body: Hyperon physics does not support restitution"
                    ));
                }
                let body_type = match body.body_type {
                    hyperon_box3d::RigidBody::Static => hyperon_physics::RigidBody::Static,
                    hyperon_box3d::RigidBody::Kinematic => hyperon_physics::RigidBody::Kinematic,
                    hyperon_box3d::RigidBody::Dynamic => hyperon_physics::RigidBody::Dynamic,
                };
                let mut custom = hyperon_physics::PhysicsBody::new(body_type, collider);
                custom.material = hyperon_physics::PhysicsMaterial {
                    density: body.material.density,
                    friction: body.material.friction,
                    static_friction: body.material.friction,
                };
                physics
                    .insert(&runtime.world, entity, custom)
                    .expect("entities come from this world");
            } else {
                runtime
                    .physics
                    .insert(&runtime.world, entity, body)
                    .expect("entities come from this world");
            }
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
            if let Some(physics) = &mut self.hyperon_physics {
                physics.step(
                    &mut self.world,
                    &self.hyperon_settings,
                    self.fixed_timestep.as_secs_f32(),
                );
            } else {
                self.physics.step(
                    &mut self.world,
                    &self.physics_settings,
                    self.fixed_timestep.as_secs_f32(),
                );
            }
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
        assert!(
            error.contains("Copper") && error.contains("radius"),
            "{error}"
        );
    }

    #[test]
    fn hyperon_scene_simulates_authored_body_with_scene_settings() {
        let mut scene = Scene::default();
        scene.physics.engine = PhysicsEngine::Hyperon;
        scene.physics.gravity = [0.0, 0.0, -3.0];
        scene.physics.iterations = 3;
        scene.entities[0].transform.translation = [0.0, 0.0, 5.0];
        scene.entities[0].components.insert(
            "Rigid Body".into(),
            serde_json::to_value(RigidBodyComponent::default()).unwrap(),
        );
        let mut runtime = Runtime::from_scene(&scene).unwrap();
        let entity = runtime.world.handles()[0];
        assert!(runtime.hyperon_physics.as_ref().unwrap().contains(entity));
        assert!(!runtime.physics.contains(entity));
        assert_eq!(runtime.hyperon_settings.iterations, 3);
        assert_eq!(
            runtime.hyperon_settings.gravity.to_array(),
            scene.physics.gravity
        );
        runtime.advance(Duration::from_millis(100));
        assert!(runtime.world.get(entity).unwrap().transform.translation[2] < 5.0);
    }

    #[test]
    fn hyperon_scene_rejects_unsupported_and_invalid_physics() {
        let mut scene = Scene::default();
        scene.physics.engine = PhysicsEngine::Hyperon;
        for (body, expected) in [
            (serde_json::json!({ "shape": "Capsule" }), "collider"),
            (serde_json::json!({ "restitution": 0.5 }), "restitution"),
            (
                serde_json::json!({ "shape": "Sphere", "radius": -1.0 }),
                "radius",
            ),
        ] {
            scene.entities[0]
                .components
                .insert("Rigid Body".into(), body);
            let error = Runtime::from_scene(&scene).err().unwrap();
            assert!(
                error.contains("Copper") && error.contains(expected),
                "{error}"
            );
        }
        scene.entities[0].components.clear();
        scene.physics.iterations = 0;
        assert!(
            Runtime::from_scene(&scene)
                .err()
                .unwrap()
                .contains("iterations")
        );
        scene.physics.iterations = 6;
        scene.physics.substeps = 0;
        assert!(
            Runtime::from_scene(&scene)
                .err()
                .unwrap()
                .contains("substeps")
        );
        scene.physics.substeps = 4;
        scene.physics.gravity[2] = f32::NAN;
        assert!(
            Runtime::from_scene(&scene)
                .err()
                .unwrap()
                .contains("gravity")
        );
        scene.physics.gravity[2] = -9.81;
        scene.entities[0].components.insert(
            "Rigid Body".into(),
            serde_json::to_value(RigidBodyComponent::default()).unwrap(),
        );
        scene.entities[0].transform.rotation = [0.0; 4];
        assert!(
            Runtime::from_scene(&scene)
                .err()
                .unwrap()
                .contains("rotation")
        );
    }
}
