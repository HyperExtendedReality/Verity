//! Verity runtime world and fixed-step physics simulation.

pub mod agents;

use std::{
    collections::HashMap,
    time::{Duration, Instant},
};

use agents::{Agents, CrowdComponent};

use verity_core::{ComponentRegistry, EntityHandle, Scene, World};
use verity_physics::{
    JointComponent, PhysicsSettings, PhysicsWorld, RigidBodyComponent, SoftBodyComponent,
    StepProfile,
};

/// Every component type the runtime simulates.
pub fn registry() -> ComponentRegistry {
    let mut registry = ComponentRegistry::default();
    registry.register::<RigidBodyComponent>();
    registry.register::<JointComponent>();
    registry.register::<SoftBodyComponent>();
    registry.register::<CrowdComponent>();
    registry
}

/// A city's network and the traffic and people running on it.
pub struct CityRun {
    pub city: verity_pcg::City,
    pub network: verity_pcg::Network,
    pub sim: verity_pcg::Simulation,
}

impl CityRun {
    /// Build the city laid on `height`.
    pub fn new(city: verity_pcg::City, height: verity_pcg::Height) -> Self {
        let network = verity_pcg::Network::build(&city, height);
        let sim = verity_pcg::Simulation::new(&city, &network);
        Self { city, network, sim }
    }
}

/// Owns the authoritative runtime world and its native physics simulation.
///
/// A runtime must not reuse its physics world with another [`World`]: entity
/// handles are scoped to the world that issued them.
pub struct Runtime {
    pub world: World,
    pub physics: PhysicsWorld,
    pub physics_settings: PhysicsSettings,
    /// Each entity's Joint component in the simulation.
    joints: HashMap<EntityHandle, verity_physics::LinkId>,
    /// Each entity's Soft Body component in the simulation.
    soft: Vec<(EntityHandle, verity_physics::SoftId)>,
    /// Joints that yielded or broke in the last [`Self::advance`].
    pub joint_events: Vec<verity_physics::JointEvent>,
    /// Physics phase times summed over the ticks of the last [`Self::advance`].
    pub last_profile: StepProfile,
    /// Ticks the last [`Self::advance`] ran.
    pub last_ticks: u64,
    /// Crowds from Crowd components.
    pub agents: Agents,
    pub city: Option<CityRun>,
    /// Where the world is watched from; agents far from all of it think less.
    pub focus: Vec<glam::Vec3>,
    /// Time spent on crowds and the city in the last [`Self::advance`].
    pub last_agents: Duration,
    fixed_timestep: Duration,
    accumulated: Duration,
}

impl Runtime {
    /// Creates an empty world running physics at approximately 60 Hz.
    pub fn new() -> Self {
        Self {
            world: World::new(),
            physics: PhysicsWorld::default(),
            physics_settings: PhysicsSettings::default(),
            joints: HashMap::new(),
            soft: Vec::new(),
            joint_events: Vec::new(),
            last_profile: StepProfile::default(),
            last_ticks: 0,
            agents: Agents::default(),
            city: None,
            focus: Vec::new(),
            last_agents: Duration::ZERO,
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
        runtime.physics_settings.gravity = glam::Vec3::from_array(physics.gravity);
        runtime.physics_settings.iterations = physics.iterations;
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
            runtime
                .physics
                .insert(&runtime.world, entity, body)
                .expect("entities come from this world");
        }
        runtime.add_joints()?;
        runtime.add_soft_bodies()?;
        runtime.agents = Agents::from_world(&runtime.world)?;
        if let Some(recipe) = &scene.city {
            let city: verity_pcg::City =
                serde_json::from_value(recipe.clone()).map_err(|e| format!("City: {e}"))?;
            runtime.city = Some(CityRun::new(city, &verity_pcg::flat));
        }
        Ok(runtime)
    }

    /// Add the scene's Joint components to the simulation. A joint's other
    /// anchor is wherever this entity's anchor starts.
    fn add_joints(&mut self) -> Result<(), String> {
        let joints: Vec<_> = (self.world.query::<JointComponent>())
            .map(|(entity, joint)| (entity, joint.clone()))
            .collect();
        for (entity, joint) in joints {
            let name = self.world.get(entity).map_or("", |e| e.name).to_owned();
            let other = if joint.connected_to.is_empty() {
                None
            } else {
                let found = find_named(&self.world, &joint.connected_to)
                    .map_err(|e| format!("{name}: Joint: {e}"))?;
                Some(found)
            };
            let pose = |e: EntityHandle| {
                let t = &self
                    .world
                    .get(e)
                    .expect("entities come from this world")
                    .transform;
                (
                    glam::Vec3::from_array(t.translation),
                    glam::Quat::from_array(t.rotation).normalize(),
                )
            };
            let (position, rotation) = pose(entity);
            let point = position + rotation * glam::Vec3::from_array(joint.anchor);
            let anchor = other.map_or(point, |other| {
                let (position, rotation) = pose(other);
                rotation.conjugate() * (point - position)
            });
            let authored = joint
                .to_joint(other, entity, anchor)
                .map_err(|e| format!("{name}: Joint: {e}"))?;
            let id = (self.physics)
                .add_joint(&self.world, authored)
                .map_err(|_| format!("{name}: Joint: both entities need a Rigid Body"))?;
            self.joints.insert(entity, id);
        }
        Ok(())
    }

    /// Add the scene's Soft Body components to the simulation.
    fn add_soft_bodies(&mut self) -> Result<(), String> {
        let soft: Vec<_> = (self.world.query::<SoftBodyComponent>())
            .map(|(entity, soft)| (entity, soft.clone()))
            .collect();
        for (entity, soft) in soft {
            let name = self.world.get(entity).map_or("", |e| e.name).to_owned();
            let error = |e: String| format!("{name}: Soft Body: {e}");
            let physics = &mut self.physics;
            let t = &self
                .world
                .get(entity)
                .expect("entities come from this world")
                .transform;
            let position = glam::Vec3::from_array(t.translation);
            let rotation = glam::Quat::from_array(t.rotation).normalize();
            let mut body = soft.to_soft_body(position, rotation).map_err(&error)?;
            let strength = if soft.pin_strength > 0.0 {
                soft.pin_strength
            } else {
                f32::INFINITY
            };
            let local = |p: glam::Vec3| rotation.conjugate() * (p - position);
            let other = if soft.connected_to.is_empty() {
                None
            } else {
                Some(find_named(&self.world, &soft.connected_to).map_err(&error)?)
            };
            if other.is_none() {
                body.pin(|p| soft.pins(local(p)), strength);
            }
            let id = physics.add_soft_body(body);
            if let Some(other) = other {
                let inside = soft.pin == verity_physics::SoftPin::InsideBody;
                let t = &self
                    .world
                    .get(other)
                    .expect("found in this world")
                    .transform;
                let (center, turn) = (
                    glam::Vec3::from_array(t.translation),
                    glam::Quat::from_array(t.rotation).normalize(),
                );
                let collider = physics.get(other).map(|b| b.collider);
                let within = |p: glam::Vec3| {
                    let q = turn.conjugate() * (p - center);
                    match collider {
                        Some(verity_physics::Collider::Cuboid { half_extents }) => {
                            q.abs().cmple(half_extents + 0.05).all()
                        }
                        Some(verity_physics::Collider::Sphere { radius }) => {
                            q.length() <= radius + 0.05
                        }
                        Some(verity_physics::Collider::Capsule {
                            radius,
                            half_height,
                        }) => {
                            let core = q.z.clamp(-half_height, half_height);
                            (q - glam::Vec3::Z * core).length() <= radius + 0.05
                        }
                        Some(verity_physics::Collider::Ground) | None => false,
                    }
                };
                physics
                    .pin_soft_body(
                        &self.world,
                        id,
                        |p| {
                            if inside {
                                within(p)
                            } else {
                                soft.pins(local(p))
                            }
                        },
                        other,
                        strength,
                    )
                    .map_err(|_| error("the pinned entity needs a Rigid Body".into()))?;
            }
            self.soft.push((entity, id));
        }
        Ok(())
    }

    /// Each soft body's entity, particle positions and triangles.
    pub fn soft_meshes(
        &self,
    ) -> impl Iterator<Item = (EntityHandle, &[glam::Vec3], &[[u32; 3]])> + '_ {
        (self.soft.iter()).filter_map(|&(entity, id)| {
            let body = self.physics.soft_body(id)?;
            Some((entity, body.positions(), body.triangles()))
        })
    }

    /// An entity's Joint component in the simulation, if it has one.
    pub fn joint_status(&self, entity: EntityHandle) -> Option<verity_physics::JointStatus> {
        let id = self.joints.get(&entity)?;
        self.physics.joint_status(*id)
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
        self.joint_events.clear();
        self.last_profile = StepProfile::default();
        self.last_agents = Duration::ZERO;
        self.agents.crowd.focus.clone_from(&self.focus);
        if let Some(city) = &mut self.city {
            city.sim.set_focus(&self.focus);
        }
        while self.accumulated >= self.fixed_timestep {
            let dt = self.fixed_timestep.as_secs_f32();
            self.physics.step(&mut self.world, &self.physics_settings, dt);
            self.joint_events
                .extend_from_slice(self.physics.joint_events());
            self.last_profile.accumulate(&self.physics.last_profile());
            let start = Instant::now();
            self.agents.step(&self.world, dt);
            if let Some(city) = &mut self.city {
                city.sim.step(&city.network, dt);
            }
            self.last_agents += start.elapsed();
            self.accumulated -= self.fixed_timestep;
            ticks = ticks.checked_add(1).expect("physics tick count overflow");
        }
        self.last_ticks = ticks;
        ticks
    }
}

/// The one entity of this name.
fn find_named(world: &World, name: &str) -> Result<EntityHandle, String> {
    let mut matches =
        (world.handles().iter().copied()).filter(|&e| world.get(e).is_some_and(|o| o.name == name));
    let found = matches
        .next()
        .ok_or_else(|| format!("no entity named {name:?}"))?;
    if matches.next().is_some() {
        return Err(format!("more than one entity is named {name:?}"));
    }
    Ok(found)
}

impl Default for Runtime {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use verity_core::{Entity, Material, Transform};
    use verity_physics::{Collider, PhysicsBody, RigidBody};

    #[test]
    fn fixed_step_carries_fractional_time_and_moves_dynamic_body() {
        let mut runtime = Runtime::new();
        runtime
            .set_fixed_timestep(Duration::from_millis(10))
            .unwrap();
        let entity = runtime.world.spawn(Entity {
            name: "falling".into(),
            transform: Transform {
                translation: [0.0, 0.0, 5.0],
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
            runtime.world.get(entity).unwrap().transform.translation[2],
            5.0
        );
        assert_eq!(runtime.advance(Duration::from_millis(21)), 3);
        assert!(runtime.world.get(entity).unwrap().transform.translation[2] < 5.0);
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

    fn jointed_scene() -> Scene {
        let mut scene = Scene::default();
        scene.entities[0].name = "Hanging".into();
        scene.entities[0].transform.translation = [0.0, 0.0, 5.0];
        scene.entities[0].components.insert(
            "Rigid Body".into(),
            serde_json::to_value(RigidBodyComponent::default()).unwrap(),
        );
        scene.entities[0].components.insert(
            "Joint".into(),
            serde_json::to_value(JointComponent {
                anchor: [0.0, 0.0, 1.0],
                break_force: 5_000.0,
                ..Default::default()
            })
            .unwrap(),
        );
        scene
    }

    #[test]
    fn authored_cloth_hangs_from_its_top_edge_and_hides_nothing_else() {
        let mut scene = Scene::default();
        scene.entities[0].transform.translation = [0.0, 0.0, 5.0];
        scene.entities[0].components.insert(
            "Soft Body".into(),
            serde_json::to_value(SoftBodyComponent::default()).unwrap(),
        );
        let mut runtime = Runtime::from_scene(&scene).unwrap();
        for _ in 0..60 {
            runtime.advance(runtime.fixed_timestep());
        }
        let (_, positions, triangles) = runtime.soft_meshes().next().unwrap();
        assert_eq!(positions.len(), 21 * 21);
        assert_eq!(triangles.len(), 20 * 20 * 2);
        let top = positions.iter().fold(f32::MIN, |t, p| t.max(p.z));
        let bottom = positions.iter().fold(f32::MAX, |b, p| b.min(p.z));
        assert!(top > 4.9 && bottom < 4.5, "held at {top}, fell to {bottom}");
    }

    #[test]
    fn invalid_soft_bodies_are_reported_by_name() {
        let mut scene = Scene::default();
        scene.entities[0].name = "Flag".into();
        scene.entities[0].components.insert(
            "Soft Body".into(),
            serde_json::to_value(SoftBodyComponent::default()).unwrap(),
        );
        let bad = SoftBodyComponent {
            size: [0.0; 3],
            ..Default::default()
        };
        scene.entities[0]
            .components
            .insert("Soft Body".into(), serde_json::to_value(bad).unwrap());
        let error = Runtime::from_scene(&scene).err().unwrap();
        assert!(error.contains("Flag") && error.contains("size"), "{error}");
    }

    #[test]
    fn authored_joints_break_and_report_it() {
        // A 1000 kg cube's 9.8 kN against a 5 kN joint.
        let mut runtime = Runtime::from_scene(&jointed_scene()).unwrap();
        let entity = runtime.world.handles()[0];
        let status = runtime.joint_status(entity).unwrap();
        assert_eq!(status.stage, verity_physics::JointStage::Intact);
        let mut events = Vec::new();
        for _ in 0..30 {
            runtime.advance(runtime.fixed_timestep());
            events.extend_from_slice(&runtime.joint_events);
        }
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].body_b, entity);
        let status = runtime.joint_status(entity).unwrap();
        assert_eq!(status.stage, verity_physics::JointStage::Broken);
    }

    #[test]
    fn invalid_joints_are_reported_by_entity_name() {
        let mut scene = jointed_scene();
        scene.entities[0].components.insert(
            "Joint".into(),
            serde_json::to_value(JointComponent {
                connected_to: "Nobody".into(),
                ..Default::default()
            })
            .unwrap(),
        );
        let error = Runtime::from_scene(&scene).err().unwrap();
        assert!(error.contains("Nobody"), "{error}");

        let mut scene = jointed_scene();
        let twin = scene.entities[0].clone();
        scene.entities.push(twin.clone());
        scene.entities.push(twin);
        let joint = serde_json::to_value(JointComponent {
            connected_to: "Hanging".into(),
            ..Default::default()
        })
        .unwrap();
        scene.entities[0].components.insert("Joint".into(), joint);
        let error = Runtime::from_scene(&scene).err().unwrap();
        assert!(error.contains("more than one"), "{error}");
    }

    #[test]
    fn verity_scene_simulates_authored_body_with_scene_settings() {
        let mut scene = Scene::default();
        scene.physics.gravity = [0.0, 0.0, -3.0];
        scene.physics.iterations = 3;
        scene.entities[0].transform.translation = [0.0, 0.0, 5.0];
        scene.entities[0].components.insert(
            "Rigid Body".into(),
            serde_json::to_value(RigidBodyComponent::default()).unwrap(),
        );
        let mut runtime = Runtime::from_scene(&scene).unwrap();
        let entity = runtime.world.handles()[0];
        assert!(runtime.physics.contains(entity));
        assert_eq!(runtime.physics_settings.iterations, 3);
        assert_eq!(
            runtime.physics_settings.gravity.to_array(),
            scene.physics.gravity
        );
        runtime.advance(Duration::from_millis(100));
        assert!(runtime.world.get(entity).unwrap().transform.translation[2] < 5.0);
    }

    #[test]
    fn verity_scene_rejects_unsupported_and_invalid_physics() {
        let mut scene = Scene::default();
        for (body, expected) in [
            (
                serde_json::json!({ "shape": "Capsule", "half_height": 0.0 }),
                "half height",
            ),
            (serde_json::json!({ "restitution": -0.5 }), "restitution"),
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
