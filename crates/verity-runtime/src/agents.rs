//! Agents in a running scene: the Crowd component spawns groups of them
//! (wanderers, flocks, hordes, fleeing herds, people walking to a place) on a
//! navmesh of the ground with static bodies cut out. A city's traffic and
//! pedestrians run alongside.

use std::sync::Arc;

use glam::{Quat, Vec2, Vec3};
use serde::{Deserialize, Serialize};
use verity_agents::{Crowd, Goal, Group};
use verity_core::{Choice, Component, EntityHandle, FieldMut, World};
use verity_nav::{FlowField, NavMesh};
use verity_physics::{RigidBody, RigidBodyComponent, Shape};

/// Ground height under a point of the plane.
pub type Height = dyn Fn(Vec2) -> Option<f32> + Send + Sync;

/// Navmesh cells, metres; and how far it reaches past every crowd and target.
const CELL: f32 = 1.0;
const MARGIN: f32 = 80.0;
const MAX_CELLS: f32 = 1024.0;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum Behaviour {
    Stand,
    #[default]
    Wander,
    /// Walk to points around the target entity, one after another.
    Visit,
    /// Every agent chases the target entity, around obstacles.
    Horde,
    /// Run from the target entity when it comes near.
    Flee,
}

impl Choice for Behaviour {
    fn options(&self) -> &'static [&'static str] {
        &["Stand", "Wander", "Visit target", "Horde (chase target)", "Flee target"]
    }
    fn index(&self) -> usize {
        *self as usize
    }
    fn select(&mut self, index: usize) {
        *self = [Self::Stand, Self::Wander, Self::Visit, Self::Horde, Self::Flee][index.min(4)];
    }
}

/// Spawns agents around its entity when the scene plays.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct CrowdComponent {
    pub behaviour: Behaviour,
    /// The entity a visiting, chasing or fleeing crowd watches.
    pub target: String,
    pub count: f32,
    /// How far from the entity agents spawn.
    pub spread: f32,
    pub speed: f32,
    pub radius: f32,
    pub separation: f32,
    /// Flocking: drawn together, and heading the same way.
    pub cohesion: f32,
    pub alignment: f32,
    pub color: [f32; 3],
}

impl Default for CrowdComponent {
    fn default() -> Self {
        Self {
            behaviour: Behaviour::Wander,
            target: String::new(),
            count: 50.0,
            spread: 10.0,
            speed: 1.4,
            radius: 0.3,
            separation: 1.0,
            cohesion: 0.0,
            alignment: 0.0,
            color: [0.85, 0.7, 0.3],
        }
    }
}

impl Component for CrowdComponent {
    const NAME: &'static str = "Crowd";

    fn fields(&mut self) -> Vec<(&'static str, FieldMut<'_>)> {
        vec![
            ("Behaviour", FieldMut::Choice(&mut self.behaviour)),
            ("Target", FieldMut::Entity(&mut self.target)),
            ("Count", FieldMut::Float(&mut self.count, 0.0..=100_000.0)),
            ("Spread", FieldMut::Float(&mut self.spread, 0.0..=10_000.0)),
            ("Speed", FieldMut::Float(&mut self.speed, 0.0..=50.0)),
            ("Radius", FieldMut::Float(&mut self.radius, 0.05..=10.0)),
            ("Separation", FieldMut::Float(&mut self.separation, 0.0..=10.0)),
            ("Cohesion", FieldMut::Float(&mut self.cohesion, 0.0..=5.0)),
            ("Alignment", FieldMut::Float(&mut self.alignment, 0.0..=5.0)),
            ("Color", FieldMut::Color(&mut self.color)),
        ]
    }
}

/// One Crowd component's group as it runs.
struct Spawned {
    group: u16,
    behaviour: Behaviour,
    target: Option<EntityHandle>,
    field: Option<u32>,
    /// Where the field last led to.
    solved_for: Option<Vec3>,
}

/// The agents of a running scene.
#[derive(Default)]
pub struct Agents {
    pub crowd: Crowd,
    /// Colour of each crowd group, for drawing.
    pub colors: Vec<[f32; 3]>,
    spawned: Vec<Spawned>,
    /// Where crowds and their targets started, which the navmesh covers.
    area: Option<(Vec2, Vec2)>,
}

impl Agents {
    /// Agents for the scene's Crowd components, standing on flat ground at
    /// z = 0 until [`Self::navigate`] is given the real ground.
    pub fn from_world(world: &World) -> Result<Self, String> {
        let mut agents = Self::default();
        let crowds: Vec<_> = world.query::<CrowdComponent>().map(|(e, c)| (e, c.clone())).collect();
        let mut area: Option<(Vec2, Vec2)> = None;
        let mut grow = |p: Vec3| {
            let p = p.truncate();
            area = Some(area.map_or((p, p), |(lo, hi)| (lo.min(p), hi.max(p))));
        };
        for (entity, c) in crowds {
            let object = world.get(entity).expect("entities come from this world");
            let name = object.name.to_owned();
            let at = Vec3::from_array(object.transform.translation);
            if !at.is_finite() || !c.speed.is_finite() || !c.radius.is_finite() || !c.spread.is_finite() {
                return Err(format!("{name}: Crowd: values must be finite"));
            }
            let target = if c.target.is_empty() {
                None
            } else {
                let found = (world.handles().iter().copied())
                    .find(|&h| world.get(h).is_some_and(|o| o.name == c.target))
                    .ok_or_else(|| format!("{name}: Crowd: no entity named {:?}", c.target))?;
                grow(Vec3::from_array(world.get(found).unwrap().transform.translation));
                Some(found)
            };
            if target.is_none() && matches!(c.behaviour, Behaviour::Visit | Behaviour::Horde | Behaviour::Flee) {
                return Err(format!("{name}: Crowd: this behaviour needs a target"));
            }
            grow(at - Vec3::splat(c.spread));
            grow(at + Vec3::splat(c.spread));
            let goal = match c.behaviour {
                Behaviour::Stand => Goal::Stand,
                Behaviour::Wander => Goal::Wander,
                Behaviour::Visit => Goal::Seek,
                Behaviour::Horde => Goal::Flow(agents.crowd.fields.len() as u32),
                Behaviour::Flee => Goal::Stand,
            };
            let group = agents.crowd.add_group(Group {
                goal,
                speed: c.speed.max(0.0),
                acceleration: (c.speed * 3.0).max(1.0),
                separation: c.separation.max(0.0),
                cohesion: c.cohesion.max(0.0),
                alignment: c.alignment.max(0.0),
                sight: (c.radius * 8.0).max(2.0),
                corridor: 0.0,
            });
            let field = (c.behaviour == Behaviour::Horde).then(|| {
                agents.crowd.fields.push(FlowField::new(Vec2::ZERO, CELL, [1, 1], |_| false));
                agents.crowd.fields.len() as u32 - 1
            });
            agents.colors.push(c.color);
            let count = c.count.clamp(0.0, 100_000.0) as u32;
            for k in 0..count {
                // A sunflower spiral fills the spread evenly.
                let r = c.spread * ((k as f32 + 0.5) / count as f32).sqrt();
                let angle = k as f32 * 2.399_963;
                let p = at + Vec3::new(r * angle.cos(), r * angle.sin(), 0.0);
                agents.crowd.spawn(group, p, c.radius.max(0.05));
            }
            agents.spawned.push(Spawned { group, behaviour: c.behaviour, target, field, solved_for: None });
        }
        agents.area = area;
        agents.navigate(world, None);
        Ok(agents)
    }

    /// Build the navmesh over `height` (flat at z = 0 when `None`), with the
    /// footprints of static bodies cut out, and stand everyone on it.
    pub fn navigate(&mut self, world: &World, height: Option<Arc<Height>>) {
        let Some((lo, hi)) = self.area else { return };
        let (lo, hi) = (lo - Vec2::splat(MARGIN), hi + Vec2::splat(MARGIN));
        let size = ((hi - lo) / CELL).ceil().min(Vec2::splat(MAX_CELLS)).as_uvec2();
        let clearance = self.crowd.radius.iter().copied().fold(0.3f32, f32::max);
        let obstacles: Vec<(Vec3, Quat, Shape, Vec3, f32)> = (world.query::<RigidBodyComponent>())
            .filter(|(_, b)| b.body_type == RigidBody::Static)
            .filter_map(|(e, b)| {
                let t = &world.get(e)?.transform;
                let scale = Vec3::from_array(t.scale).abs();
                Some((
                    Vec3::from_array(t.translation),
                    Quat::from_array(t.rotation).normalize(),
                    b.shape,
                    Vec3::from_array(b.half_extents) * scale,
                    b.radius * scale.max_element(),
                ))
            })
            .collect();
        let ground = height.clone();
        let mesh = NavMesh::from_heights(lo, CELL, size.to_array(), 0.6, move |p| {
            let z = match &ground {
                Some(h) => h(p)?,
                None => 0.0,
            };
            let blocked = obstacles.iter().any(|&(at, turn, shape, half, radius)| {
                // Only bodies reaching from below the feet to above the knees block.
                let local = turn.conjugate() * (p.extend(z + 0.5) - at);
                match shape {
                    Shape::Cuboid => (local.abs() - half).max_element() < clearance,
                    Shape::Sphere | Shape::Capsule => local.length() < radius + clearance,
                }
            });
            (!blocked).then_some(z)
        });
        let mesh = Arc::new(mesh);
        for spawn in &self.spawned {
            if let Some(f) = spawn.field {
                self.crowd.fields[f as usize] = FlowField::from_navmesh(&mesh, CELL);
            }
        }
        for s in &mut self.spawned {
            s.solved_for = None;
        }
        self.crowd.ground = height;
        self.crowd.set_navmesh((!mesh.is_empty()).then_some(mesh));
        for i in 0..self.crowd.len() {
            let p = self.crowd.position[i];
            if let Some((_, on)) = self.crowd.navmesh().and_then(|m| m.nearest(p, 20.0)) {
                self.crowd.position[i] = on;
            }
        }
    }

    /// Follow targets, then step every agent.
    pub fn step(&mut self, world: &World, dt: f32) {
        for s in &mut self.spawned {
            let Some(at) = s.target.and_then(|t| world.get(t)).map(|o| Vec3::from_array(o.transform.translation)) else {
                continue;
            };
            let group = &mut self.crowd.groups[s.group as usize];
            match s.behaviour {
                Behaviour::Horde => {
                    // Re-solve once the target has moved a cell.
                    if s.solved_for.is_none_or(|p| p.distance(at) > CELL)
                        && let Some(field) = s.field.and_then(|f| self.crowd.fields.get_mut(f as usize))
                    {
                        field.solve(&[at]);
                        s.solved_for = Some(at);
                    }
                }
                Behaviour::Flee => group.goal = Goal::Flee { from: at, radius: 25.0 },
                Behaviour::Visit if s.solved_for.is_none() => {
                    s.solved_for = Some(at);
                    for i in 0..self.crowd.len() as u32 {
                        if self.crowd.group[i as usize] == s.group {
                            self.crowd.set_target(i, around(at, i, 0));
                        }
                    }
                }
                _ => {}
            }
        }
        self.crowd.step(dt);
        let arrived = std::mem::take(&mut self.crowd.arrived);
        for (k, &i) in arrived.iter().enumerate() {
            let s = self.spawned.iter().find(|s| s.group == self.crowd.group[i as usize]);
            if let Some(at) = s.and_then(|s| s.target).and_then(|t| world.get(t)) {
                let at = Vec3::from_array(at.transform.translation);
                self.crowd.set_target(i, around(at, i, (self.crowd.time * 7.0) as u32 ^ k as u32));
            }
        }
    }
}

/// A point within 12 m of `at`, picked by agent and turn.
fn around(at: Vec3, agent: u32, turn: u32) -> Vec3 {
    let h = verity_agents::hash(agent, turn);
    let r = 12.0 * verity_agents::unit(h).sqrt();
    at + Vec2::from_angle(verity_agents::unit(verity_agents::hash(h, 1)) * std::f32::consts::TAU).extend(0.0) * r
}

#[cfg(test)]
mod tests {
    use super::*;
    use verity_core::{Entity, Transform};

    fn entity(world: &mut World, name: &str, at: [f32; 3]) -> EntityHandle {
        world.spawn(Entity {
            name: name.into(),
            transform: Transform { translation: at, ..Transform::default() },
            ..Entity::default()
        })
    }

    #[test]
    fn a_horde_goes_around_a_static_wall_to_its_target() {
        let mut world = World::new();
        let spawner = entity(&mut world, "Spawner", [-30.0, 0.0, 0.0]);
        entity(&mut world, "Player", [30.0, 0.0, 0.0]);
        let wall = entity(&mut world, "Wall", [0.0, 0.0, 1.0]);
        world.insert_component(
            wall,
            RigidBodyComponent { body_type: RigidBody::Static, half_extents: [1.0, 20.0, 2.0], ..Default::default() },
        )
        .unwrap();
        world.insert_component(
            spawner,
            CrowdComponent { behaviour: Behaviour::Horde, target: "Player".into(), count: 60.0, speed: 3.0, ..Default::default() },
        )
        .unwrap();
        let mut agents = Agents::from_world(&world).unwrap();
        assert_eq!(agents.crowd.len(), 60);
        for _ in 0..60 * 40 {
            agents.step(&world, 1.0 / 60.0);
        }
        let near = agents.crowd.position.iter().filter(|p| p.distance(Vec3::new(30.0, 0.0, 0.0)) < 10.0).count();
        assert!(near >= 50, "{near} of 60 reached the player");
        for p in &agents.crowd.position {
            assert!(!(p.x.abs() < 1.0 && p.y.abs() < 20.0), "an agent walked through the wall at {p}");
        }
    }

    #[test]
    fn a_crowd_without_its_target_is_an_error() {
        let mut world = World::new();
        let e = entity(&mut world, "Herd", [0.0; 3]);
        world.insert_component(e, CrowdComponent { behaviour: Behaviour::Flee, target: "Wolf".into(), ..Default::default() }).unwrap();
        assert!(Agents::from_world(&world).is_err());
    }
}
