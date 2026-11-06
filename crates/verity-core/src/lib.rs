//! Serializable scene data and a dense runtime entity store.
//!
//! Runtime handles are specific to the world that issued them. They remain stable
//! across removal of other entities, but must not be used with another world.

pub mod component;

use std::{any::TypeId, collections::BTreeMap};

use serde::{Deserialize, Serialize};

use component::{AnyStore, Store};
pub use component::{Choice, Component, ComponentInfo, ComponentRegistry, FieldMut};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Transform {
    pub translation: [f32; 3],
    /// Quaternion in x, y, z, w order.
    pub rotation: [f32; 4],
    pub scale: [f32; 3],
}

impl Default for Transform {
    fn default() -> Self {
        Self {
            translation: [0.0; 3],
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [1.0; 3],
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Material {
    pub base_color: [f32; 3],
    pub metallic: f32,
    pub roughness: f32,
}

impl Default for Material {
    fn default() -> Self {
        Self {
            base_color: [0.8, 0.8, 0.8],
            metallic: 0.0,
            roughness: 0.5,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Entity {
    pub name: String,
    pub transform: Transform,
    pub material: Material,
    /// Authoring data of registered components, by [`Component::NAME`].
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub components: BTreeMap<String, serde_json::Value>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PointLight {
    pub position: [f32; 3],
    pub color: [f32; 3],
    pub intensity: f32,
    pub radius: f32,
}

/// A named viewpoint saved with the scene. `vertical_fov` is in degrees.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Camera {
    pub name: String,
    pub position: [f32; 3],
    pub target: [f32; 3],
    pub vertical_fov: f32,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            name: "Camera".into(),
            position: [6.0, -8.0, 5.0],
            target: [0.0; 3],
            vertical_fov: 45.0,
        }
    }
}

/// Directional sunlight. `direction` points in the direction light travels;
/// `color` is linear RGB and `intensity` scales its illuminance.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Sun {
    pub direction: [f32; 3],
    pub color: [f32; 3],
    pub intensity: f32,
}

impl Default for Sun {
    fn default() -> Self {
        Self {
            direction: [-0.3, -0.2, -1.0],
            color: [1.0, 0.95, 0.88],
            intensity: 3.0,
        }
    }
}

/// The lighting components used to illuminate a scene.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum LightingMethod {
    DirectOnly,
    #[default]
    DirectAndHemisphere,
    /// Split Radiance Cascades GI; the hemisphere lights the scene until it is ready.
    SplitRadianceCascades,
}

/// Scene-wide ambient lighting, expressed as linear RGB radiance.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Lighting {
    pub method: LightingMethod,
    pub sky: [f32; 3],
    pub ground: [f32; 3],
}

impl Default for Lighting {
    fn default() -> Self {
        Self {
            method: LightingMethod::default(),
            sky: [0.03, 0.035, 0.045],
            ground: [0.015; 3],
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum PhysicsEngine {
    #[default]
    Box3d,
    // Scenes saved before the engine was renamed.
    #[serde(alias = "Hyperon")]
    Verity,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ScenePhysics {
    pub engine: PhysicsEngine,
    pub gravity: [f32; 3],
    pub iterations: u32,
    pub substeps: u32,
}

impl Default for ScenePhysics {
    fn default() -> Self {
        Self {
            engine: PhysicsEngine::Box3d,
            gravity: [0.0, 0.0, -9.81],
            iterations: 6,
            substeps: 4,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Scene {
    pub entities: Vec<Entity>,
    pub lights: Vec<PointLight>,
    /// Absent in older scenes, which retain their original lighting.
    #[serde(default)]
    pub sun: Option<Sun>,
    #[serde(default)]
    pub lighting: Lighting,
    #[serde(default)]
    pub assets: Vec<String>,
    #[serde(default)]
    pub cameras: Vec<Camera>,
    #[serde(default)]
    pub physics: ScenePhysics,
}

impl Default for Scene {
    fn default() -> Self {
        Self {
            entities: vec![
                Entity {
                    name: "Copper".into(),
                    transform: Transform {
                        translation: [-1.2, 0.0, 0.0],
                        ..Transform::default()
                    },
                    material: Material {
                        base_color: [0.9, 0.4, 0.2],
                        metallic: 0.8,
                        roughness: 0.3,
                    },
                    ..Entity::default()
                },
                Entity {
                    name: "Ceramic".into(),
                    transform: Transform {
                        translation: [1.2, 0.0, 0.0],
                        ..Transform::default()
                    },
                    material: Material {
                        base_color: [0.2, 0.5, 0.9],
                        metallic: 0.0,
                        roughness: 0.45,
                    },
                    ..Entity::default()
                },
            ],
            lights: vec![PointLight {
                position: [2.0, 4.0, 3.0],
                color: [1.0, 0.95, 0.85],
                intensity: 30.0,
                radius: 15.0,
            }],
            sun: Some(Sun::default()),
            lighting: Lighting::default(),
            assets: Vec::new(),
            cameras: Vec::new(),
            physics: ScenePhysics::default(),
        }
    }
}

/// A generational identifier issued by a particular [`World`].
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct EntityHandle {
    index: u32,
    generation: u32,
}

impl EntityHandle {
    pub fn index(self) -> u32 {
        self.index
    }
    pub fn generation(self) -> u32 {
        self.generation
    }
}

#[derive(Debug)]
struct Slot {
    generation: u32,
    dense_index: Option<usize>,
}

/// Borrowed components of one entity.
pub struct EntityRef<'a> {
    pub handle: EntityHandle,
    pub name: &'a str,
    pub transform: &'a Transform,
    pub material: &'a Material,
}

/// Exclusively borrowed components of one entity.
pub struct EntityMut<'a> {
    pub handle: EntityHandle,
    pub name: &'a mut String,
    pub transform: &'a mut Transform,
    pub material: &'a mut Material,
}

/// Dense structure-of-arrays storage. Removal uses swap-remove and may change
/// iteration order; handles of surviving entities remain valid.
#[derive(Debug, Default)]
pub struct World {
    slots: Vec<Slot>,
    free: Vec<u32>,
    handles: Vec<EntityHandle>,
    names: Vec<String>,
    transforms: Vec<Transform>,
    materials: Vec<Material>,
    pub lights: Vec<PointLight>,
    pub sun: Option<Sun>,
    pub lighting: Lighting,
    pub assets: Vec<String>,
    pub cameras: Vec<Camera>,
    pub physics: ScenePhysics,
    stores: std::collections::HashMap<TypeId, Box<dyn AnyStore>>,
}

impl World {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn len(&self) -> usize {
        self.handles.len()
    }
    pub fn is_empty(&self) -> bool {
        self.handles.is_empty()
    }

    /// Inserts an entity and returns its stable runtime handle. Authoring
    /// components need a registry: add them with [`World::insert_component`]
    /// or build the world with [`World::from_scene`].
    pub fn spawn(&mut self, entity: Entity) -> EntityHandle {
        debug_assert!(
            entity.components.is_empty(),
            "World::spawn ignores authoring components"
        );
        let dense_index = self.handles.len();
        let index = match self.free.pop() {
            Some(index) => index,
            None => {
                let index = u32::try_from(self.slots.len()).expect("entity slot limit exceeded");
                self.slots.push(Slot {
                    generation: 0,
                    dense_index: None,
                });
                index
            }
        };
        let slot = &mut self.slots[index as usize];
        slot.dense_index = Some(dense_index);
        let handle = EntityHandle {
            index,
            generation: slot.generation,
        };
        self.handles.push(handle);
        self.names.push(entity.name);
        self.transforms.push(entity.transform);
        self.materials.push(entity.material);
        handle
    }

    fn dense_index(&self, handle: EntityHandle) -> Option<usize> {
        let slot = self.slots.get(handle.index as usize)?;
        (slot.generation == handle.generation)
            .then_some(slot.dense_index)
            .flatten()
    }

    pub fn contains(&self, handle: EntityHandle) -> bool {
        self.dense_index(handle).is_some()
    }

    pub fn get(&self, handle: EntityHandle) -> Option<EntityRef<'_>> {
        let index = self.dense_index(handle)?;
        Some(EntityRef {
            handle,
            name: &self.names[index],
            transform: &self.transforms[index],
            material: &self.materials[index],
        })
    }

    pub fn get_mut(&mut self, handle: EntityHandle) -> Option<EntityMut<'_>> {
        let index = self.dense_index(handle)?;
        Some(EntityMut {
            handle,
            name: &mut self.names[index],
            transform: &mut self.transforms[index],
            material: &mut self.materials[index],
        })
    }

    /// Removes and returns an entity, or returns None for a stale handle.
    /// Its typed components are dropped.
    pub fn despawn(&mut self, handle: EntityHandle) -> Option<Entity> {
        let index = self.dense_index(handle)?;
        for store in self.stores.values_mut() {
            store.remove_entity(handle);
        }
        self.handles.swap_remove(index);
        let entity = Entity {
            name: self.names.swap_remove(index),
            transform: self.transforms.swap_remove(index),
            material: self.materials.swap_remove(index),
            components: BTreeMap::new(),
        };
        if let Some(moved) = self.handles.get(index) {
            self.slots[moved.index as usize].dense_index = Some(index);
        }
        let slot = &mut self.slots[handle.index as usize];
        slot.dense_index = None;
        // Retire a slot rather than wrapping and resurrecting an old handle.
        if let Some(generation) = slot.generation.checked_add(1) {
            slot.generation = generation;
            self.free.push(handle.index);
        }
        Some(entity)
    }

    pub fn iter(&self) -> impl ExactSizeIterator<Item = EntityRef<'_>> + '_ {
        self.handles
            .iter()
            .copied()
            .zip(&self.names)
            .zip(&self.transforms)
            .zip(&self.materials)
            .map(|(((handle, name), transform), material)| EntityRef {
                handle,
                name,
                transform,
                material,
            })
    }

    pub fn iter_mut(&mut self) -> impl ExactSizeIterator<Item = EntityMut<'_>> + '_ {
        self.handles
            .iter()
            .copied()
            .zip(&mut self.names)
            .zip(&mut self.transforms)
            .zip(&mut self.materials)
            .map(|(((handle, name), transform), material)| EntityMut {
                handle,
                name,
                transform,
                material,
            })
    }

    /// Dense component slices, aligned by index with [`World::handles`].
    pub fn names(&self) -> &[String] {
        &self.names
    }
    pub fn transforms(&self) -> &[Transform] {
        &self.transforms
    }
    pub fn materials(&self) -> &[Material] {
        &self.materials
    }
    pub fn handles(&self) -> &[EntityHandle] {
        &self.handles
    }

    fn store<T: Component>(&self) -> Option<&Store<T>> {
        self.stores.get(&TypeId::of::<T>())?.as_any().downcast_ref()
    }

    fn store_mut<T: Component>(&mut self) -> Option<&mut Store<T>> {
        self.stores
            .get_mut(&TypeId::of::<T>())?
            .as_any_mut()
            .downcast_mut()
    }

    /// Attaches or replaces a component, returning the previous value. Returns
    /// `Err(value)` for a stale handle.
    pub fn insert_component<T: Component>(
        &mut self,
        entity: EntityHandle,
        value: T,
    ) -> Result<Option<T>, T> {
        if !self.contains(entity) {
            return Err(value);
        }
        let store = self
            .stores
            .entry(TypeId::of::<T>())
            .or_insert_with(|| Box::new(Store::<T>::default()));
        let store: &mut Store<T> = store.as_any_mut().downcast_mut().expect("store type");
        Ok(store.insert(entity, value))
    }

    pub fn remove_component<T: Component>(&mut self, entity: EntityHandle) -> Option<T> {
        self.store_mut::<T>()?.remove(entity)
    }

    pub fn component<T: Component>(&self, entity: EntityHandle) -> Option<&T> {
        let store = self.store::<T>()?;
        Some(&store.values[store.dense(entity)?])
    }

    pub fn component_mut<T: Component>(&mut self, entity: EntityHandle) -> Option<&mut T> {
        let store = self.store_mut::<T>()?;
        let dense = store.dense(entity)?;
        Some(&mut store.values[dense])
    }

    /// Every entity with a `T`, in the store's dense (unspecified) order.
    pub fn query<T: Component>(&self) -> impl Iterator<Item = (EntityHandle, &T)> {
        self.store::<T>()
            .into_iter()
            .flat_map(|s| s.owners.iter().copied().zip(&s.values))
    }

    pub fn query_mut<T: Component>(&mut self) -> impl Iterator<Item = (EntityHandle, &mut T)> {
        self.store_mut::<T>()
            .into_iter()
            .flat_map(|s| s.owners.iter().copied().zip(&mut s.values))
    }

    /// Builds a runtime world, resolving each entity's components through
    /// `registry`. Fails on unknown or malformed components.
    pub fn from_scene(scene: &Scene, registry: &ComponentRegistry) -> Result<Self, String> {
        let mut world = Self {
            lights: scene.lights.clone(),
            sun: scene.sun,
            lighting: scene.lighting,
            assets: scene.assets.clone(),
            cameras: scene.cameras.clone(),
            physics: scene.physics,
            ..Self::new()
        };
        for entity in &scene.entities {
            let handle = world.spawn(Entity {
                components: BTreeMap::new(),
                ..entity.clone()
            });
            for (name, value) in &entity.components {
                registry
                    .insert(&mut world, handle, name, value)
                    .map_err(|e| format!("{}: {e}", entity.name))?;
            }
        }
        Ok(world)
    }

    /// The scene of this world, with the components `registry` knows about.
    pub fn to_scene(&self, registry: &ComponentRegistry) -> Scene {
        Scene {
            entities: self
                .iter()
                .map(|entity| Entity {
                    name: entity.name.to_owned(),
                    transform: entity.transform.clone(),
                    material: entity.material.clone(),
                    components: registry.extract(self, entity.handle).collect(),
                })
                .collect(),
            lights: self.lights.clone(),
            sun: self.sun,
            lighting: self.lighting,
            assets: self.assets.clone(),
            cameras: self.cameras.clone(),
            physics: self.physics,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn physics_engine_reads_its_former_name() {
        let engine: PhysicsEngine = serde_json::from_str("\"Hyperon\"").unwrap();
        assert_eq!(engine, PhysicsEngine::Verity);
    }

    #[test]
    fn scene_physics_defaults_and_round_trips_through_world() {
        let mut old = serde_json::to_value(Scene::default()).unwrap();
        old.as_object_mut().unwrap().remove("physics");
        let mut scene: Scene = serde_json::from_value(old).unwrap();
        assert_eq!(scene.physics, ScenePhysics::default());
        scene.physics = ScenePhysics {
            engine: PhysicsEngine::Verity,
            gravity: [0.0, 1.0, -3.0],
            iterations: 8,
            substeps: 2,
        };
        let world = World::from_scene(&scene, &ComponentRegistry::default()).unwrap();
        assert_eq!(
            world.to_scene(&ComponentRegistry::default()).physics,
            scene.physics
        );
    }

    fn entity(name: &str) -> Entity {
        Entity {
            name: name.into(),
            transform: Transform::default(),
            material: Material::default(),
            ..Entity::default()
        }
    }

    #[test]
    fn removal_preserves_other_handles_and_rejects_stale_handles() {
        let mut world = World::new();
        let a = world.spawn(entity("a"));
        let b = world.spawn(entity("b"));
        let c = world.spawn(entity("c"));
        assert_eq!(world.despawn(b).unwrap().name, "b");
        assert_eq!(world.get(c).unwrap().name, "c");
        assert_eq!(world.get(a).unwrap().name, "a");
        assert!(world.get(b).is_none());
        assert!(world.get_mut(b).is_none());
        assert!(world.despawn(b).is_none());
        let d = world.spawn(entity("d"));
        assert_eq!(b.index(), d.index());
        assert_ne!(b.generation(), d.generation());
        assert!(!world.contains(b));
        assert_eq!(world.get(d).unwrap().name, "d");
    }

    #[test]
    fn dense_iteration_and_mutation_remain_aligned_after_removal() {
        let mut world = World::new();
        let first = world.spawn(entity("first"));
        let second = world.spawn(entity("second"));
        let third = world.spawn(entity("third"));
        world.despawn(first);
        for item in world.iter_mut() {
            item.transform.translation[0] = if item.handle == second { 2.0 } else { 3.0 };
            item.material.metallic = 1.0;
        }
        assert_eq!(world.get(second).unwrap().transform.translation[0], 2.0);
        assert_eq!(world.get(third).unwrap().transform.translation[0], 3.0);
        assert_eq!(world.iter().len(), 2);
        for (i, item) in world.iter().enumerate() {
            assert_eq!(world.handles()[i], item.handle);
            assert_eq!(world.names()[i], item.name);
            assert_eq!(&world.transforms()[i], item.transform);
            assert_eq!(&world.materials()[i], item.material);
        }
    }

    #[test]
    fn scene_json_and_world_round_trip() {
        let scene = Scene::default();
        let json = serde_json::to_string(&scene).unwrap();
        let decoded: Scene = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, scene);
        let world = World::from_scene(&decoded, &ComponentRegistry::default()).unwrap();
        assert_eq!(world.to_scene(&ComponentRegistry::default()), decoded);
        assert_eq!(world.to_scene(&ComponentRegistry::default()), decoded);
    }

    #[test]
    fn legacy_scene_has_no_sun() {
        let scene: Scene = serde_json::from_str(r#"{"entities":[],"lights":[]}"#).unwrap();
        assert_eq!(scene.sun, None);
        assert_eq!(
            World::from_scene(&scene, &ComponentRegistry::default())
                .unwrap()
                .sun,
            None
        );
        assert_eq!(
            World::from_scene(&scene, &ComponentRegistry::default())
                .unwrap()
                .to_scene(&ComponentRegistry::default())
                .sun,
            None
        );
    }

    #[test]
    fn legacy_scene_defaults_lighting_and_assets() {
        let scene: Scene = serde_json::from_str(r#"{"entities":[],"lights":[]}"#).unwrap();
        assert_eq!(scene.lighting, Lighting::default());
        assert_eq!(scene.assets, Vec::<String>::new());
        assert_eq!(scene.cameras, Vec::<Camera>::new());
        let world = World::from_scene(&scene, &ComponentRegistry::default()).unwrap();
        assert_eq!(world.to_scene(&ComponentRegistry::default()), scene);
        assert_eq!(world.to_scene(&ComponentRegistry::default()), scene);
    }

    #[test]
    fn partial_lighting_defaults_missing_fields() {
        let scene: Scene = serde_json::from_str(
            r#"{"entities":[],"lights":[],"lighting":{"method":"DirectOnly"}}"#,
        )
        .unwrap();
        assert_eq!(scene.lighting.method, LightingMethod::DirectOnly);
        assert_eq!(scene.lighting.sky, Lighting::default().sky);
        assert_eq!(scene.lighting.ground, Lighting::default().ground);
    }

    #[test]
    fn lighting_and_assets_survive_json_and_world_conversions() {
        let mut scene = Scene::default();
        scene.lighting = Lighting {
            method: LightingMethod::DirectOnly,
            sky: [0.2, 0.3, 0.4],
            ground: [0.05, 0.06, 0.07],
        };
        scene.assets = vec!["models/ship.glb".into(), "models/tree.gltf".into()];
        scene.cameras = vec![Camera {
            name: "Nave".into(),
            ..Camera::default()
        }];
        let decoded: Scene = serde_json::from_str(&serde_json::to_string(&scene).unwrap()).unwrap();
        assert_eq!(decoded, scene);
        let world = World::from_scene(&decoded, &ComponentRegistry::default()).unwrap();
        assert_eq!(world.to_scene(&ComponentRegistry::default()), decoded);
        assert_eq!(world.to_scene(&ComponentRegistry::default()), decoded);
    }

    #[test]
    fn sunlight_round_trips_with_custom_values_and_when_disabled() {
        let mut scene = Scene::default();
        scene.sun = Some(Sun {
            direction: [0.2, -0.8, 0.4],
            color: [0.5, 0.7, 1.0],
            intensity: 2.5,
        });
        for sun in [scene.sun, None] {
            scene.sun = sun;
            let json = serde_json::to_string(&scene).unwrap();
            let decoded: Scene = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded, scene);
            let world = World::from_scene(&decoded, &ComponentRegistry::default()).unwrap();
            assert_eq!(world.sun, sun);
            assert_eq!(world.to_scene(&ComponentRegistry::default()), decoded);
            assert_eq!(world.to_scene(&ComponentRegistry::default()), decoded);
        }
    }

    #[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
    struct Health {
        current: f32,
        regenerates: bool,
    }

    impl Component for Health {
        const NAME: &'static str = "Health";
        fn fields(&mut self) -> Vec<(&'static str, FieldMut<'_>)> {
            vec![
                ("Current", FieldMut::Float(&mut self.current, 0.0..=100.0)),
                ("Regenerates", FieldMut::Bool(&mut self.regenerates)),
            ]
        }
    }

    #[test]
    fn components_follow_their_entities_through_removal() {
        let mut world = World::new();
        let [a, b, c] = ["a", "b", "c"].map(|n| world.spawn(entity(n)));
        for (i, e) in [a, b, c].into_iter().enumerate() {
            let health = Health {
                current: i as f32,
                regenerates: false,
            };
            assert_eq!(world.insert_component(e, health), Ok(None));
        }
        world.despawn(a);
        assert_eq!(world.remove_component::<Health>(b).unwrap().current, 1.0);
        assert_eq!(world.component::<Health>(c).unwrap().current, 2.0);
        assert!(world.component::<Health>(a).is_none());
        assert_eq!(
            world.query::<Health>().map(|(e, _)| e).collect::<Vec<_>>(),
            [c]
        );
        // A recycled slot must not see the despawned entity's component.
        let d = world.spawn(entity("d"));
        assert_eq!(d.index(), a.index());
        assert!(world.component::<Health>(d).is_none());
        assert!(world.insert_component(a, Health::default()).is_err());
        world.component_mut::<Health>(c).unwrap().current = 5.0;
        assert_eq!(world.query_mut::<Health>().next().unwrap().1.current, 5.0);
    }

    #[test]
    fn registered_components_round_trip_and_unknown_ones_fail() {
        let mut registry = ComponentRegistry::default();
        registry.register::<Health>();
        let mut scene = Scene::default();
        scene.entities[1].components.insert(
            "Health".into(),
            serde_json::json!({ "current": 40.0, "regenerates": true }),
        );
        let json = serde_json::to_string(&scene).unwrap();
        let decoded: Scene = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, scene);
        let world = World::from_scene(&decoded, &registry).unwrap();
        let ceramic = world.handles()[1];
        assert_eq!(world.component::<Health>(ceramic).unwrap().current, 40.0);
        assert_eq!(world.to_scene(&registry), scene);

        // Inspecting through reflection edits the authoring value.
        let info = registry.get("Health").unwrap();
        let mut value = (info.default)();
        let changed = (info.inspect)(&mut value, &mut |fields| {
            let FieldMut::Float(current, _) = &mut fields[0].1 else {
                panic!()
            };
            **current = 7.0;
            true
        });
        assert_eq!(changed, Ok(true));
        assert_eq!(value["current"], 7.0);

        scene.entities[0]
            .components
            .insert("Mana".into(), serde_json::json!({}));
        assert!(World::from_scene(&scene, &registry)
            .unwrap_err()
            .contains("Mana"));
        scene.entities[0].components = [("Health".into(), serde_json::json!("x"))].into();
        assert!(World::from_scene(&scene, &registry).is_err());
    }

    #[test]
    fn exhausted_generation_retires_slot() {
        let mut world = World::new();
        let original = world.spawn(entity("old"));
        world.slots[original.index as usize].generation = u32::MAX;
        let last = EntityHandle {
            generation: u32::MAX,
            ..original
        };
        world.handles[0] = last;
        assert!(world.despawn(last).is_some());
        let new = world.spawn(entity("new"));
        assert_ne!(new.index(), last.index());
        assert!(!world.contains(last));
    }
}
