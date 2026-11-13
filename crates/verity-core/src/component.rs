//! Typed components with field reflection, and the registry that converts them
//! between a scene's authoring data (JSON by component name) and a [`World`]'s
//! sparse-set storage.

use std::{any::Any, collections::HashMap, fmt::Debug, ops::RangeInclusive};

use serde::{Serialize, de::DeserializeOwned};
use serde_json::Value;

use crate::{EntityHandle, World};

/// One editable field of a component, for generic inspectors.
pub enum FieldMut<'a> {
    Float(&'a mut f32, RangeInclusive<f32>),
    Vec3(&'a mut [f32; 3], RangeInclusive<f32>),
    Color(&'a mut [f32; 3]),
    Bool(&'a mut bool),
    Choice(&'a mut dyn Choice),
    /// Another entity, by name; empty for none.
    Entity(&'a mut String),
}

/// A field-less enum offered as a list of options.
pub trait Choice {
    fn options(&self) -> &'static [&'static str];
    fn index(&self) -> usize;
    fn select(&mut self, index: usize);
}

/// Data attached to entities. `NAME` keys it in scene files, so keep it stable.
pub trait Component: Clone + Debug + Default + Serialize + DeserializeOwned + 'static {
    const NAME: &'static str;
    /// The fields an inspector may edit, in display order.
    fn fields(&mut self) -> Vec<(&'static str, FieldMut<'_>)>;
}

/// Dense component values with their owners, indexed by entity slot.
#[derive(Debug)]
pub(crate) struct Store<T> {
    pub(crate) owners: Vec<EntityHandle>,
    pub(crate) values: Vec<T>,
    slots: Vec<Option<u32>>,
}

impl<T> Default for Store<T> {
    fn default() -> Self {
        Self {
            owners: Vec::new(),
            values: Vec::new(),
            slots: Vec::new(),
        }
    }
}

impl<T> Store<T> {
    pub(crate) fn dense(&self, entity: EntityHandle) -> Option<usize> {
        let dense = (*self.slots.get(entity.index() as usize)?)? as usize;
        (self.owners[dense] == entity).then_some(dense)
    }

    pub(crate) fn insert(&mut self, entity: EntityHandle, value: T) -> Option<T> {
        if let Some(dense) = self.dense(entity) {
            return Some(std::mem::replace(&mut self.values[dense], value));
        }
        let slot = entity.index() as usize;
        if self.slots.len() <= slot {
            self.slots.resize(slot + 1, None);
        }
        self.slots[slot] = Some(self.values.len() as u32);
        self.owners.push(entity);
        self.values.push(value);
        None
    }

    pub(crate) fn remove(&mut self, entity: EntityHandle) -> Option<T> {
        let dense = self.dense(entity)?;
        self.slots[entity.index() as usize] = None;
        self.owners.swap_remove(dense);
        if let Some(moved) = self.owners.get(dense) {
            self.slots[moved.index() as usize] = Some(dense as u32);
        }
        Some(self.values.swap_remove(dense))
    }
}

/// Type-erased store, so a world can drop an entity from every store.
pub(crate) trait AnyStore: Debug {
    fn remove_entity(&mut self, entity: EntityHandle);
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

impl<T: Debug + 'static> AnyStore for Store<T> {
    fn remove_entity(&mut self, entity: EntityHandle) {
        self.remove(entity);
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// Inspector callback: edit the fields and return whether any changed.
pub type Inspect<'f> = &'f mut dyn FnMut(&mut [(&'static str, FieldMut<'_>)]) -> bool;

/// Conversions for one registered component type.
pub struct ComponentInfo {
    pub name: &'static str,
    /// The authoring value of a newly added component.
    pub default: fn() -> Value,
    /// Edit an authoring value through the type's fields. Ok(true) if it changed.
    pub inspect: fn(&mut Value, Inspect) -> Result<bool, String>,
    insert: fn(&mut World, EntityHandle, &Value) -> Result<(), String>,
    extract: fn(&World, EntityHandle) -> Option<Value>,
}

fn decode<T: Component>(value: &Value) -> Result<T, String> {
    T::deserialize(value).map_err(|e| format!("{}: {e}", T::NAME))
}

fn encode<T: Component>(value: &T) -> Value {
    serde_json::to_value(value).expect("components serialize to JSON")
}

/// The component types a scene may use, by name.
#[derive(Default)]
pub struct ComponentRegistry {
    infos: Vec<ComponentInfo>,
    by_name: HashMap<&'static str, usize>,
}

impl ComponentRegistry {
    pub fn register<T: Component>(&mut self) -> &mut Self {
        assert!(
            !self.by_name.contains_key(T::NAME),
            "component {} registered twice",
            T::NAME
        );
        self.by_name.insert(T::NAME, self.infos.len());
        self.infos.push(ComponentInfo {
            name: T::NAME,
            default: || encode(&T::default()),
            inspect: |value, inspect| {
                let mut typed = decode::<T>(value)?;
                let changed = inspect(&mut typed.fields());
                if changed {
                    *value = encode(&typed);
                }
                Ok(changed)
            },
            insert: |world, entity, value| {
                world
                    .insert_component(entity, decode::<T>(value)?)
                    .map(drop)
                    .map_err(|_| format!("{}: entity is not in this world", T::NAME))
            },
            extract: |world, entity| world.component::<T>(entity).map(encode),
        });
        self
    }

    pub fn get(&self, name: &str) -> Option<&ComponentInfo> {
        self.by_name.get(name).map(|&i| &self.infos[i])
    }

    pub fn iter(&self) -> impl Iterator<Item = &ComponentInfo> {
        self.infos.iter()
    }

    pub(crate) fn insert(
        &self,
        world: &mut World,
        entity: EntityHandle,
        name: &str,
        value: &Value,
    ) -> Result<(), String> {
        let info = self
            .get(name)
            .ok_or_else(|| format!("unknown component {name:?}"))?;
        (info.insert)(world, entity, value)
    }

    pub(crate) fn extract<'a>(
        &'a self,
        world: &'a World,
        entity: EntityHandle,
    ) -> impl Iterator<Item = (String, Value)> + 'a {
        self.infos.iter().filter_map(move |info| {
            (info.extract)(world, entity).map(|v| (info.name.to_owned(), v))
        })
    }
}
