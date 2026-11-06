//! Native Box3D rigid bodies synchronized with a Verity runtime world.

pub mod component;
pub mod ffi;

pub use component::{RigidBodyComponent, Shape};

use glam::{Quat, Vec3};
use std::{
    collections::HashMap,
    ffi::c_void,
    ptr::NonNull,
    sync::{Arc, Mutex},
};
use verity_core::{EntityHandle, Transform, World};

// ponytail: serialize native calls across worlds; use per-world locks once
// Box3D's global creation/allocator state supports concurrent access.
static NATIVE_LOCK: Mutex<()> = Mutex::new(());

/// Runtime world settings. The caller supplies the fixed timestep to [`PhysicsWorld::step`].
#[derive(Debug, Clone)]
pub struct Box3dSettings {
    pub gravity: Vec3,
    pub substeps: u32,
}

impl Default for Box3dSettings {
    fn default() -> Self {
        Self {
            gravity: Vec3::new(0.0, -9.81, 0.0),
            substeps: 4,
        }
    }
}

/// A body in world space. Transform scale is visual only; size the collider itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[repr(i32)]
pub enum RigidBody {
    #[default]
    Static = 0,
    Kinematic = 1,
    Dynamic = 2,
}

/// One centered collider per body, in meters. Changes replace the native shape
/// without replacing the body or losing its velocity.
#[derive(Debug, Clone, PartialEq)]
pub enum Collider {
    Cuboid {
        half_extents: Vec3,
    },
    Sphere {
        radius: f32,
    },
    /// Y-aligned capsule. `half_height` is half the distance between cap centers.
    Capsule {
        radius: f32,
        half_height: f32,
    },
    /// A convex hull in body space; build it once and clone it to share it.
    ConvexHull(ConvexHull),
    /// A triangle mesh in body space. Box3D only creates mesh contacts on
    /// static bodies, so use it for level geometry.
    TriMesh(TriMesh),
}

impl Collider {
    /// The native shape kind, its sizes, and its shared geometry (null for the
    /// analytic shapes).
    fn native(&self) -> (i32, [f32; 3], *const c_void) {
        let (kind, size) = match *self {
            Self::ConvexHull(ref hull) => return (3, [0.0; 3], hull.0.native.as_ptr()),
            Self::TriMesh(ref mesh) => return (4, [0.0; 3], mesh.0.native.as_ptr()),
            Self::Cuboid { half_extents } => {
                assert!(
                    half_extents.is_finite() && half_extents.min_element() > 0.0,
                    "cuboid half extents must be finite and positive"
                );
                (0, half_extents.to_array())
            }
            Self::Sphere { radius } => {
                assert!(
                    radius.is_finite() && radius > 0.0,
                    "sphere radius must be finite and positive"
                );
                (1, [radius, 0.0, 0.0])
            }
            Self::Capsule {
                radius,
                half_height,
            } => {
                assert!(
                    radius.is_finite()
                        && radius > 0.0
                        && half_height.is_finite()
                        && half_height > 0.0,
                    "capsule radius and half height must be finite and positive"
                );
                (2, [radius, half_height, 0.0])
            }
        };
        (kind, size, std::ptr::null())
    }
}

/// Box3D's processed hull or mesh, created once and read-only afterwards, with
/// copies of what a renderer needs to trace it.
struct Geometry {
    native: NonNull<c_void>,
    mesh: bool,
    lower: Vec3,
    upper: Vec3,
    planes: Vec<[f32; 4]>,
    nodes: Vec<[f32; 4]>,
    triangles: Vec<[f32; 4]>,
}

// SAFETY: the native data is never written after creation; Box3D only reads it
// (under NATIVE_LOCK), and it is destroyed once, when the last owner drops.
unsafe impl Send for Geometry {}
unsafe impl Sync for Geometry {}

impl Drop for Geometry {
    fn drop(&mut self) {
        let _guard = NATIVE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // SAFETY: created by the matching constructor; shapes referencing a mesh
        // hold this Arc and are destroyed before it drops.
        unsafe {
            if self.mesh {
                ffi::bb_mesh_destroy(self.native.as_ptr());
            } else {
                ffi::bb_hull_destroy(self.native.as_ptr());
            }
        }
    }
}

impl std::fmt::Debug for Geometry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct(if self.mesh { "TriMesh" } else { "ConvexHull" })
            .field("lower", &self.lower)
            .field("upper", &self.upper)
            .field("planes", &self.planes.len())
            .field("triangles", &(self.triangles.len() / 3))
            .finish()
    }
}

/// A convex hull computed by Box3D (at most 128 vertices). Cloning shares it;
/// colliders compare hulls by identity.
#[derive(Clone, Debug)]
pub struct ConvexHull(Arc<Geometry>);

impl PartialEq for ConvexHull {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl ConvexHull {
    /// The hull of `points` (body space, metres); `None` when they are not
    /// finite or span no volume.
    pub fn new(points: &[Vec3]) -> Option<Self> {
        if points.len() < 4
            || points.len() > i32::MAX as usize
            || !points.iter().all(|p| p.is_finite())
        {
            return None;
        }
        let _guard = NATIVE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // SAFETY: Vec3 is three packed f32; the count matches the slice.
        let native = NonNull::new(unsafe {
            ffi::bb_hull_create(points.as_ptr().cast(), points.len() as i32)
        })?;
        let (mut lower, mut upper) = ([0.0; 3], [0.0; 3]);
        // SAFETY: a live hull; output buffers are sized from its face count.
        let planes = unsafe {
            ffi::bb_hull_bounds(native.as_ptr(), lower.as_mut_ptr(), upper.as_mut_ptr());
            let count = ffi::bb_hull_planes(native.as_ptr(), std::ptr::null_mut(), 0);
            let mut planes = vec![[0.0; 4]; count.max(0) as usize];
            ffi::bb_hull_planes(native.as_ptr(), planes.as_mut_ptr().cast(), count);
            planes
        };
        Some(Self(Arc::new(Geometry {
            native,
            mesh: false,
            lower: Vec3::from_array(lower),
            upper: Vec3::from_array(upper),
            planes,
            nodes: Vec::new(),
            triangles: Vec::new(),
        })))
    }

    /// Face planes: unit normal and offset, `dot(normal, p) <= offset` inside.
    pub fn planes(&self) -> &[[f32; 4]] {
        &self.0.planes
    }

    /// Body-space bounds.
    pub fn bounds(&self) -> (Vec3, Vec3) {
        (self.0.lower, self.0.upper)
    }
}

/// A triangle mesh with Box3D's bounding volume tree. Cloning shares it;
/// colliders compare meshes by identity.
#[derive(Clone, Debug)]
pub struct TriMesh(Arc<Geometry>);

impl PartialEq for TriMesh {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl TriMesh {
    /// A mesh of counter-clockwise triangles over `vertices` (body space,
    /// metres); `None` when vertices aren't finite, an index is out of range, or
    /// every triangle is degenerate.
    pub fn new(vertices: &[Vec3], indices: &[[u32; 3]]) -> Option<Self> {
        let fits = |n: usize| n <= i32::MAX as usize;
        if vertices.len() < 3
            || indices.is_empty()
            || !fits(vertices.len())
            || !fits(indices.len())
            || !vertices.iter().all(|v| v.is_finite())
            || indices
                .iter()
                .flatten()
                .any(|&i| i as usize >= vertices.len())
        {
            return None;
        }
        let _guard = NATIVE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // SAFETY: validated counts and in-range indices; Box3D copies the input.
        let native = NonNull::new(unsafe {
            ffi::bb_mesh_create(
                vertices.as_ptr().cast(),
                vertices.len() as i32,
                indices.as_ptr().cast(),
                indices.len() as i32,
            )
        })?;
        let (mut node_count, mut triangle_count) = (0, 0);
        // SAFETY: a live mesh; output buffers are sized from its own counts.
        let (nodes, triangles) = unsafe {
            ffi::bb_mesh_counts(native.as_ptr(), &mut node_count, &mut triangle_count);
            let mut nodes = vec![[0.0; 4]; 2 * node_count.max(0) as usize];
            let mut triangles = vec![[0.0; 4]; 3 * triangle_count.max(0) as usize];
            ffi::bb_mesh_export(
                native.as_ptr(),
                nodes.as_mut_ptr().cast(),
                triangles.as_mut_ptr().cast(),
            );
            (nodes, triangles)
        };
        let root = |i: usize| {
            nodes
                .get(i)
                .map_or(Vec3::ZERO, |n| Vec3::new(n[0], n[1], n[2]))
        };
        Some(Self(Arc::new(Geometry {
            native,
            mesh: true,
            lower: root(0),
            upper: root(1),
            planes: Vec::new(),
            nodes,
            triangles,
        })))
    }

    /// Box3D's tree, two vec4 a node: lower bound (w: the right child, or a
    /// leaf's first triangle) and upper bound (w: 0 for an inner node, else the
    /// leaf's triangle count), indices as `f32::from_bits`. The left child
    /// follows its parent.
    pub fn nodes(&self) -> &[[f32; 4]] {
        &self.0.nodes
    }

    /// Triangles in leaf order, three corners (vec4, w unused) each.
    pub fn triangles(&self) -> &[[f32; 4]] {
        &self.0.triangles
    }

    /// Body-space bounds.
    pub fn bounds(&self) -> (Vec3, Vec3) {
        (self.0.lower, self.0.upper)
    }
}

/// Density in kg/m³, friction, and restitution. Density must be positive;
/// friction nonnegative; restitution in [0, 1]. All must be finite.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PhysicsMaterial {
    pub density: f32,
    pub friction: f32,
    pub restitution: f32,
}

impl Default for PhysicsMaterial {
    fn default() -> Self {
        Self {
            density: 1000.0,
            friction: 0.6,
            restitution: 0.0,
        }
    }
}

impl PhysicsMaterial {
    fn native(self) -> [f32; 3] {
        assert!(
            self.density.is_finite() && self.density > 0.0,
            "density must be finite and positive"
        );
        assert!(
            self.friction.is_finite() && self.friction >= 0.0,
            "friction must be finite and nonnegative"
        );
        assert!(
            self.restitution.is_finite() && (0.0..=1.0).contains(&self.restitution),
            "restitution must be in [0, 1]"
        );
        [self.density, self.friction, self.restitution]
    }
}

/// World-space linear velocity (m/s) and angular velocity (rad/s). Changes are
/// sent to Box3D before stepping, then updated from the simulation afterward.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Velocity {
    pub linear: Vec3,
    pub angular: Vec3,
}

/// Persistent force (N) at the center of mass and torque (N·m), in world space.
/// Set the force to zero to stop applying it.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ExternalForce {
    pub force: Vec3,
    pub torque: Vec3,
}

/// One-shot linear (N·s) and angular (N·m·s) impulses, in world space.
/// Automatically zeroed after applying them.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ExternalImpulse {
    pub linear: Vec3,
    pub angular: Vec3,
}

/// Physics properties associated with one Verity entity.
#[derive(Debug, Clone, PartialEq)]
pub struct PhysicsBody {
    pub body_type: RigidBody,
    pub collider: Collider,
    pub material: PhysicsMaterial,
    pub velocity: Velocity,
    pub force: ExternalForce,
    pub impulse: ExternalImpulse,
}

impl PhysicsBody {
    pub fn new(body_type: RigidBody, collider: Collider) -> Self {
        Self {
            body_type,
            collider,
            material: PhysicsMaterial::default(),
            velocity: Velocity::default(),
            force: ExternalForce::default(),
            impulse: ExternalImpulse::default(),
        }
    }
}

/// An entity is not live in the supplied world.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InvalidEntity;

impl std::fmt::Display for InvalidEntity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("entity is not live in this world")
    }
}
impl std::error::Error for InvalidEntity {}

/// Owns a native world and entity-associated physics data. Use each instance
/// with one `World` only: handles are world-local, so recreate the physics world
/// when replacing the runtime world.
pub struct PhysicsWorld {
    id: u32,
    bodies: HashMap<EntityHandle, PhysicsBody>,
    native: HashMap<EntityHandle, NativeBody>,
    entities: HashMap<u64, EntityHandle>,
    moved: Vec<u64>,
}

impl Default for PhysicsWorld {
    fn default() -> Self {
        let _guard = NATIVE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // SAFETY: creation is serialized and the gravity array is valid.
        let id = unsafe { ffi::bb_world_create([0.0, -9.81, 0.0].as_ptr()) };
        assert_ne!(id, 0, "Box3D could not create a world");
        Self {
            id,
            bodies: HashMap::new(),
            native: HashMap::new(),
            entities: HashMap::new(),
            moved: Vec::new(),
        }
    }
}

impl PhysicsWorld {
    pub fn body_count(&self) -> usize {
        self.bodies.len()
    }
    pub fn contains(&self, entity: EntityHandle) -> bool {
        self.bodies.contains_key(&entity)
    }
    pub fn get(&self, entity: EntityHandle) -> Option<&PhysicsBody> {
        self.bodies.get(&entity)
    }
    pub fn get_mut(&mut self, entity: EntityHandle) -> Option<&mut PhysicsBody> {
        self.bodies.get_mut(&entity)
    }
    pub fn iter(&self) -> impl Iterator<Item = (EntityHandle, &PhysicsBody)> {
        self.bodies.iter().map(|(&entity, body)| (entity, body))
    }

    /// Associate physics with a live entity, replacing existing properties.
    pub fn insert(
        &mut self,
        world: &World,
        entity: EntityHandle,
        body: PhysicsBody,
    ) -> Result<Option<PhysicsBody>, InvalidEntity> {
        if !world.contains(entity) {
            return Err(InvalidEntity);
        }
        Ok(self.bodies.insert(entity, body))
    }

    /// Remove physics and immediately destroy the corresponding native body.
    pub fn remove(&mut self, entity: EntityHandle) -> Option<PhysicsBody> {
        let body = self.bodies.remove(&entity)?;
        if let Some(native) = self.native.remove(&entity) {
            self.entities.remove(&native.id);
            {
                let _guard = NATIVE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
                // SAFETY: the body belongs to this live world and is removed once.
                unsafe { ffi::bb_body_destroy(native.id) };
            }
            // Geometry must drop outside the native lock.
            drop(native);
        }
        Some(body)
    }

    /// Reconcile bodies, transforms and physics properties without advancing time
    /// or consuming impulses. Despawned entities and native bodies are removed.
    pub fn sync(&mut self, world: &World) {
        for (&entity, body) in &self.bodies {
            if let Some(item) = world.get(entity) {
                body.collider.native();
                body.material.native();
                state(item.transform, &body.velocity);
                assert!(
                    body.force.force.is_finite() && body.force.torque.is_finite(),
                    "force and torque must be finite"
                );
                assert!(
                    body.impulse.linear.is_finite() && body.impulse.angular.is_finite(),
                    "impulses must be finite"
                );
            }
        }
        self.bodies.retain(|entity, _| world.contains(*entity));
        let mut stale = Vec::new();
        {
            let _guard = NATIVE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
            // SAFETY: IDs belong to this world, calls are serialized, and inputs
            // were validated above; geometry remains alive through shape removal.
            unsafe {
                let entities = &mut self.entities;
                self.native.retain(|entity, body| {
                    if self.bodies.contains_key(entity) {
                        return true;
                    }
                    ffi::bb_body_destroy(body.id);
                    entities.remove(&body.id);
                    stale.push(body.collider.clone());
                    false
                });
                for (&entity, body) in &self.bodies {
                    let input = state(world.get(entity).unwrap().transform, &body.velocity);
                    let (kind, size, geometry) = body.collider.native();
                    let material = body.material.native();
                    let native = self.native.entry(entity).or_insert_with(|| {
                        let id = ffi::bb_body_create(self.id, body.body_type as i32, &input);
                        assert_ne!(id, 0, "Box3D could not create a body");
                        self.entities.insert(id, entity);
                        NativeBody {
                            id,
                            shape: 0,
                            body_type: body.body_type,
                            collider: body.collider.clone(),
                            material: body.material,
                            state: input,
                        }
                    });
                    if native.body_type != body.body_type {
                        ffi::bb_body_set_type(native.id, body.body_type as i32);
                        native.body_type = body.body_type;
                    }
                    if native.shape == 0
                        || native.collider != body.collider
                        || native.material != body.material
                    {
                        if native.shape != 0 {
                            ffi::bb_shape_destroy(native.shape);
                            native.shape = 0;
                        }
                        native.shape = if geometry.is_null() {
                            ffi::bb_shape_create(native.id, kind, size.as_ptr(), material.as_ptr())
                        } else {
                            ffi::bb_shape_create_geometry(
                                native.id,
                                kind,
                                geometry,
                                material.as_ptr(),
                            )
                        };
                        assert_ne!(native.shape, 0, "Box3D could not create a collider");
                        stale.push(std::mem::replace(
                            &mut native.collider,
                            body.collider.clone(),
                        ));
                        native.material = body.material;
                    }
                    let mut flags = 0;
                    if input.position != native.state.position
                        || input.rotation != native.state.rotation
                    {
                        flags |= 1;
                    }
                    if input.linear_velocity != native.state.linear_velocity {
                        flags |= 2;
                    }
                    if input.angular_velocity != native.state.angular_velocity {
                        flags |= 4;
                    }
                    if flags != 0 {
                        ffi::bb_body_set_state(native.id, &input, flags);
                    }
                    native.state = input;
                }
            }
        }
        drop(stale);
    }

    /// Synchronize, advance by one positive fixed timestep in seconds, then
    /// write moved transforms and velocities back to the runtime and bodies.
    pub fn step(&mut self, world: &mut World, settings: &Box3dSettings, dt: f32) {
        assert!(settings.gravity.is_finite(), "gravity must be finite");
        assert!(
            settings.substeps > 0 && settings.substeps <= i32::MAX as u32,
            "substeps must fit a positive C int"
        );
        assert!(
            dt.is_finite() && dt > 0.0,
            "physics timestep must be finite and positive"
        );
        self.sync(world);
        let _guard = NATIVE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // SAFETY: synchronization established live IDs and validated every input.
        unsafe {
            for (&entity, body) in &mut self.bodies {
                let native = &self.native[&entity];
                if body.force.force != Vec3::ZERO || body.force.torque != Vec3::ZERO {
                    ffi::bb_body_apply(
                        native.id,
                        body.force.force.to_array().as_ptr(),
                        body.force.torque.to_array().as_ptr(),
                        0,
                    );
                }
                if body.impulse.linear != Vec3::ZERO || body.impulse.angular != Vec3::ZERO {
                    ffi::bb_body_apply(
                        native.id,
                        body.impulse.linear.to_array().as_ptr(),
                        body.impulse.angular.to_array().as_ptr(),
                        1,
                    );
                    body.impulse = ExternalImpulse::default();
                }
            }
            ffi::bb_world_step(
                self.id,
                settings.gravity.to_array().as_ptr(),
                dt,
                settings.substeps as i32,
            );
            let count = ffi::bb_world_moved(self.id, std::ptr::null_mut(), 0);
            self.moved.resize(count.max(0) as usize, 0);
            ffi::bb_world_moved(self.id, self.moved.as_mut_ptr(), count);
            for id in &self.moved {
                let Some(&entity) = self.entities.get(id) else {
                    continue;
                };
                let Some(native) = self.native.get_mut(&entity) else {
                    continue;
                };
                let Some(item) = world.get_mut(entity) else {
                    continue;
                };
                ffi::bb_body_get_state(native.id, &mut native.state);
                item.transform.translation = native.state.position;
                item.transform.rotation = native.state.rotation;
                if let Some(body) = self.bodies.get_mut(&entity) {
                    body.velocity = Velocity {
                        linear: Vec3::from_array(native.state.linear_velocity),
                        angular: Vec3::from_array(native.state.angular_velocity),
                    };
                }
            }
        }
    }
}

impl Drop for PhysicsWorld {
    fn drop(&mut self) {
        let _guard = NATIVE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // SAFETY: this instance exclusively owns the live world.
        unsafe { ffi::bb_world_destroy(self.id) };
    }
}

struct NativeBody {
    id: u64,
    shape: u64,
    body_type: RigidBody,
    collider: Collider,
    material: PhysicsMaterial,
    state: ffi::State,
}

fn state(transform: &Transform, velocity: &Velocity) -> ffi::State {
    let position = Vec3::from_array(transform.translation);
    let rotation = Quat::from_array(transform.rotation);
    assert!(
        position.is_finite()
            && rotation.is_finite()
            && (rotation.length_squared() - 1.0).abs() < 0.001,
        "physics position must be finite and rotation must be a unit quaternion"
    );
    assert!(
        velocity.linear.is_finite() && velocity.angular.is_finite(),
        "physics velocities must be finite"
    );
    ffi::State {
        position: transform.translation,
        rotation: transform.rotation,
        linear_velocity: velocity.linear.to_array(),
        angular_velocity: velocity.angular.to_array(),
    }
}
