# hyperon_box3d

Box3D's native C17 rigid-body engine, statically linked into the Hyperon runtime. This crate uses `hyperon-core` and `glam`; it does not require a separate ECS or graphics backend.

## Build and run

Requires Rust, a C17 compiler (MSVC Build Tools on Windows, GCC/Clang on Unix), and the sibling `../box3d` source checkout. Cargo compiles the native library; CMake, bindgen, libclang, and runtime DLLs are unnecessary. Set `BOX3D_SOURCE_DIR` to use a compatible Box3D checkout elsewhere. Box3D is built in single precision.

```sh
cargo test -p hyperon_box3d
cargo run -p hyperon_box3d --example falling_box
```

Add a `PhysicsBody` to a live `hyperon_core::World` entity using `PhysicsWorld::insert`. A body has a `RigidBody` type, one `Collider`, and default material, velocity, force and impulse. Fields can be changed through `get_mut`. Run `step` from the engine's fixed-update loop, supplying a fixed, positive duration in seconds:

```rust,ignore
use glam::Vec3;
use hyperon_box3d::{Box3dSettings, Collider, PhysicsBody, PhysicsWorld, RigidBody};
use hyperon_core::{Entity, Material, Transform, World};

let mut world = World::new();
let entity = world.spawn(Entity {
    name: "cube".into(),
    transform: Transform { translation: [0.0, 4.0, 0.0], ..Default::default() },
    material: Material::default(),
});
let mut physics = PhysicsWorld::default();
physics.insert(&world, entity, PhysicsBody::new(
    RigidBody::Dynamic, Collider::Cuboid { half_extents: Vec3::splat(0.5) },
)).unwrap();
let settings = Box3dSettings::default();
physics.step(&mut world, &settings, 1.0 / 60.0);
```

## Behavior

- `Box3dSettings` controls gravity and solver substeps (default: 4); the caller owns the fixed-timestep scheduler and supplies the duration explicitly.
- Changes to position, rotation, velocity, body type, collider, or material synchronize before each step. Only bodies reported moved by Box3D are read back after the step. Transform edits teleport and wake the body; scale is visual only.
- `PhysicsBody::force` persists each step. `impulse` is applied once and zeroed. Zero forces and impulses do not wake sleeping bodies. `sync(&world)` applies body/transform edits and removes stale bodies without advancing time or consuming impulses.
- `remove` destroys the native body immediately. Despawned entities are removed on the next `sync` or `step`; stale generational handles cannot attach to recycled entities. Dropping the physics world releases native bodies and shapes. Entity handles are local to their `World`: recreate `PhysicsWorld` when replacing the runtime world, even if the new world happens to contain identical handles.
- Coordinates use meters, kilograms, seconds, and unit quaternions; transforms are world-space. Invalid input panics in Rust before native calls. `insert` rejects an entity that is not live in the supplied world.
- Each body supports one centered cuboid, sphere, Y-aligned capsule, `ConvexHull`, or `TriMesh`. Build hulls and meshes once and clone them to share Box3D's processed geometry. Box3D only creates mesh contacts on static bodies. Hull planes and mesh tree/triangles are available for renderers. `PhysicsWorld::iter` exposes entity/body pairs; resolve their poses from the same runtime world. Compound shapes, height fields, joints, sensors, collision events, spatial queries, interpolation, and debug rendering are not wrapped.
- The native engine uses one worker. Native calls are serialized across worlds because Box3D has shared global state; hull and mesh data is immutable after creation. The unsafe `ffi` module exposes the low-level adapter; its callers must ensure valid lifetimes and serialize their calls with all other Box3D calls. Safe applications should use the `PhysicsWorld` API.

The crate depends on the local Box3D checkout. Box3D retains its upstream MIT license in `../box3d/LICENSE`.
