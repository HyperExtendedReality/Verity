# Verity Engine

Verity, also known as Verity Engine, is a Rust game engine.

`verity-render` provides the Vulkan renderer, including direct lighting and
shadow maps. Scene point lights and directional sunlight are usable without a
GI extension. `verity-lux` is an optional global-illumination extension; see
its [README](crates/verity-lux/README.md) for its current status.

The renderer supports up to 1024 point lights. The first four in submission
order (imported glTF lights, then scene/world lights) cast shadows using six
512×512 depth maps each; later point lights still illuminate but do not cast
shadows. Point-shadow filtering does not cross cube-face edges, and casters
within 0.1% of a light's radius of its position are clipped. Sun shadows use
four cascades and fade out at the shadow-distance limit.

`Scene::sun` and `World::sun` override a renderer-configured or imported glTF
sun. When absent, the configured/imported sun is used instead. A zero direction
uses straight down; sunlight color is linear RGB multiplied by its intensity.

The editor stores lighting in the scene alongside its entities. Existing scene
files without a sun retain their original lighting when opened.

Building the renderer requires Vulkan and the Slang compiler (`slangc` on
`PATH`, or the `SLANGC` environment variable). Run the editor with
`cargo run -p verity-editor`.

## Terra terrain

`verity-terra` generates eroded landforms and keeps them as an editable 3D
density field, so digging makes real cavities and overhangs. Fly over one and
dig into it with:

```powershell
cargo run --release -p verity-lux --example terrain -- --landform island --seed 7 --gi pxtgi
```

Landforms are `valley`, `island`, `archipelago`, `continent`, `mountains` and
`plains`, and `--gi` is `none`, `lux` or `pxtgi`. Click to capture the mouse, then use WASD to fly, Q/E to descend and
rise, Shift to go faster, the left and right buttons to dig and fill, F to throw
a box, and the wheel for brush size. Set `VERITY_SMOKE` to drop boxes and dig a
trench under them through the real renderer, report, and exit.

```rust
let terrain = Terrain::new(Landscape::generate(Settings {
    seed: 42,
    ..Landform::Continent.into()
}));
```

The world is 384 × 384 m at 1 m voxels, Z-up, from −48 to 144 m. Relief is
eroded by stream-power river incision, droplets and thermal slumping, and a seed
reproduces it exactly on any thread count. `Terrain::brush` edits the field and
returns dirty chunks; `Mesher` turns them into Surface Nets meshes on worker
threads and rejects results an edit has overtaken.

Chunks draw through `Renderer::set_retained_mesh`, which keeps vertices on the
GPU until they change, so a frame pays only for the indices of visible chunks.
`GpuMaterial::terrain` shades them procedurally in the resolve pass
(`shaders/lib/terrain.slang`), from a mesh whose texture coordinates hold the
original ground height and wetness. Landscapes want a longer shadow reach than
interiors: see `Renderer::set_shadow_distance`.

GI methods follow terrain through `RenderExtension::retained_changed`: the
renderer batches edits and hands over only what changed, at most twice a second,
so bounced light can trail a brush stroke by up to half a second. A new method
implements that hook and, for a hit on a `TERRAIN_SURFACE` material, calls
`terrain_albedo` from `shaders/lib/terrain.slang`, so bounced light matches the
ground on screen. Split Radiance Cascades and PXTGI both do, updating only the
changed chunks in place. With Lux, use `Settings::landscape`: its defaults are
sized for rooms, leave its budgets uncapped and give it no sky. A method that
only adds diffuse light can return true from `RenderExtension::sky_reflections`
to keep sky reflections on glossy surfaces such as water.

Bodies collide with terrain as a field rather than a mesh, so an edit applies at
once and nothing can tunnel through. Give one static body `Collider::Ground` and
hand physics a snapshot, which is cheap as chunks are shared until edited:

```rust
physics.insert(&world, ground, PhysicsBody::new(RigidBody::Static, Collider::Ground))?;
physics.set_ground(Arc::new(terrain.clone()), Bounds::EMPTY);
// After digging, the same with the region that changed: bodies there wake.
physics.set_ground(Arc::new(terrain.clone()), changed);
```

Any shape implementing `verity_physics::Ground` works the same way. Rigid and
soft bodies both rest on it.

There is no distance LOD, streaming or
saving, and Surface Nets does not guarantee manifold topology for every
ambiguous sampled field.

## Matrix virtual geometry

`verity-matrix` cooks every static material primitive into a continuous LOD. In
the editor, open **View → Enable Matrix** to draw through it; **View → Matrix LOD
debug view** colors surfaces by tier: green, blue and yellow for cluster levels,
orange for voxels, purple for imposters. Both are editor-session settings, not
scene data. `Renderer::set_matrix_enabled` is the same switch in code.

There are three tiers, chosen on the GPU each frame by `matrix_select.slang`, and
the CPU records a fixed number of indirect draws however large the scene:

- **Clusters.** A DAG built with meshoptimizer's `clusterlod`, cut at about one
  pixel of error.
- **Voxels.** Simplifying foliage removes whole leaves: a 6000-leaf test canopy
  keeps 0.2% of its area. A mesh that loses more than 95% of its area counts as an
  aggregate, and its clusters that thinned are rebuilt from the original triangles
  as bricks of 4x4x4 voxels, as Nanite Foliage does, and marched in the fragment
  shader. Each voxel is kept with the probability its contents cover it, chosen by
  a stable hash, so density survives (0.625 opacity against 0.653 for the source in
  the test canopy) without per-pixel noise. A voxel of scattered leaves shades as
  one leaf picked from them, which keeps their lighting statistics.
- **Imposters.** `Renderer::place_copies` plants copies of a model. An aggregate
  placed this way gets a 12x12 hemi-octahedral atlas of color, normal and depth,
  baked on the GPU, and below 64 pixels across each copy is one card blending three
  frames with parallax. Between 80 and 64 pixels geometry dissolves into the card
  through a stable dither, each tier taking complementary pixels. The atlas stores a mean normal, so lighting uses the average
  a spread of facets would give (`Surface::agreement`) rather than a cosine of the
  mean.

All tiers write the same visibility buffer, with the tier in the top bits of the
ID, so resolve, shadows and the G-buffer treat them alike. Shadows use the same
selection, except that a copy drawn as an imposter casts its coarsest cut, and each
cascade and point-light face draws only the clusters that reach it.

`cargo run --release -p verity-render --example forest -- --debug` shows the tiers.
On an RTX 4070 Laptop at 1600x900, 2500 trees of 20000 leaves take 1.7 ms against
209 ms drawing every triangle, and New Sponza 8 ms against 63 ms, within 0.3/255
of the exact image.

Cooking runs at load, about 40 s for New Sponza. Primitives of 100000 triangles or
more are cached in the system temp folder, keyed by content, which cuts that to
5 s; bump `COOK_VERSION` when cooking changes.

Limits: geometry is static and fully resident, with no streaming or wind. Voxels
and imposters use a material's scalar roughness and no normal map. Clusters are
frustum culled but not occlusion culled. Selection lists have fixed capacities;
`Renderer::matrix_stats` shows how much was drawn.

The renderer uses GPU frustum culling and, when supported, a current-frame
reverse-Z minimum-depth pyramid for occlusion. The depth prepass draws all
casters in the camera view before the visibility pass, so occlusion does not
avoid the prepass geometry cost. Unsupported culling resources fall back to
conservative drawing. The CPU meshlet scan preserves input triangle order and
winding, with up to 64 vertices and 124 triangles per meshlet.

LOD error is meshoptimizer's approximate metric, not a guaranteed pixel or
geometric error bound. Geometry bounds and LODs are for static meshes; skin,
morph, or displacement needs conservative updated data. Vertices remain
fully GPU-resident; index paging and upload are synchronous. This is not
continuous cluster LOD or asynchronous device-local geometry streaming, and
performance depends on content and hardware.

Design references: [meshoptimizer](https://meshoptimizer.org/) and
[NVIDIA's cluster LOD generation](https://github.com/nvpro-samples/vk_lod_clusters/blob/main/docs/lod_generation.md).

## Physics performance

`verity-physics` solves contact forces and body motion with AVBD. Its solver
combines each deferred dual update with the next body visit and reuses the
contact error for that body's solve; the last dual update still runs at the end
of the step for warm starting. Velocities are taken before a final stabilization
sweep, so correcting penetration adds no energy. Speculative contacts, built where
a body's predicted path meets another, stop fast bodies at surfaces, so they do
not tunnel through thin walls. Hard joints push along their step-start direction,
as SHAKE does, so a pendulum keeps its swing; they turn to backward Euler's
damping once bodies turn fast, where the undamped form is unstable. Run
`cargo run --release -p verity-physics --example quality` to compare accuracy
with closed-form results and Box3D.

Joints give way like their material. A Joint component picks a preset (steel,
wood, rubber, bone, ligament) or custom values, and can limit rotation per axis.
Past its yield load a joint stretches or bends for good while hardening, and it
breaks at its break load or after its ductility. `PhysicsWorld::joint_events`
reports both, `joint_status` reports the load, and the editor shows it while
playing. `assets/scenes/joints.verity` is a demonstration. A joint can also have a
Hill muscle: what it cannot hold falls on the limits, so a tensed arm survives what
breaks a limp one.

Soft Body components add cloth, solved as finite elements and tearing where stress
exceeds its strength, and solids of any shape that stay dented past yield or crack
past their limit. A shell pinned around a rigid core crushes like a car's body,
calibrated to crash tests. Soft bodies collide with themselves and each other.
`assets/scenes/soft-bodies.verity` shows them;
`cargo run --release -p verity-render --example soft_bodies` runs it.

Within each graph color, independent body systems are assembled in place before
writing poses, then solved together with SIMD LDLᵀ. Runtime detection selects
eight-body batches and batched quaternion updates on x86-64 CPUs with AVX2, and
a build of the same 256-bit code with AVX-512's 32 registers where available.
Other CPUs retain the four-body path. Every path gives bit-identical results.
Partial batches keep the
four-body/singleton paths, with singletons using the Schur solve. Contact
regularization is prepared once per step; a separate unregularized penetration
flag preserves the sleep decision.

The CPU path packs broadphase candidates four boxes per SIMD block, caches
step-start orientation matrices for collision detection and contact anchors,
and shares relative-orientation projections across box SAT axes. Only a new
minimum-penetration SAT candidate needs normalization, and face clipping
alternates two polygon buffers instead of copying each intermediate polygon.
When a clipped face has more than four candidates, reduction computes each
distance and signed area once.
Single-contact manifolds use spatial SIMD for matrix assembly; multi-contact
manifolds use contact-lane SIMD and assemble the cross block directly from
reduced rows. Manifold warm-start state stays in those lanes, and matching
features in the same order retain their existing lanes.
Runtime AVX2 detection also stamps two of a body's multi-contact manifolds per
register, with results bit-identical to the scalar path. Manifolds store world
anchors rather than angular Jacobians to stay small enough for cache.
If collision wakes a sleeping island, warm-start preparation waits until the
second narrowphase pass so forces and penalties are scaled only once. Equal
static and kinetic friction coefficients share the cone test, whose lane
masks pass directly to the dual update. As in the paper's friction
experiments, cone-clamped tangential forces use the full penalty in the
Hessian; normal rows retain stiffness rescaling. Contact damping projects only
onto the normal row. This conservative approximation can converge differently
when contacts slide.

Solver pair hashes fold high bits into low bucket bits. Single-threaded steps
run directly without a one-worker Rayon pool, reuse the broadphase pair buffer
and coloring scratch space, and append narrowphase hits without temporary
manifold-sized slots for rejected pairs. Shape mass properties and rotated
inertia are cached until their inputs change. A fully sleeping world skips
solver preparation when its bodies and inputs are unchanged.

The default remains six iterations. These optimizations do not change the
collision margin, four-contact limit, damping, or sleep thresholds.
Floating-point reordering can change pile trajectories. These changes are not
a general performance win over Box3D: distinguish the benchmark's first
60 active steps from its full 300-step average, which includes settling.

Run `cargo run --release -p verity-physics --example compare` to compare
against Box3D. Set `RAYON_NUM_THREADS=1` for a single-thread comparison.
Use `VERITY_SCENE=pyramid`, `rain`, `slide`, `free_slide`, or `heavy` to focus a workload, and
`VERITY_ITERATIONS` and `BOX3D_SUBSTEPS` to experiment with solver settings. Lower counts
can change stability and sleep behavior; compare simulation quality as well
as elapsed time before changing the default. Each timing is the median of five
runs for that metric; the printed top and lowest heights come from the run
with the median full-step time, which may differ from the median active-time run.
The final floor penetration and RMS speed are snapshots. The heavy-stack scene
also checks final cube overlap; these checks do not measure transient instability.
See [CPU physics experiments](crates/verity-physics/PERFORMANCE.md) for
attempted changes and their results.

For a local benchmark on the build machine, `RUSTFLAGS="-C target-cpu=native"`
may improve code generation. The resulting binary can require CPU features
unavailable on other machines; do not distribute it as a portable build.

## Shore breakers

`verity-ocean::ShoreWaves` supplies a CPU-queryable shore-wave field and a
parametric breaking profile. `breaker_grid()` borrows the precomputed fields;
`profile()` supplies the same profile coefficients used for CPU queries.
Upload both to `verity-render` with `Renderer::set_breakers`, then advance
`ShoreWaves::set_time` and `Renderer::set_breaker_time` with the same time.
Re-upload the grid and profile after changing either. The renderer copies the
grid on upload; advancing time does not upload it again.

Run `cargo run --release -p verity-render --example shore_breakers` for a
Vulkan beach scene. The separate breaker pass can coexist with a render
extension and masks dry land. It renders an opaque, stylized water surface
with crest foam; it does not yet composite FFT ocean cascades, simulate spray,
or contribute to the earlier G-buffer, shadows, or GI pass. Near overturning
crests, the CPU height query selects an approximate single branch; a
heightfield cannot represent every visible part of a folded wave.

## Editor

The editor uses a dark interface and requests a dark native window theme. The
Scene outliner, Inspector, and project Assets browser are resizable around the
viewport.

- Select an entity in the hierarchy, choose Translate, Rotate, or Scale above
  the viewport, then drag a colored axis or rotation ring. Translation and
  rotation use world axes; scale uses the object's local axes. The inspector
  also accepts numeric transforms.
- Select Environment & Physics to choose direct-only or direct plus hemisphere ambient
  lighting and edit the sky and ground colors. Sun is directional; point lights
  can be added and edited separately.
- Open `assets/scenes/physics-playground.verity` from the Assets browser and
  press Play to test Verity physics with a pyramid and a falling pile. Select
  Environment & Physics to change gravity and solver settings.
  Select an object to add or edit its Rigid Body component. Collider dimensions
  do not follow visual scale. The editor simulates every scene with Verity, which
  does not support capsules or restitution. **View → Uncap FPS** stops waiting for
  vsync and draws continuously, which may tear.
- The Asset Browser is rooted at `assets/`. Double-click folders to navigate,
  a `.verity` or `.json` scene to open it, or a `.glb` to import it. Imported
  paths are saved relative to the project and reloaded with the scene. Models
  must use the renderer's supported GLB/KTX2 format. Imported models are
  currently whole-scene geometry rather than editable entities; remove them
  from the Imported Models list in the hierarchy. Imported point lights remain
  active, while directional lighting is controlled by the editor's Sun setting.
