# Hyperon

`hyperon-render` provides the Vulkan renderer, including direct lighting and
shadow maps. Scene point lights and directional sunlight are usable without a
GI extension. `hyperon-lux` is an optional global-illumination extension; see
its [README](crates/hyperon-lux/README.md) for its current status.

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
`cargo run -p hyperon-editor`.

## Editor

The editor uses a dark interface and requests a dark native window theme. The
Scene outliner, Inspector, and project Assets browser are resizable around the
viewport.

- Select an entity in the hierarchy, choose Translate, Rotate, or Scale above
  the viewport, then drag a colored axis or rotation ring. Translation and
  rotation use world axes; scale uses the object's local axes. The inspector
  also accepts numeric transforms.
- Select Environment to choose direct-only or direct plus hemisphere ambient
  lighting and edit the sky and ground colors. Sun is directional; point lights
  can be added and edited separately. Lux GI is not an editor lighting option.
- The Asset Browser is rooted at `assets/`. Double-click folders to navigate,
  a `.hyperon` or `.json` scene to open it, or a `.glb` to import it. Imported
  paths are saved relative to the project and reloaded with the scene. Models
  must use the renderer's supported GLB/KTX2 format. Imported models are
  currently whole-scene geometry rather than editable entities; remove them
  from the Imported Models list in the hierarchy. Imported point lights remain
  active, while directional lighting is controlled by the editor's Sun setting.
