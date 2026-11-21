# Verity Lux

`verity-lux` is Verity's global illumination: several methods behind one
extension, chosen per scene. Direct lighting, shadow maps, and the fallback
hemisphere ambient belong to `verity-render`; they do not require Lux.

| Method | Module | Traces |
| --- | --- | --- |
| Split Radiance Cascades | `cascades` | A GI scene of its own |
| PXTGI | `pxtgi` | The physics scene |
| Screen and voxel GI | `voxel` | The depth buffer, then a voxel copy of the scene |
| Residual GI | `voxel` | The voxels' settled light, corrected along the depth buffer |
| Ultra PBR / MSGI (experimental) | `cascades` | Material-weighted rays against Split's surface cache |
| Sliced Holographic Radiance Cascades (experimental) | `voxel` / `holographic` | Radiance through one voxel clipmap level, or through a standalone volume |

Install `Lux` with `Renderer::set_extension` before loading the geometry that
should take part:

```rust
use verity_lux::{Lux, Method, cascades::settings::Settings};

let method = Method::SplitRadianceCascades(Settings::budgeted());
renderer.set_extension(Box::new(Lux::new(method)))?;
renderer.load_gltf(path, 512)?;

// Each frame, after stepping physics. Methods that do not need it ignore it.
renderer.extension_mut::<Lux>().unwrap().sync(&physics, &world);
renderer.draw_world(&world)?;
```

`Lux::for_lighting` takes a scene's `LightingMethod` instead. In the editor, pick
a method under **Scene lighting**: models reload behind the change, because a
method only learns about geometry loaded after it starts. Reach what only one
method offers through `Lux::method_mut`.

Split Radiance Cascades and PXTGI trace with Vulkan hardware ray queries where
the GPU has them, and otherwise walk their own BVH. `hardware_rays` in either
method's settings, or `Lux::set_hardware_rays` while running, picks software
tracing on any GPU. PXTGI switches from the next frame. Split recompiles its passes
and shows the hemisphere until they are ready, as it does when it starts. In
`examples/sponza_gi.rs`, `--software` starts in software and H switches.

### Hardware tiers

`tier::HardwareProfile` reads the GPU the first time the renderer lends it to
Lux and chooses the acceleration structures for that device.
`Lux::hardware_profile` reports the result. `Lux::for_hardware()` also picks
the GI method from it, as an alternative to naming one:

| Tier | Devices | Rays | GI from `for_hardware` |
| --- | --- | --- | --- |
| `HardwareExtended` | Ray queries plus NVIDIA cluster or partitioned structures, AMD DGF, or opacity micromaps (RTX 40, RDNA 3 and later) | Hardware | Split |
| `Hardware` | Ray queries with core Vulkan structures only (RTX 20/30, RDNA 2, Arc, ray-traced mobile GPUs) | Hardware | Split |
| `Software` | Desktop GPUs without ray queries | Lux's own four-wide BVH | Split |
| `Low` | Mobile vendors, software rasterizers, or less than 2 GiB of device memory, without ray queries | No triangle BVH | Screen and voxel GI, SDF visibility over hashed bricks |

Split uses budgeted settings on integrated and low-memory GPUs. The device
class (discrete, integrated, low) is reported separately from the tier, so a
phone with ray queries traces in hardware and still gets the smaller budget.

Scenes choose this with `LightingMethod::HardwareGi`, listed in the editor as
"GI for this GPU": `Lux::for_lighting` turns it into `Lux::for_hardware()`,
and `Lux::status_line` names what the tier chose. Its Split presets are sized
for indoor probe spacing; large outdoor scenes such as terrain run far faster
on `Settings::landscape()`, so name Split with those settings for them.

Static meshes get the compressed four-wide software BLAS. Retained meshes
(terrain chunks) keep uncompressed binary nodes: packing them four-wide made
software GI 8–20% slower on an RTX 4070 and no faster on an RDNA 3 iGPU.

Every choice can be forced with an environment switch: `1` turns it on where the
device supports it, `0` turns it off, and leaving it unset keeps the profile's
choice. A switch never enables an extension the device lacks.

| Switch | Effect |
| --- | --- |
| `VERITY_LUX_CLUSTER_BLAS` | NVIDIA triangle cluster builds for static geometry |
| `VERITY_LUX_PARTITIONED_TLAS` | NVIDIA partitioned TLAS with updates to changed instance slots |
| `VERITY_LUX_DGF_BLAS` | AMD DGF builds for exactly encodable triangle positions |
| `VERITY_LUX_OPACITY_MICROMAPS` | Vulkan opacity micromaps for provable alpha coverage |
| `VERITY_LUX_OPAQUE_RAYS` | Direct hardware commits for eligible opaque two-sided geometry |
| `VERITY_LUX_ORIENTED_BLAS` | Selective oriented child bounds in the software BLAS |
| `VERITY_LUX_SPATIAL_BLAS` | Static spatial splits that duplicate references, not geometry |

Every switch is off by default, on every tier: on Sponza the vendor builds made
GI slower on an RTX 4070, spatial splits gained a few percent of software GI for
several seconds of extra build time per model, and oriented bounds made it
slower. See `docs/validation.md`. They do not change PXTGI. `VERITY_LUX_ACCEL_DIAGNOSTICS=1`
prints the profile and reports which backends were actually selected and
the number of BLASes for each.

### Acceleration and sampling

Static Split meshes pack a compressed four-child BLAS for software GPU
traversal. The TLAS and editable meshes keep their binary layout. Native
construction dispatches AVX2 only on supported x86 CPUs; the portable builder
produces the same split decisions. The software layout applies to meshes loaded
after the profile is read; on ray-traced tiers it stays compact, since software
tracing there only serves comparisons.

For `--voxel` and `--residual`, `--voxel-visibility=dda|sdf|roma` selects diffuse
visibility, and `--voxel-storage=dense|hash|dag` independently selects the
empty-brick lookup. `dda` and `dense` are the defaults. ROMA snaps diffuse
rays to precomputed occupancy directions; direct shadows keep their existing
trace. Sparse storage accelerates empty-space checks; it does not replace the
fine radiance storage.

CPU tools can use `cascades::kd::KdTree` for finite-segment geometric queries
against static scenes; it does not apply render materials. `cage` provides
reusable tetrahedral rest-pose geometry and independently deformable CPU ray
queries, not raster animation or automatic cage authoring.

Split's `--reference` uses [NILE](https://research.nvidia.com/labs/rtr/publication/ahmed2026nile/)
to sample directions, scattering lobes, emitters, and Russian roulette in
stable slots for each bounce. The sampling sequence changes, not the PDFs,
MIS weights, or ray budget. Reference startup compiles only its own passes;
the cascade transport passes compile when switching back to that mode.

<details>
<summary>Sobol direction number license</summary>

Copyright (c) 2008, Frances Y. Kuo and Stephen Joe  
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice,
  this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.
* Neither the names of the copyright holders nor the names of the
  University of New South Wales and the University of Waikato and its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

</details>

To add a method, implement `GiMethod`, add a `Method` variant and a row in
`METHODS`, and give it a `LightingMethod` so scenes can name it. The editor lists
whatever `METHODS` holds. Follow terrain through
`RenderExtension::retained_changed`.

Both methods trace retained meshes such as `verity-terra` chunks, which may be
set before or after the extension is installed. See the Terra section of the
root README.

The Verity Sponza example runs without GI by default; add `--gi` to enable it.

## Split Radiance Cascades

Each frame traces a budget of receivers into a world cache, merges cascades from
far to near, and packs directional probes that every pixel reads. Three things
keep the cost of a frame down:

- `reduce` lists the entries that survive, so `merge` touches only those instead
  of scanning the whole pool once per level.
- Entries with no new sample merge every fourth frame. Any change to lights or
  geometry merges them all.
- `seed` writes only the fields descriptors use, not whole 240-byte entries.

Split's deterministic diffuse projection and gather should preserve a unit
constant Lambertian field regardless of surface normal. The gather combines
angular and spatial weighted sums, normalizes each active light source by its
own weight sum, and keeps confidence separate from the light estimate.

`Settings::multi_bounce`, on by default, carries light past the first bounce: a
ray's hit reflects the irradiance last frame's probes hold there. Each bounce
takes about one trip through the receivers, and hits outside every probe, off
screen, stay at one bounce.

`Settings::surface_cache`, on by default, keeps outgoing light on the surfaces
themselves. Each glTF mesh is cut into islands of nearly flat, connected
triangles (`cascades/surface.rs`). Each instance packs its islands into an atlas
of `surface_texel` (0.3 m) texels. `surfaceUpdate` shades `surface_updates`
texels a frame with direct light and `surface_bounce_rays` (2) rays that read other texels, so
bounces build up across the scene, on screen or not. A ray hit reads its
texel's bounce light, where it used to read world probes, which leak through
walls. From `surface_from_level` (3) on, a hit takes all of its light from the
texel and skips shading. Blended and double-sided surfaces such as foliage are
not charted, and neither are retained meshes such as terrain. A hit on one
claims a hashed cell instead (`surface_hash_cells`, 65,536 by default): a cell
covers 0.3 m out to 10 m from the camera, twice that to 30 m, and so on, and
each side of a surface gets its own, so a leaf's two sides never share light.
`surfaceUpdate` shades `surface_hash_updates` (8,192) cells a frame like atlas
texels, and drops a cell no hit has read for 240 frames. Bounce rays that hit
these surfaces now bring their light instead of being left out. On the Bistro
street view at 1920×1080 (RTX 4070 Laptop, runs alternating off and on) the
cells add 0.22–0.28 ms to `surfaceUpdate` and 0.15–0.31 ms to the GI total.

Surface-cache bounces take consecutive samples from a fixed-point R2 sequence
on each scheduled texel or hash-slot visit. A different phase per slot avoids
sampling every surface in the same direction; indexing by visits keeps the
sequence dense even when a slot is touched only every few frames, without
changing the cosine weighting. `Settings::budgeted` uses one bounce ray per
visit; `Settings::default` and `Settings::landscape` keep two. Set
`surface_bounce_rays` to two in the budgeted preset if startup flicker matters
more than the cost of the surface-cache pass.

In the editor, open **Scene Lighting → Method → Experimental GI** to try these
presets, including empty-brick skipping for Screen and voxel GI or Residual GI.
Choosing a preset saves it with the scene; the standard methods are unchanged.

Experimental Split options in `sponza_gi`:

- `--surface-cold-priority` lends rays from settled surface slots to new or
  invalidated ones in the same GPU wave, without raising the bounce budget.
  Settled slots still get periodic refreshes. This is opt-in.
- `--surface-bounce-guide` proposes directions from last frame's cascade guide
  for the existing surface-cache bounce rays. It keeps a cosine component and
  weights the result by the mixture PDF. Probe interpolation at receivers is
  unchanged.
- `--emitter-low-discrepancy` changes the eight final-gather emissive candidates
  from hash draws to a shifted 3D R3 sequence (triangle selection and two
  area coordinates). The light PDF and reference tracer are unchanged.
- `--gather=guided` replaces probe interpolation at the receiver with a traced
  diffuse gather. It mixes cosine sampling with an eight-bin incident-light
  guide built from the previous frame's Split data, then reads the existing
  surface cache at hits. `--gather=cosine` is the unguided control;
  `--gather=probes` is the default. Benchmark the whole GI pipeline, including
  `packProbes` and `guidedResolve`, before choosing the guided mode.

Materials with `KHR_materials_diffuse_transmission` pass that share of their
diffuse light through: lights behind a leaf shine through it in the
transmission color, both in the renderer's direct light and at GI hits, and at
hits the bounce light in the far side's cell passes through too. The
extension's textures are not read yet.

A ray that leaves the scene brings the scene's ambient hemisphere
(`Lighting::sky` and `ground`, or `Renderer::set_ambient`), as PXTGI's and the
voxel method's do. The renderer drops that hemisphere while GI is on, so this
is the only sky light a surface gets. A cache texel takes it through its bounce
rays and a hit reads it from its texel or the probes; a hit neither covers
traces one sky ray.

Emissive materials (glTF `emissiveFactor`, `KHR_materials_emissive_strength`
and `emissiveTexture`) glow in the renderer and light the scene through GI.
The brightest `MAX_EMITTERS` (4096) emissive triangles go in a light list,
picked in proportion to area times emission. Each shaded pixel draws eight a
frame and keeps one in proportion to its unshadowed light there (resampled
importance sampling), behind one shadow ray (`worldResolve`); each ray hit samples one; probe rays
leave their emission out so it is not counted twice. Emitters past the cap
reach the scene only through rays that hit them. Emitted light shares the background group's history, so
an emitter that changes needs the scene reloaded. PXTGI and the voxel method
see emitters only on screen. The reference (`Settings::reference`) samples one
emitter at every path vertex, the shaded pixel included, and weighs it against
paths that hit emitters with the balance heuristic (multiple importance
sampling, Veach 1997, chapter 9).

Against the path-traced reference (RTX 4070 Laptop, 1280×720, 600 frames).
Error compares the mean of the last four frames with the reference's; single
frames swing by up to 0.14. These rows predate sky light and the bench's fix
for TAA jitter: see [Sky light and a full jitter cycle](#sky-light-and-a-full-jitter-cycle).

| View | Cache | GI time | Mean bounced light | Error | Flicker |
| --- | --- | --- | --- | --- | --- |
| Bistro street, `--scene=BistroExterior.glb --eye=-2,-13,2 --target=40,60,2.5` | off | 4.68 ms | 0.60× | 0.424 | 0.0037 |
| | on | 5.13 ms | 0.72× | 0.329 | 0.0048 |
| Sponza corridor | off | 3.09 ms | 0.77× | 0.397 | 0.0105 |
| | on | 3.30 ms | 1.09× | 0.378 | 0.0050–0.0147 |
| Sponza courtyard, `--eye=4,0,1.7 --target=-4,1.5,2.5` | off | 3.91 ms | 0.91× | 0.249 | 0.0068 |
| | on | 4.29 ms | 0.97× | 0.251 | 0.0066 |

Tuning moved nothing past run-to-run noise (about ±0.01 error; corridor
flicker varies threefold between identical runs). Texels of 0.15 or 0.5 m,
twice the updates, four bounce rays, or reading the cache from cascade 2 all
landed within it, and the last two cost 0.2–0.4 ms. Shading bounce hits on
uncharted surfaces, instead of leaving them out, brought the corridor to
1.00× and 0.354 but Bistro to 0.345 for 0.3 ms more, so it was dropped.

Sponza charts its 3.7 million triangles into 133 thousand islands and 485
thousand texels: about 31 MB of texels and 59 MB of per-triangle chart words.

`examples/sponza_gi.rs --bench=DIR` prints GPU time per pass and saves frames as
PFM. In code, `SplitRadianceCascades::profile` and `request_capture` do the same.
`Settings::budgeted` spends its budget on a cache that keeps probes: a 4-pixel
candidate grid and half-size resolve lost nothing on Sponza, and what they save
buys 2048 updates a frame and about a million cached entries. The cache is what
sets its accuracy.

### Ultra PBR / MSGI (experimental)

Choose **Scene lighting → Method → Experimental GI → Ultra PBR / MSGI**, or run:

```sh
cargo run --release -p verity-lux --example sponza_gi -- --gi --gather=msgi --eye=4,0,1.7 --target=-4,1.5,2.5
```

MSGI draws one ray per resolve texel from an energy-weighted cosine/GGX mixture
in the visible material's shading frame. It samples the stronger lobe more
often, advances the direction on a blue-noise-shifted R2 sequence, and divides
the full diffuse and glossy response by the same mixture PDF used for emissive
MIS. It shares Split's surface cache and emitter candidate resampling; eight
candidates still cost at most one visibility ray. When the secondary-hit cache
is cold, Split's diffuse probes replace the missing transport, but lighting
from listed emitters keeps the same paired estimate. For a material receiver,
emitter resampling weights each candidate by the full BSDF rather than the
cosine alone, so glossy surfaces pick the emitters their specular lobe sees.

The resolve stores its BSDF-weighted light divided by the anchor's
view-independent reflectance (`materialSpaceAlbedo`: diffuse albedo under the
dielectric layer plus F0). The composite multiplies by each full-resolution
pixel's own reflectance. The half-resolution resolve and its temporal and
spatial filters can then share light across texture detail without blurring
it. Rough surfaces accept a wider base-color difference between filter taps
than glossy ones.

Direct lighting is unchanged. MSGI is an opaque, cached final gather, not a
full glossy multi-bounce path tracer or a new material format. Use
`--gather=cosine` for the one-ray diffuse control, or `--gather=probes` for
ordinary Split. For an uncached material-weighted reference, run
`--reference --gather=msgi`; plain `--reference` keeps the Lambertian
receiver used to compare diffuse-probe methods.

#### Screen history for traced gathers

MSGI, the cosine gather and the cascade-guided gather trace one ray per
resolve texel a frame, so they depend on screen history far more than the
probes do. Three things keep it:

- The temporal pass reconstructs last frame's texels through last frame's own
  jittered projection. Through this frame's jitter, as the renderer's motion
  matrix provides it, each old point landed up to a pixel off. On grazing
  surfaces (floors, arch soffits) it then failed the plane test, and history
  was discarded every frame: at a still camera only 35% of Sponza's texels
  kept 8 or more frames, against 86% now.
- A traced gather runs at least three à-trous passes and lets each texel's
  history length alone set its blend, up to `history_frames`.
- A light change resets a texel's history only when its neighbourhood mean,
  not its single ray, moved by more than three times that mean's noise. While
  the camera moves, MSGI shortens history only on glossy materials.

Sponza at 640×360, RTX 4070 Laptop, still camera, 16 frames after 96 warm-up
frames. Noise is the mean per-pixel difference of one frame's bounce light
from the 16-frame mean, as a fraction of the mean bounce light; the block
column averages 8×8 pixels first. Error is the 8×8 block error of the 16-frame
mean against the material reference (MSGI) or the diffuse reference (others).

| Gather | Noise (before → after) | 8×8 block noise | Error |
| --- | ---: | ---: | ---: |
| Cosine | 0.50 → 0.20 | 0.295 → 0.095 | 0.149 |
| Cascade-guided | 0.50 → 0.20 | 0.292 → 0.095 | 0.149 |
| MSGI | 0.37 → 0.16 | 0.158 → 0.057 | 0.255 → 0.249 |
| Split probes | 0.074 → 0.071 | 0.024 → 0.021 | 0.207 |

MSGI's bounce light is still about 0.78× its material reference's, while
Split's is 0.87× and the cosine gather's 0.90× of the diffuse reference; most
of the material reference's extra 17% over the diffuse one is not reproduced.

For an A/B capture, hold the render size, camera, scene, AA mode and warm-up
fixed. The Desktop AA jitter has 16 phases; average all 16 saved still images
instead of comparing individual frames. Compare MSGI with the material-weighted
reference and Split probes with the diffuse reference. Their lighting models
are different, so the resulting errors do not rank the two methods directly.

```sh
cargo run --release -p verity-lux --example sponza_gi -- --bench=tmp/msgi-run --gather=msgi --size=640x360 --warm=96 --eye=4,0,1.7 --target=-4,1.5,2.5
cargo run --release -p verity-lux --example sponza_gi -- --bench=tmp/msgi-ref --reference --gather=msgi --size=640x360 --warm=300 --eye=4,0,1.7 --target=-4,1.5,2.5
```

The standard Split preset does not compile the guided, material, or reference
kernels at startup. Each is compiled only when selected. The driver still
compiles genuinely new pipelines on first use. On a pass-set cache miss, Lux
seeds Vulkan from a recent cache so changing one shader need not recompile
all the unchanged passes. Sponza's `--bench` reports time spent waiting for
pipeline creation separately from GPU frame timings.

Split's probe packer projects one cosine per antipodal bin pair, without
changing the directional samples or their coverage.

## Measuring against a reference

`examples/sponza_gi.rs --bench=DIR` prints GPU time per pass and saves frames as
PFM (`--cold` saves the first frames GI draws). `--reference` traces the
Split scene without a cache. `--pxtgi-reference` traces diffuse light on the
physics scene with the same mean material colors as PXTGI and without its
cache or filter. `--pxtgi` leaves reflections off for that comparison; add
`--reflections` to measure the usual PXTGI configuration.

`examples/flip.rs` shows where a run differs from the reference, as a viewer
would see it: HDR-FLIP ([Andersson et al.](https://research.nvidia.com/publication/2021-05_hdr-flip-visualizing-errors-rendered-high-dynamic-range-images),
through `flip-rs`) of each saved frame against the reference's, with a Magma
heatmap beside each as `.flip.png`:

```bash
cargo run --release -p verity-lux --example flip -- ref/still15.pfm run/still*.pfm
```

### Native GI A/B: Sponza courtyard

RTX 4070 Laptop, 640×360, 96 warm-up frames, 120 timed frames per camera phase.
Error is the absolute difference of 8×8 luminance blocks averaged over 16 still
captures, divided by the reference's mean indirect light. Split uses its
diffuse reference, MSGI its material-weighted reference, and the other diffuse
methods the PXTGI reference. References warm up for 300 frames. The timings are
individual runs on one device, not speedup guarantees.

| Mode | Still GPU ms (before → after) | Moving GPU ms (before → after) | Still error (before → after) |
| --- | ---: | ---: | ---: |
| Split | 2.863 → 2.242 | 3.467 → 2.191 | 0.206 → 0.208 |
| Ultra PBR / MSGI | 3.618 → 2.657 | 4.481 → 2.800 | 0.257 → 0.256 |
| PXTGI diffuse | 0.673 → 0.457 | 0.547 → 0.288 | 0.111 → 0.111 |
| PXTGI with reflections | 0.617 → 0.429 | 0.656 → 0.355 | — |
| Screen and voxel | 1.184 → 0.788 | 1.139 → 0.641 | 0.236 → 0.236 |
| Residual (four-ray benchmark) | 0.975 → 0.695 | 0.992 → 0.570 | 0.233 → 0.235 |
| Screen/voxel + HRC level 0 | 12.526 → 4.954 | 9.882 → 4.765 | 0.244 → 0.244 |

On a separate 320×180 fixture with a small emissive card reflected by a
metallic plate, MSGI's normalized luminance-block error against the material
reference fell from 0.080 to 0.061. Still GPU time was 0.806 → 0.816 ms.

The standalone 64³ HRC volume stayed at 10.1% slice error against brute force
with eight azimuth pages. HRC remains opt-in: its integration is faster here,
but the voxel-only method still costs less and scored lower diffuse error.

Sponza at 1280×720 on an RTX 4070 Laptop GPU, 300 frames after starting, against
that reference. Error is the mean absolute difference over 8×8 pixel blocks, as a
fraction of the reference's mean bounced light, and flicker is the mean change
between frames as a fraction of it. Running 2400 frames changed neither view's
error by much, so what remains is bias rather than slow convergence.

| View | Method | GPU time | Mean bounced light | Error | Flicker |
| --- | --- | --- | --- | --- | --- |
| Atrium | Split | 11.7 ms | 0.93× | 0.36 | 0.002 |
| Atrium | PXTGI | 3.8 ms | 1.13× | 0.28 | 0.006 |
| Atrium | Screen and voxel | 1.8 ms | 1.17× | 0.29 | 0.004 |
| Corridor | Split | 5.7 ms | 1.01× | 0.20 | 0.002 |
| Corridor | PXTGI | 3.2 ms | 0.98× | 0.07 | 0.013 |
| Corridor | Screen and voxel | 1.6 ms | 0.74× | 0.30 | 0.010 |

The atrium is `--eye=8,0,2 --target=-6,0,4` and the corridor the default view.
PXTGI took 4.1 ms in the atrium with reflections. These rows predate everything
below, and were measured against the reference before its fix, which read dark.

### The reference

The reference used to keep its running mean in a half-float image, which stops
taking small steps after a few hundred frames: it drifted 4% bright at 40 frames
and 13% dark at 1200 relative to 300, and the 600-frame images behind earlier
tables read 11% dark. It now sums in fp32, and 100 and 1200 frames agree to
0.009 error. Its model holds up too: every Sponza material's last mip, the mean
colour hits take, is within about 1% of its texture's linear mean. What it leaves
out is the renderer's own direct shading, which runs brighter than ray-traced
Lambertian light: shadow maps, its diffuse model and the spread normal maps give.

### The courtyard

The courtyard to the right of the corridor, `--eye=4,0,1.7 --target=-4,1.5,2.5`,
looks along the sunlit strip of floor with the right-hand arcade in shadow. Same
device and protocol, against a 1200-frame reference:

| Method | Rays | GPU time | Mean bounced light | Error | Flicker |
| --- | --- | --- | --- | --- | --- |
| Split, before | Hardware | 9.9 ms | 1.05× | 0.30 | 0.002 |
| Split | Hardware | 4.9 ms | 1.06× | 0.26 | 0.003 |
| Split | Software | 5.8 ms | 1.05× | 0.26 | 0.003 |
| PXTGI, before | Software | 4.4 ms | 1.14× | 0.22 | 0.006 |
| PXTGI | Hardware | 0.8 ms | 1.00× | 0.077 | 0.006 |
| PXTGI | Software | 4.3 ms | 0.99× | 0.077 | 0.006 |

What changed:

- PXTGI traces with hardware ray queries where it can (see below).
- PXTGI's diffuse rays take the scene's light, not the screen's. Each metre of
  screen march added error (0.074 with none, 0.094 at 0.1 m, 0.11 at 1 m,
  0.17 at the old 8 m), as did reading the screen at hits; neither saved time
  once rays were hardware. Reflections still take the screen first.
- Rays that leave the scene take the sky along the ray, as the cache and
  reference do, not the hemisphere at the surface normal.
- `Settings::budgeted` uses half-metre probes and 0.8 m intervals. Finer probes
  spread the update budget thin: 0.125 m gave 0.44 error in 19 ms.
- Software traversal: PXTGI's box test is one multiply-add a slab (Aila and Laine
  2009), 15% off its trace, and each mesh walk reads one buffer. Split's box test
  multiplies by a reciprocal instead of dividing, 8% off its closest-hit pass.
  Traversal stacks in groupshared memory and a world box per primitive both
  measured slower, so neither stayed.

With traced reflections PXTGI takes 1.0 ms in hardware. `--rays=4` brings its
flicker to 0.004 for another 0.3 ms. Split spends most of its time in `retain`,
`reduce`, `merge` and `commitCache`, which move cache entries and barely depend
on how rays are traced. These are one view of one scene on one device.

### Since then

Same view, now with the renderer's TAA in the image. Its 4-frame jitter makes
neighbouring frames differ, so flicker compares frames four apart, and error and
flicker read higher than the table above; compare these with each other only.

| Change | Method | GPU time | Error | Flicker |
| --- | --- | --- | --- | --- |
| Before | Split | 4.66 ms | 0.276 | 0.008 |
| Entries 272 → 240 bytes | Split | 4.42 ms | 0.275 | 0.009 |
| Unchanged entries stay in their history slot | Split | 3.87 ms | 0.278 | 0.009 |
| Before | PXTGI | 0.79 ms | 0.164 | 0.015 |
| Filters read normals from the depth image | PXTGI | 0.67 ms | 0.161 | 0.015 |
| Hits take their own direct light; filter by the mean's noise | PXTGI | about 0.70 ms | 0.146 | 0.016 |
| Before | Screen and voxel | 2.01 ms | 0.278 | 0.010 |
| All of the above | Screen and voxel | 1.88 ms | 0.275 | 0.010 |

The last PXTGI time adds the measured 0.03 ms of the hits' shadow rays to the
row above: by then other work shared the GPU, and passes no change touched
read up to 0.05 ms slower. Split's cache passes, before and after: `retain` 0.58 to 0.23 ms, `reduce` 0.50
to 0.30, `commitCache` 0.44 to 0.19, `merge` 0.43 to 0.44.

- Entries dropped the emissive fields: Verity's materials do not emit, so that
  source is never active.
- An entry with no new sample and nothing changed is a 48-byte stub in the
  working list: key, links, counts. Its radiance stays in its history slot,
  which `commitCache` keeps, rewriting only its links, and which readers reach
  through the stub (`loadEntry`, `mergedOf`). When no geometry or light changed,
  `retain` writes stubs from the header alone. A stub is only made for a slot
  commit gives the same bucket, so commit never has to move one.
- `temporal` writes the G-buffer's 24-bit normal beside linear depth (rg32f), so
  `atrous` and the composite's upsample stop reloading the full-size G-buffer at
  each tap: 0.18 to 0.07 ms and 0.18 to 0.14 ms. Screen and voxel GI shares
  these passes.
- A pixel's diffuse ray lights its hit with its own shadow ray and takes only
  the cache cell's bounce light. The cell's outgoing light is the direct light
  where the cell was claimed, which leaked sun across the shadow edges a cell
  straddles: 0.155 to 0.146 error for about 0.03 ms. The cache's own rays
  gained nothing from the same change, so they keep the cell's light.
- The à-trous filter stops at edges by the variance of the accumulated mean
  (one frame's variance over the frames averaged), not of a single frame,
  which over-blurred settled texels: 0.163 to 0.155 error, flicker 0.015 to 0.016.
- `--uncapped` (25.6 ms) gave 0.288 error, no better than the budget: Split's
  bias is its probe grid and directions, not its update budget.
- Nor is it the probe grid alone, nor the hits' BSDF. Split, same view, scored
  on the mean of four frames (one TAA jitter period) with the sun's single-pixel
  glints and a 2-pixel margin masked: glints of up to 90 shift a pixel between
  runs, and on single frames they decided the score. Before: 4.1 ms, 0.28 error
  (0.21 over 80-pixel blocks), 0.009 flicker; two runs of one build differ by
  0.006.
  - A ray per shaded pixel across the first interval, adding the difference
    between the hit's light and what the probes hold in that direction,
    clamped to four times the pixel's mean: 0.277 error, 0.027 flicker, 6.2 ms.
    Taking a miss's light from level 1 instead gave 0.334 and 0.12 flicker.
  - Lambertian hits with base colour, to match the reference's diffuse model:
    0.51 error at 0.81× the light. The full BSDF is the closer of the two.
  - Quarter-metre probes 0.296, 3.2 m intervals 0.292, 2400 warm-up frames 0.274.
  Most of the error is broad, not contact detail, and its cause is still open.
- Nor is it light leaking between probes, nor the probes' data being attributed
  to cell centres. Courtyard, scored on bounced light (each image minus a
  `--no-gi-light` run) over the mean of four frames, four runs each, which read
  0.245 ± 0.002 error and 0.007 flicker for the code as it stands. This scoring
  reads higher than the tables above, so compare these rows with each other only.
  None of these stayed in the code.
  - Probe visibility by oracle: `worldResolve` dropped each probe corner whose
    centre the receiver could not see by a traced ray: 0.286, worse. So visibility
    data per probe (moment or DDGI-style depth) would not help either.
  - Merging from the parent cell that holds the entry instead of trilinear over 8:
    0.233, flicker 0.0085, `merge` 0.50 to 0.21 ms. In the corridor it gave 0.39
    against 0.39 and tripled flicker.
  - Probe-centred shells, after polygonal radiance cascades' rule that stored
    regions tile space: each interval ends where the ray leaves a sphere around the
    cell centre, the bin is the direction from the centre, and merge looks each
    corner up from its own centre (parallax correction): 0.358, at 1.02× the
    light. Centring levels 1 to 3 and keeping level 0 ray-relative: 0.335. The
    ray-relative bins already hold what receivers in a cell see, which is what
    the resolve wants.
- A change of sun now counts as a change of the background source, which holds
  sunlight: before, only point lights' group was told. Small steps, up to about
  2.6° and 5% of the light a frame, keep the cache's history blending as time of
  day moves; larger ones drop it.
- The shared BSDF's diffuse is now energy-preserving Oren–Nayar (EON, Portsmouth,
  Kutz and Hill 2025), and its multiple scattering scales the GGX lobe rather
  than adding a uniform term. Hammon's diffuse divided by N·H and gave 262
  pixels of 17 to 41 in the direct image beside neighbours near 0.05; those were
  the "glints" the comparison above masks. Same view, against a reference
  re-captured with the new direct light, three runs: 0.248 to 0.253 error (0.18
  over 80-pixel blocks), 0.006 to 0.007 flicker, 0.91× the light. The passes that
  shade with it took the same time to within 0.01 ms.
- What PXTGI's error leaves is fine detail: 0.146 over 8-pixel blocks, 0.030
  over 80. Shaded areas still read about a quarter bright, and neither halving
  the cache cells nor turning the filter off changed that.
- Tried and dropped: capping each new sample at its history's mean + 3σ cut
  PXTGI's flicker 12% but lost 5% of its light (error 0.170); at 5σ it changed
  nothing. Fewer rays for settled texels: still-camera flicker *is* the settled
  texels' noise, and one ray there took it from 0.015 to 0.022.

### Sky light and a full jitter cycle

The bench's numbers above were taken over half of the renderer's TAA jitter. It
has 8 phases, not 4, and a readback takes two frames, so 8 saved frames saw only
the even or the odd ones. Two runs on opposite halves differ in sunlit texture
by more than the bounced light: scored that way, a run on the other half from
its reference read 0.44 error where it now reads 0.25. `--bench` now saves 16 frames 3 apart, which is every phase twice.
Error is the mean of the last 8 against the reference's. Flicker compares frames
one cycle (24 frames) apart, so it reads about twice what the tables above
show; compare it within this table only. Error over 80-pixel blocks is beside
error over 8.

Split, `Settings::budgeted`, before and after sky light, one filter pass and
the cheaper `seed`. Courtyard rows are the range of two or three runs.

| View | | GPU time | Mean bounced light | Error | Error, 80 px | Flicker |
| --- | --- | --- | --- | --- | --- | --- |
| Sponza courtyard | before | 4.38–4.42 ms | 0.97× | 0.250–0.251 | 0.172–0.175 | 0.015–0.016 |
| | after | 4.01–4.03 ms | 1.05× | 0.236–0.240 | 0.156–0.160 | 0.014 |
| Sponza corridor | before | 3.44–3.46 ms | 1.10× | 0.376–0.377 | 0.178–0.182 | 0.024–0.027 |
| | after | 3.13 ms | 1.13× | 0.383–0.386 | 0.193–0.195 | 0.021–0.033 |
| Bistro street | before | 6.36 ms | 0.73× | 0.354 | 0.309 | 0.008 |
| | after | 6.10 ms | 0.93× | 0.212 | 0.159 | 0.011 |

- Sky light. Split took its sky from a setting that defaulted to black, and an
  escaped ray counted as black whatever it was set to: the legacy renderer
  kept its own ambient light, and Verity's does not. In the open street that
  was a fifth of the light. The corridor, already 10% bright from light
  leaking between probes, gains another 2%.
- One à-trous pass instead of three, 0.22 ms. The probes are smooth before the
  filter sees them. With 3, 1 and 0 passes the courtyard scored 0.239, 0.236
  and 0.236 error and 0.015, 0.015 and 0.015 flicker; the corridor 0.385 with 3
  against 0.383–0.386 with 1; the moving camera's frames differed by 0.3% of
  the image, as two passes differ from three. Turning away for 90 frames and
  back, the first frame was already at 0.24 with any count. Only GI's first
  second (`--cold`) tells them apart: after 45 frames, 0.361, 0.371 and 0.373.
- `seed` writes 12 bytes of an empty descriptor instead of 52. Readers test
  `valid` and `padding` first and those share a cache line; the key and origin
  are on another, and half of all descriptors are empty. 0.46 to 0.36 ms.
- Transport loses no energy. With Lambertian base-colour hits and one bounce
  on both sides, Split gives 1.006× the reference's light (0.26 error, 0.15
  over 80 pixels). With its own diffuse lobe alone it gives 0.72×: the rest of
  what its hits reflect is specular, which the reference's hits fold into base
  colour. So the 0.81× that Lambertian hits gave earlier was not lost light.
- What is left is where the light lands: 0.84× in shadow and 1.07× on sunlit
  surfaces before, 0.94× and 1.17× after. Not self-intersection: a 5 cm ray
  offset instead of 5 mm made every class brighter, 1.18× in all, since
  Sponza's modelled relief occludes that much within centimetres.
- Tried and dropped: reading only header fields in `rank`, `parents` and
  `prefix` instead of the 240-byte entry. No change; the compiler already
  splits the load.

### Descriptors of their own

Each screen sample seeds eight descriptors, one per level and LOD variant:
the cells its surface falls in, which `rank`, `parents`, `prefix` and
`generateDirections` dedupe and give rays. They lived in the first slots of the
240-byte entry pool, where the trace later writes its deposits, so every touch
of one cost a cache line or two. They now have a 32-byte `Descriptor` buffer of
their own (`gi descriptors`, 32 bytes × 8 per sample, 15 MB at the budgeted
grid on 1280×720). A deposit's key names the same cell as its descriptor's, so
every probe-directory lookup reads the descriptors.

Same protocol as above, against the commit before:

| View | GPU time | Error |
| --- | --- | --- |
| Sponza courtyard | 4.14 → 3.49–3.63 ms | 0.235–0.236 → 0.234–0.238 |
| Sponza corridor | 3.13 → 2.54 ms | 0.383–0.386 → 0.386 |
| Bistro street | 6.10 → 5.35 ms | 0.212 → 0.214 |

`rank` went from 0.35 to 0.06 ms, `seed` from 0.38 to 0.18, and
`generateDirections` and `seedProbes` halved.

`remapProbeHeads` maps a probe whose representative slot commit did not keep
(most of them, on most frames) to history slot 0, so the next frame's
`bounceFeedback` misses it. Searching for it instead lets multi-bounce through
the probes reach everywhere it should, but measured worse in the courtyard:
0.245–0.247 error at 1.07× the light. More bounce lands where the image is
already too bright. That bias is below.

### Merging far light

Where the light lands, by direct light at the receiver (8-pixel blocks with
none, some, and more than 0.02): before this change the courtyard read 0.94×
the reference in shadow and 1.15× on sunlit ground. Two experiments, since
removed, split it:

- A per-pixel gather that traces each shaded pixel's own rays and shades hits
  exactly as the cache does, no probes: 1.08× in shadow, 1.13× sunlit. With
  Lambertian base-colour hits, one bounce, and no sky at hits, to match a
  one-bounce reference: 1.03× and 1.09×. Neither normal-mapped hit normals nor
  a ray offset matching the reference's 2 cm accounts for that 3–9% (2 cm made
  it 1.14× overall, 1 mm 1.05×), so it lies in how the two scenes and their
  materials differ, which these runs cannot separate further.
- The probes, against that gather on the same settings: 0.86× in shadow
  against 1.03×. Within the first interval alone (0.8 m) they held 1.22× the
  gather's light, so the loss was in the far cascades.

`finishMergeGroup` adds a direction's far light, the next cascade through its
4 finer child directions, to its near hits. Far cascades have up to 2048
directions a cell, so at 2048 updates a frame many children hold no sample yet.
The merge divided near and far together by their combined support, which gave
the near hits the weight of the missing children: an arcade floor, whose near
hits are dark ceiling and whose light comes from the courtyard, read dark, and
a sunlit floor, whose near hits are lit, read bright. The merged support then
weighted directions in `packProbes` and the parent's merge the same way. Now
the far light is averaged over the children that have samples, and a direction
with near and any far estimate counts whole.

| View | GPU time | Light | Error | Error, 80 px | Shadow / sunlit |
| --- | --- | --- | --- | --- | --- |
| Sponza courtyard | 3.49–3.63 → 3.50 ms | 1.04× → 1.08× | 0.234–0.238 → 0.208–0.209 | 0.154–0.159 → 0.118–0.121 | 0.94 / 1.15 → 1.03 / 1.10 |
| Sponza corridor | 2.54 → 2.65 ms | 1.13× → 1.18× | 0.386 → 0.396 | 0.196 → 0.213 | 1.13 → 1.18 in shadow |
| Bistro street | 5.35 → 5.48 ms | 0.93× → 1.03× | 0.214 → 0.168 | 0.161 → 0.106 | 0.87 / 0.96 → 1.01 / 1.01 |

The corridor, nearly all in shadow, already read bright from light leaking
between probes; its far light now has its full weight and it reads 4% brighter.
The time differences are within run-to-run noise; `merge` itself is unchanged.
With the merge fixed, the `remapProbeHeads` search above still made every view
worse (courtyard 0.219–0.222, corridor 0.399, Bistro 0.171), adding 3–4% light
to images already at or above the reference's, so the old mapping stays.

### Foveated probes

Probes already coarsen with distance: `splitLodBlend` doubles a point's probe
spacing and intervals each time its distance doubles, blending across the
boundary. `Settings::foveation` (degrees, 0 = off) also coarsens them away
from the gaze, by `1 + angle / foveation`, the linear falloff of visual acuity
(Guenter et al. 2012). Intervals scale with the spacing, so a coarser point
loses no near interval and needs no separate contact term. The gaze is the
view's centre unless `SplitRadianceCascades::set_gaze` gives an eye
tracker's direction. `sponza_gi --foveation=DEG` sets it.

Sponza courtyard, 1280×720, `budgeted`, against a 1200-frame `--reference`:
mean HDR-FLIP (`examples/flip.rs`) of the last four still frames against the
reference's last.

| Run | GPU time | FLIP |
| --- | --- | --- |
| Off | 3.48 ms | 0.296 |
| 10° | 3.51 ms | 0.299 |
| 5° | 3.69 ms | 0.295 |
| 1° | 3.28 ms | 0.299 |
| Off, 1024 updates | 3.39 ms | 0.295 |
| 5°, 1024 updates | 3.39 ms | 0.294 |

On a flat screen it buys nothing yet. Halving the update budget costs this
view nothing either, so the budget is not what limits it, and foveation has
nothing to move toward the centre. It should pay where receivers outrun the
budget: both eyes of a headset at full resolution. That is untested, and so
is how fast the fovea's fine probes warm up after a saccade. It stays off by
default.

### 192-byte entries

The cache passes move entries, so their size sets their cost. A quarter of
each entry was bounds: two light-hit spheres and two dependency boxes, 96
bytes of fp32. They are now halves relative to the entry's `origin`, rounded
outward (box minima down, maxima up, a sphere's radius grown by its centre's
rounding error), so a bound only ever grows and one past a half's range reads
as unknown. Both make an entry more likely to be invalidated, never less.
They live behind properties on `Entry` (`common.slang`), so readers and
writers kept their field names; because they are relative, `origin` must be
set before them. Radiance stays fp32: history blending takes small steps that
a half loses, as the reference's running mean did. The header (key to
`padding`) now comes first, so a retained stub is 48 contiguous bytes.

Sponza courtyard, 1280×720, `budgeted`, two runs of each build alternating;
error is the mean HDR-FLIP of the last four still frames against the
foveation section's reference.

| Pass | 240 bytes | 192 bytes |
| --- | --- | --- |
| `retain` | 0.254–0.259 ms | 0.186–0.188 ms |
| `reduce` | 0.331–0.332 ms | 0.240–0.241 ms |
| `merge` | 0.522–0.527 ms | 0.441–0.478 ms |
| `commitCache` | 0.225–0.226 ms | 0.249 ms |
| Still frame | 3.65–3.69 ms | 3.38–3.57 ms |
| Error | 0.296–0.298 | 0.293–0.294 |

The four cache passes went from 1.34 to 1.12–1.15 ms. `commitCache` lost
0.02 ms, likely to the conversions. The corridor and Bistro views were not
re-measured.

## Screen and voxel GI

A ray follows the depth buffer and takes its light from last frame's image. Past
what the screen shows, it walks a 256³ grid of voxels fitted to the loaded scenes.
Each frame a quarter of the voxels update one side's direct light and send a bounce
ray, so light bounces further over time. History and filtering are PXTGI's, shared
through `shaders/screen_gi.slang`.

`--voxel-bricks` tests conservative empty-brick skipping in both `--voxel`
and `--residual`. A 4³ brick is only skipped when its finalized occupancy is
zero; crossing a brick still consumes the corresponding fine-DDA steps.
Compare total GI time with and without the flag, including occupancy updates.

`--voxel-source-history` enables per-tap history rejection for both modes.
It attenuates large changes in screen, voxel and uncovered-ray coverage and
rejects history when Residual's voxel baseline appears or disappears. It does
not change the screen/voxel estimator or add rays.

Without a valid previous color image, the voxel methods go directly to their
voxel estimates. `--no-screen` also bypasses screen tracing on a warm view.
Residual uses the settled voxel estimate alone whenever a screen correction
cannot be sampled.

`--holographic=N` replaces voxel bounce walks inside clipmap level `N` with
the sliced HRC gather once its azimuth sweep completes. It integrates each
cone's solid angle and first directional moment analytically. Scrolling or
rebuilding that clipmap invalidates the sweep; injection uses regular voxel
rays until the replacement has finished. This is not the default voxel GI
path. `--example holographic` measures the CPU volume version against a
brute-force slice.

Outside the Holographic Radiance Cascades volume, voxel injection traces one
cosine-weighted bounce ray per side every eight frames. A per-cell scrambled
digital net changes only the ray directions, with no extra rays, passes, or
storage. In a synthetic directional-moment test, RMSE fell from 0.26940 to
0.12063 at 8 visits and from 0.09173 to 0.01673 at 64 versus independent hash jitter.

A cell a surface barely clips stays solid only in proportion to the surface in it.
Marking them all plugs narrow openings: in Sponza's corridor that left 0.60 of the
reference's light rather than 0.74, and keeping too few leaks (1.83).

This is a first version, and these are its limits:

- Static scene geometry is voxelized on the GPU as the clipmap levels scroll.
- Terrain and other retained meshes are not voxelized, and neither are moving
  bodies, so they appear only through the screen rays.
- Coarse clipmap levels cover more distance but lose thin surfaces and openings.
- Light reaching deep into a scene through small openings comes up short, as the
  corridor above shows.
- A voxel's colour is its material's mean colour, and materials have no emission.
- History does not react when a light changes, so light that has bounced more
  than once catches up over a few seconds.

## Residual GI

Residual GI (`ScreenVoxelSettings::residual`, `LightingMethod::Residual`) runs
the voxel method's passes with a different estimate per texel: a control
variate. The light its voxel has settled on arriving at the surface is the
answer, smooth and without a ray, and each screen ray adds only how far its
light differs from what the voxels show along it. A ray the screen can't
follow adds nothing and walks no voxels. Voxels also keep how much of their
arriving light is sampled point lights' direct light, which the renderer has
already given the surface, and the estimate leaves that out.

Sponza, RTX 4070 Laptop, 1280×720, 300 frames, against `--pxtgi-reference`
(1200 frames). Error is as above, over the mean of the last four frames; TAA
jitter swamped flicker, so it is left out.

| View | Method | Trace pass | Mean bounced light | Error |
| --- | --- | --- | --- | --- |
| Corridor | Screen and voxel | 0.51 ms | 0.76× | 0.42 |
| Corridor | Residual, 2 rays (default) | 0.22 ms | 0.76× | 0.48 |
| Courtyard | Screen and voxel | 0.70 ms | 1.09× | 0.45 |
| Courtyard | Residual, 2 rays (default) | 0.33 ms | 1.07× | 0.47 |

Each method's whole frame takes 0.7–1.1 ms. These rows take the sun from the
shadow map (below); before that the four read 0.47, 0.51, 0.44 and 0.45.

The baseline blends the eight voxels around the surface as a texture is
filtered, each by how nearly it faces the same way. Each voxel side also keeps
the mean way its bounce light came from, luminance-weighted, and the baseline
tilts toward the pixel's own normal (normal maps included) by a first-order
fit of that light. Read from the one voxel holding the surface it scored 0.57
and 0.48; blended, 0.53 and 0.46; tilted, 0.52 and 0.45, the brightest bounce
gaining most. The direction costs 32 MB more for the million-voxel pool.

Two rays a texel were as good as four (0.51 against 0.52 in the corridor, 0.45
both in the courtyard), so `ScreenVoxelSettings::residual` uses two, at under
half the trace pass of screen and voxel GI. The gap to it that is left is detail
between pixels: block by block the baseline correlates 0.76 with the
reference where per-pixel voxel walks reach 0.84. A voxel's settled light is
one estimate a cell, gathered 1.5 cells off its surface over 64 frames.
`sponza_gi --residual`
runs it (`--rays=N` sets its rays).

### Sunlight from the shadow map

Both voxel methods left the corridor's brightest bounce about 40% short: one
patch in shade, lit only by bounce. Holding the first bounce alone on both
sides (the reference at one bounce, voxels leaving only their direct light),
the voxels gave 0.69× of it, and later bounces added as much to each (×1.20
against ×1.25), so the loss was in the first bounce. With sun shadows off on
both sides too they gave 0.95×: the voxels' shadow rays blocked sunlight the
scene does not.

Testing each pixel facing the sun as a voxel would showed where: of what the
renderer's shadow map lit, the walk blocked 53% in the corridor (few such
pixels) and 8.8% in the courtyard, most of it at roof edges 16 m along the
sun, whose cells there are 0.25–0.5 m. A level spans 128 cells, so cells at a
distance are the same size whatever `--cell` is, which is why finer voxels,
more levels, longer rays, more frames, other ray starts and passing the first
cell of each coarser level all left it alone.

So a voxel now takes its sun from the renderer's shadow map wherever one of
its cascades holds the voxel's face, sampled at the face rather than the
centre, and walks the voxels only beyond them. Against the shadow map the same
test blocks 2.0% and 0.5%. The corridor's brightest fifth went from 43% short
to 32%, and its error from 0.47 to 0.42; the courtyard, already a little
bright, went from 0.44 to 0.45.

## PXTGI

PXTGI traces the physics scene rather than a GI scene of its own, and runs none
of the cascade passes. `Lux::sync` keeps it following the physics world.

Rays run against a `verity_physics::RayScene`: rigid colliders, the surfaces of
soft bodies, and loaded glTF meshes as static trace geometry. With hardware ray
queries the same scene becomes acceleration structures: a BLAS per static mesh,
built when it loads, and a box per body or soft body that the shader tests as
before. The TLAS is rebuilt inside the frame's commands whenever the scene
changes, so moving bodies cost no stall. The same scene
answers CPU rays with `RayScene::ray`. Diffuse rays trace the scene, where a
hashed radiance cache carries further bounces; `max_distance` lets them follow
the screen first for geometry drawn but missing from it. Reflection rays follow
the screen first, taking their light from the previous frame's lit image.

World-cache reads pair radiance with the mass of the last completed shade
pass, so gathering a new cell cannot mark it ready ahead of shaded data.
Glossy history stays at the exact texel on a still view; during motion only
depth- and normal-compatible taps survive reprojection.

`--pxtgi --pxtgi-temporal-reuse` reprojects one previous raw first diffuse hit.
It verifies that the current closest hit still matches, re-shades it, and
averages its density-adjusted contribution with the first fresh hit. It does
not reuse RGB or chain history. Misses, moving physics and bad reconnections
use the fresh result. This is an opt-in, biased world-only comparison
(`--screen=0`), not a reference tracer or a full ReSTIR PT implementation.

Cache cells double in size with each doubling of distance past 8 m, and each
cell claims its parent one size up, which stays alive while the cell is used.
A cell with fewer than four rays blends toward its parent, so a surface just
come into view, or one the camera walked up to, starts with soft bounce light
rather than none (the fallback of Hierarchical Voxel Irradiance Caching).

Physics supplies more than geometry:

- A resting scene costs one pose comparison per body. Nothing is refit or
  uploaded, so sleeping bodies are free on the GPU side.
- Moving bodies report the region they disturbed. For 16 frames the GI keeps
  less history and refreshes its cache there; the rest of the view stays
  converged.
- Trees refit as bodies move and rebuild every 15 refits, or when bodies come
  and go.

When a light's power changes, everything remembered scales with it for one
frame. Light is linear in its sources, and without this, stale light feeds back
through last frame's image and fades only over seconds. The scaling is exact when
one source or all of them change; with several, it estimates each source's share
from its power, and the short history after a change corrects the rest.

Limits to know about:

- Static glTF meshes answer rays only. They make no contacts, so bodies fall
  through them; give level geometry colliders as well.
- A hit's albedo is its material's mean colour (the texture's last mip), since
  rays carry no texture coordinates. Alpha-masked surfaces count as solid, and
  alpha-blended ones are not traced.
- Motion vectors are camera-only, so history on a moving body is kept or dropped
  by depth rather than reprojected.
- Soft bodies trace as neutral grey, and materials have no emission.

Entities without a body trace as the cubes they are drawn as, so walls need no
colliders. In the editor, select **Scene & Lighting** and pick the method under
**Scene Lighting**; PXTGI follows the scene while editing and the simulation in
play mode.

`examples/pxtgi_room.rs` is a physics room with flags for capture and for each
measurement below, taken on an RTX 4070 Laptop GPU:

| Measurement | Result |
| --- | --- |
| GPU time, the room at 1600×900 | 1.3 ms |
| GPU time, the room plus Sponza's main scene | 1.3 ms (4.4 ms in software), measured with the old 8 m screen march |
| Bounce light against a CPU path tracer on the same scene (`--reference`) | 9.9% mean error in hardware, 9.7% in software (9.4% with the old screen march) |
| Light left one frame after it goes out (`--extinction`) | 0.0% (the Bevy build: 68.4% after 100 ms) |
| Light dimmed to half (`--extinction=half`) | 50.8% after three frames |
| Residual 10 frames after a wall panel moves | 13%, against 29% without moved regions |
| Bounce light after the camera steps 3.5 m in (`--approach`) | 5.5% below where it settles at worst, against 10.4% without parent cells; 0.79 to 0.80 ms |

History runs to 48 frames while the camera is still, for steadier light. That is
what took the room's error from 8.8% to 9.6%: at 32 frames Sponza flickers half as
much again.

The reference is Lambertian with the renderer's diffuse share, so it leaves out
specular interreflection. These are one scene and one device, not a certification.

## Research and validation

The implementation is checked against
[Split Radiance Cascades v2, revised October 11, 2026](https://arxiv.org/abs/2607.20384v2).
Background references include the local
[Holographic Radiance Cascades paper](../GI%20Research/2505.02041v1.pdf) and
[Neural Radiance Caching paper](../GI%20Research/Neural%20Radiance%20Caching.pdf).
Holographic Radiance Cascades is a 2D reference here; its dense 3D form is not
added because its memory scales as N⁴. Online neural radiance caching is also
deferred because it needs a separate training and inference path. Related code
and shader references include [RadianceCascadesGIVulkan](https://github.com/tlegoc/RadianceCascadesGIVulkan),
[the 3D merge shader](https://www.shadertoy.com/view/X3XfRM),
[SRC-DGI](https://github.com/mxcop/src-dgi), and
[vlam](https://github.com/entropylost/vlam). The supplied [Chandra source](ref/chandra/)
and [Split Radiance Cascades paper](ref/Split%20Radiance%20Cascades.pdf) are the
local algorithm references.

[Holographic Radiance Cascades](https://arxiv.org/abs/2505.02041) describes a
2D interval-extension method. Its radiance/transmittance algebra overlaps the
existing Split cache, which already stores sparse 3D intervals. A distinct 3D
An HRC implementation would need its own sparse ray-extension algorithm; directly building
the paper's dense representation in 3D has N⁴ memory scaling, so this crate does
not implement it.

[Validation notes](docs/validation.md) explain how to run the current examples.
They do not carry forward measurements from earlier implementations. The
`scripts/check_quality.py` utility compares supplied image sequences and reports
regional spatial and temporal metrics; it does not generate images or establish
radiometric accuracy.
