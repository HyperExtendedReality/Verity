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

To add a method, implement `GiMethod`, add a `Method` variant and a row in
`METHODS`, and give it a `LightingMethod` so scenes can name it. The editor lists
whatever `METHODS` holds. Follow terrain through
`RenderExtension::retained_changed`.

Both methods trace retained meshes such as `verity-terra` chunks, which may be
set before or after the extension is installed. See the Terra section of the
root README.

The Verity Sponza example runs without GI by default. Its source documents a
known first-frame GPU hang when the experimental GI extension is enabled with
`--gi`; the notes after the PXTGI section describe the older Bevy implementation
retained in `legacy/`, not the current Verity API or its supported features.

## Split Radiance Cascades

Each frame traces a budget of receivers into a world cache, merges cascades from
far to near, and packs directional probes that every pixel reads. Three things
keep the cost of a frame down:

- `reduce` lists the entries that survive, so `merge` touches only those instead
  of scanning the whole pool once per level.
- Entries with no new sample merge every fourth frame. Any change to lights or
  geometry merges them all.
- `seed` writes only the fields descriptors use, not whole 240-byte entries.

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
picked in proportion to area times emission. Each shaded pixel samples one a
frame with a shadow ray (`worldResolve`), and so does each ray hit; probe rays
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

## Measuring against a reference

`examples/sponza_gi.rs --bench=DIR` prints GPU time per pass and saves frames as
PFM (`--cold` saves the first frames GI draws), and `--reference` saves a
path-traced image of the same view:
`PxtgiDebug::Reference` traces diffuse light with no cache or filter, and hits
take mean colours as PXTGI's do. `--pxtgi` leaves reflections off to compare
diffuse with diffuse; add `--reflections` for what PXTGI normally costs.

`examples/flip.rs` shows where a run differs from the reference, as a viewer
would see it: HDR-FLIP ([Andersson et al.](https://research.nvidia.com/publication/2021-05_hdr-flip-visualizing-errors-rendered-high-dynamic-range-images),
through `flip-rs`) of each saved frame against the reference's, with a Magma
heatmap beside each as `.flip.png`:

```bash
cargo run --release -p verity-lux --example flip -- ref/still15.pfm run/still*.pfm
```

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

## Screen and voxel GI

A ray follows the depth buffer and takes its light from last frame's image. Past
what the screen shows, it walks a 256³ grid of voxels fitted to the loaded scenes.
Each frame a quarter of the voxels update one side's direct light and send a bounce
ray, so light bounces further over time. History and filtering are PXTGI's, shared
through `shaders/screen_gi.slang`.

A cell a surface barely clips stays solid only in proportion to the surface in it.
Marking them all plugs narrow openings: in Sponza's corridor that left 0.60 of the
reference's light rather than 0.74, and keeping too few leaks (1.83).

This is a first version, and these are its limits:

- Scenes voxelize on the CPU as they load, and the whole grid rebuilds with each
  scene loaded.
- Terrain and other retained meshes are not voxelized, and neither are moving
  bodies, so they appear only through the screen rays.
- One grid covers everything, so cells grow with the scene, and it is too coarse
  for a landscape. There are no clipmaps yet.
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
| Corridor | Screen and voxel | 0.46 ms | 0.64× | 0.47 |
| Corridor | Residual | 0.42 ms | 0.65× | 0.53 |
| Corridor | Residual, 2 rays | 0.21 ms | 0.64× | 0.54 |
| Courtyard | Screen and voxel | 0.68 ms | 1.05× | 0.44 |
| Courtyard | Residual | 0.49 ms | 1.01× | 0.46 |
| Courtyard | Residual, 2 rays | 0.31 ms | 1.05× | 0.47 |

The baseline blends the eight voxels around the surface as a texture is
filtered, each by how nearly it faces the same way. Read from the one voxel
holding the surface it scored 0.57 and 0.48.

It is cheaper, and halving its rays costs it little, but still less accurate,
and the screen rays are not why: with them off (`--no-screen`) the corridor
measured 0.53, the baseline alone. A voxel's settled light is gathered from
1.5 cells off its surface, and lags 64 frames, where the voxel method walks
voxels from each pixel every frame. `sponza_gi --residual` runs it.

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

# bevy_lux

`bevy_lux` implements diffuse global illumination in Bevy with Split
Radiance Cascades, following the strict base of the v2 paper. It uses persistent
world-space intervals with validated temporal reconstruction.

The crate and its local helper crates target Bevy 0.20.0.

## Add GI to a scene

Install `RadianceCascadesPlugin`, attach `RadianceCascades` to a camera, and mark
participating meshes with `RadianceCascadesGeometry`:

```rust
use bevy::pbr::DefaultOpaqueRendererMethod;
use bevy::prelude::*;
use bevy_lux::{RadianceCascades, RadianceCascadesGeometry, RadianceCascadesPlugin};

App::new()
    .insert_resource(DefaultOpaqueRendererMethod::deferred())
    .add_plugins(DefaultPlugins)
    .add_plugins(RadianceCascadesPlugin::default())
    .add_systems(Startup, setup)
    .run();

fn setup(mut commands: Commands, mut meshes: ResMut<Assets<Mesh>>, mut materials: ResMut<Assets<StandardMaterial>>) {
    commands.spawn((
        Camera3d::default(),
        RadianceCascades::default(),
        Transform::from_xyz(0.0, 2.0, 6.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
    commands.spawn((
        Mesh3d(meshes.add(Plane3d::default().mesh().size(10.0, 10.0))),
        MeshMaterial3d(materials.add(Color::WHITE)),
        RadianceCascadesGeometry,
    ));
}
```

GI uses deferred rendering so the renderer can read receiver material and
surface data. Receiver inputs are depth, packed G-buffer base color, metallic,
roughness and shading normal, plus a separate raster attachment for receiver
identity and geometric normal. Opaque and alpha-mask surfaces receive GI.
Alpha-blended and premultiplied meshes can participate as shadow casters and
emitters using stochastic texture-alpha coverage; they do not receive the
deferred GI composite. This is not refractive or volumetric transmission.
Forward-only receivers and MSAA remain unsupported.
The examples show setup, controls, raw output, and scene registration patterns.

## Ray-traced GI

Set `method` to `GiMethod::RayTraced` on the camera's `RadianceCascades`
component. It is a separate estimator from Split Radiance Cascades and runs
none of the cascade passes; it always traces with Vulkan hardware ray queries,
whatever `hardware_ray_queries` says. It never falls back to the cascades:
until its acceleration structures and pipelines are ready, or on a device
without ray-query support, the view keeps Bevy's native lighting.
`RadianceCascadesStatus::active_method()` reports the method of the latest
dispatch.

The diffuse gather uses screen history, a hashed world-space radiance cache and
cascaded irradiance volumes. Cache cells accumulate direct and bounce lighting
separately; `ray_traced.multi_bounce` controls the latter. Temporal and spatial
reservoir reuse, spherical-harmonic reconstruction and bilateral filtering
trade exact path-traced results for real-time performance. Visible-surface
specular lighting and transparent receivers remain outside this diffuse method.
Bounded rigid-geometry changes invalidate cache cells whose retained lighting
paths intersect the changed region; changes without reliable bounds reset the
cache. Irradiance volumes are cleared on geometry changes, so continuously
moving geometry can delay their convergence.

## Physics-collider GI (PXTGI)

With the default `pxtgi` feature, set `RadianceCascades::method` to
`GiMethod::Pxtgi` and add `bevy_box3d` colliders to the scene. PXTGI traces
those colliders rather than `RadianceCascadesGeometry` meshes. Collider material
colours are approximated by texture means; off-screen geometry without a
collider cannot be hit. `PxtgiSettings` controls ray count, multi-bounce cache,
reflections and debug views. `reset_history()` and image-replacing debug views
invalidate screen history; changing the reflection mode invalidates reflection
history.
The optional multi-bounce cache is shared between views and currently has
unsynchronized cell accesses; its results can depend on GPU scheduling and
can mix view-dependent lighting. Disable `multi_bounce` where those risks are
unacceptable.

The Sponza example accepts `--pxtgi` and reports `render/pxtgi` GPU pass
timings in its F1 menu. For a matched fixed-camera regression capture:

```text
cargo run --release --example sponza -- --pxtgi --capture-stability --capture-raw --camera-pose-frame=0 --time=12 --width=640 --height=360 --warmup-frames=120 --capture-prefix=tmp/pxtgi
```

The existing Sponza benchmark and validation scripts measure Split, not PXTGI.
PXTGI's raw support marks eligible receivers, not convergence or collider-hit
coverage; an independent matched reference is needed to assess lighting accuracy.

On the Vulkan RTX 4070 Laptop GPU at 320×180, the Sponza PXTGI motion
capture reached the history reset and moving-light stages but failed its
100 ms light-extinction gate: the latest observed residual was 68.4%, against
a 5% limit. This is not a target-device performance or quality certification.
During the stationary part of that capture, repeated identical transform writes
kept geometry revision 137 unchanged from frame 128 to 159; before the motion
tracking fix the revision advanced from 141 to 172 over the same interval.

## Tune a view

`RadianceCascades` contains the per-camera quality and appearance settings.

For the 1280×720, 4 GB GPU profile, use `RadianceCascadesPlugin::low_end()`
and `RadianceCascades::low_end()`. Cornell and Sponza use this profile by default;
the library's `Default` settings retain their work and memory budgets. The profile
uses 1 m base probes, a 1.6 m cascade-zero cutoff, and 65,536 retained cache
records. Smaller probes need more completed transport work to converge smoothly.
It targets GTX 1650 / RX 5500 XT class hardware; that performance target has not
been certified on those GPUs. See [quality and performance validation](docs/quality-and-performance.md)
for the measurement protocol and current evidence.

Every view uses one amortized sampling scheme. `sampling` separates
candidate-grid density from selected receiver updates, cache capacity,
traversal work, and resolve scale; zero fields inherit `resolution_divisor`
and `max_rays`, and `max_rays` caps the receiver updates selected per frame
while the candidate grid stays fixed. Selected receivers are traced with one
compact dispatch whose lanes are spaced out for small budgets.
Use `..Default::default()` or `..RadianceCascades::low_end()` in struct literals
to account for the `sampling` field.

Use `..Default::default()` in struct literals to inherit shared settings.

| Setting | Purpose |
| --- | --- |
| `probe_spacing`, `interval_length` | Set the world-space cascade spacing and interval scale. |
| `resolution_divisor`, `max_rays` | Set the sparse receiver grid and trace budget; geometry-aware reconstruction fills the full viewport. |
| `normal_bias` | Offset surface samples to reduce self-intersection. |
| `intensity`, `environment` | Set GI strength and the diffuse environment contribution. |
| `multi_bounce` | Enable an additional off-screen cache for diffuse multi-bounce lighting; disabled by default. |
| `cached_hit_sky` | With `multi_bounce`, shade ray-hit sky light from the secondary cache instead of environment visibility rays; disabled by default. |
| `hardware_ray_queries` | Trace the primary closest-hit and visibility passes with hardware ray queries; falls back to the software BVH when unavailable. Disabled by default. |
| `history_frames`, `temporal_blend` | Control validated screen and world-space history. Zero disables primary history; the optional secondary cache still retains history for temporal feedback. |
| `split_transmittance_blend` | Set per-observation retention for interval transmittance while lighting radiance refreshes. |
| `spatial_filter_steps` | Control geometry-aware a-trous filtering for screen history; zero disables the screen-space filter. |
| `emissive_samples` | Tune emitter sampling. |
| Diffuse transport | Split evaluates diffuse illumination, including emissive first hits. Bevy retains native punctual/environment lighting, their shadows, and visible-surface specular shading. Screen history checks receiver identity and surface compatibility. |
| Split transport | Each world probe stores separate 6×6 octahedral Lambertian irradiance and support fields for background, emissive, and finite punctual sources. Resolve normalizes each field independently; Penumbra adds no glossy receiver transport. |
| `artistic_occlusion` | Enable the optional occlusion adjustment. |

Call `reset_history()` after discontinuous camera or lighting changes. Bevy
still owns scene transforms and asset lifetime. The low-end profile uses a
160×90 candidate grid, updates at most 1,024 receivers per frame, caps admitted
traversal queries at 16,384, retains at most 65,536 world-cache records, and
resolves at 640×360 for a 1280×720 view. Depth, receiver identity, normal, and
plane checks gate reconstruction back to the output viewport. Split combines
persistent world-space intervals with temporal reconstruction. Surfaces without
a compatible traced receiver can still have missing GI; reconstruction cannot
recover untraced lighting.

When no active camera has positive GI intensity, scene updates and receiver
identity extraction are suspended. Previously allocated resources can remain
resident. On re-enabling GI, referenced assets are revalidated and scene
preparation may take time before GI resumes.

When `multi_bounce` is enabled, prior primary ray hits seed a separate cache
whose probes are two LOD levels coarser. Its completed irradiance feeds later
primary and secondary hits. The configured receiver-update, traversal and cache
entry limits are shared between the caches, and the extra allocations count
toward the view budget. The secondary pass waits when the update budget is one,
or when a nonzero traversal-query or cache-entry limit is one. A zero query or
entry limit retains its existing unrestricted meaning. Secondary feedback uses
the nearest coarse probe and blends supported adjacent LODs across their
overlap. Secondary history is required for temporal multi-bounce feedback even
when primary screen history is disabled.

With `cached_hit_sky`, secondary rays that escape carry the environment, and
ray hits covered by the secondary cache take their sky light from it, as in the
paper, instead of tracing environment visibility rays. Uncovered hits keep the
visibility-tested sample. The cached sky is diffuse-only and one frame late; in
Sponza it lowered GI GPU time by about 1.3 ms (6.6%) with matched captures
within run-to-run variation (see validation notes).

With `hardware_ray_queries`, the primary closest-hit and visibility passes walk
a hardware TLAS/BLAS with wgpu ray queries instead of the software BVH. The
hardware only proposes candidate triangles. Every BLAS is non-opaque, and each
candidate is re-tested with the software triangle, culling, layer and
alpha-coverage tests. Accepted hits therefore keep the software distance and
barycentrics, and toggling the setting keeps all GI history. Hit shading,
query recording and replay, the overflow trace and the secondary cache stay on
the software traversal. This requires Vulkan with the SPIR-V passthrough path,
wgpu's experimental ray-query feature (Bevy requests adapter features by
default), and unpaged scene geometry. BLASes are built from the scene geometry
buffer when a view first requests hardware traversal; deformed meshes rebuild
theirs every frame. This acceleration-structure memory is outside the scene and
view budgets. While the structures or pipelines are not ready, the software
traversal runs. `RadianceCascadesStatus::hardware_ray_queries` reports which
traversal the latest dispatch used.

The library defaults to Split Radiance Cascades, as does the Sponza demo; its
`--rtgi` flag starts with ray-traced GI. In the interactive Sponza demo, `F1`
toggles the debug menu: live status, a method selector that shows only the
selected method's controls, sun shadows, time of day and GPU pass timings.
`M` switches the method, `F12` toggles cursor grab and `H` toggles the
cascades' hardware ray queries, which Sponza requests by default.
`--software-traversal` starts with the software traversal and `--shadow-maps`
with Bevy's shadow maps instead of ray-traced sun shadows.
Sponza defaults to 2,048 receiver updates and 131,072 traversal queries per
frame, requested grid divisor 4, resolve scale 4, and two direct-light samples.
The library's low-end preset remains at 1,024 updates and 16,384 queries. The demo keeps its requested
profile fixed while running; `--max-rays`, `--max-traversal-queries`, and
`--resolution-divisor` adjust it; `--multi-bounce` enables the secondary cache,
and `--cached-hit-sky` adds cached hit sky lighting.
Interactive mode requests VSync by default; `--no-vsync` requests AutoNoVsync
instead and may produce tearing. The platform can choose a fallback present mode.
Split's Sponza profile uses 1 m probe spacing
and a 1.6 m first interval; it permits 32,768 receiver cells and 262,144 cached
intervals within a 1 GiB view budget, subject to device allocation limits. Run
`scripts/benchmark_sponza.ps1` for the release 1920×1080 windowed benchmark
with VSync-off requested. It records a GI-disabled baseline and requires 120
warmup plus 240 measured CPU/GPU samples. The offscreen benchmark remains
available for drained measurements; use `--benchmark-offscreen-no-gi` for an
offscreen GI-disabled baseline. Queue-envelope timings can include GPU idle
time and are not measurements of active GPU execution. Schedule CPU diagnostics
measure elapsed time, including waits, not active CPU time. The JSON report
includes `samples.trace_gpu_ms`, a distribution of Split trace pass GPU
timestamps observed within the measurement window; it is empty when GI is
disabled. Startup logs record the first GI dispatch and the first raw readback
reaching 95% eligible-pixel support; neither event waits for warmup.

The default shadow profile uses two 2048-pixel cascades over 40 m; the sun's
shadow maps are disabled when its daylight illuminance is zero.
`--shadow-profile=reference` selects four 2048-pixel cascades over 150 m;
`--shadow-profile=balanced` selects the default. The default asset profile is
`source`. An explicit `--asset-profile=balanced` selects a locally cooked bundle
only; its fidelity is not certified by structural verification alone.

To cook that optional bundle, place the
[gltfpack 1.3 Windows release](https://github.com/zeux/meshoptimizer/releases/tag/v1.3)
ZIP and `gltfpack.exe` in `tmp/tools/gltfpack-1.3/`, then run
`python scripts/optimize_sponza.py` and `python scripts/optimize_sponza.py --verify`.
The cook verifies the ZIP hash, keeps the supplied assets untouched, and writes
`assets/sponza-optimized/`; redistribution permission for derivatives is unknown.
Run the converted scene with
`cargo run --release --example sponza -- --asset-profile=balanced`.
The cook retains CPU-readable PNG textures because the GI atlas does not yet
accept GPU block-compressed textures.
The benchmark and validation scripts accept `-AssetProfile` and `-ShadowProfile`.
For matched scene-linear comparisons, add `--capture-raw` to
`--capture-stability` and use `scripts/check_sponza_quality.py` on the two
capture prefixes. Schema 14 reports windowed GPU pass distributions in
`samples.gpu_passes`; unavailable timings have null statistics.

`scripts/validate_sponza_gi.ps1 -Release`
checks Split with static captures, a moving blocker, and a moving point
light. Dark camera directions use matched GI-on/off controls instead of
requiring courtyard brightness in a shadowed arcade. These scripts enforce the
target budgets; their current results are not certified until the acceptance run passes.
See the [validation status and historical measurements](docs/validation.md).
With the low-end profile, four irradiance images and five temporal metadata
images use 640×360 storage for a 1280×720 viewport, about 31.6 MiB combined.
The sparse GI resolve, receiver identity, scene cache, and Bevy-owned rendering
resources are accounted separately. Split reconstructs the paper's diffuse
indirect term from persistent world-space intervals. Unsupported samples remain
distinct from observed black radiance.

Bevy's native environment maps continue to control native PBR lighting.
`RadianceCascadesEnvironment` supplies a separate input to Split's indirect
transport; assigning it does not automatically enable or replace Bevy's native
`EnvironmentMap`.

Emissive geometry contributes radiance through first-hit cascade rays. No
primary next-event estimator or emitter classification is required. Remove
the former `RadianceCascadesEmitterSampling` component from existing scenes.
The renderer still uses next-event estimation to shade secondary hit surfaces;
these reflected paths do not duplicate first-hit emission. Small emitters can
require more samples to converge.

See [rendering boundaries](docs/rendering-boundaries.md) for the native Bevy
visibility-buffer path, sparse-upload constraints, and the diffuse transport
and light-ownership boundary.

On native targets, Slang's generated WGSL is cached between runs under the
platform user cache directory (`%LOCALAPPDATA%\bevy_penumbra\shaders` on
Windows). Set `BEVY_PENUMBRA_SHADER_CACHE_DIR` to use another directory. The
cache key includes the embedded shader sources, Slang version, target, and O2
optimization level. Every backend starts with validated WGSL; Naga maps resource
bindings for the selected backend. Device pipeline caches are saved atomically when supported.
Sponza compiles the Split pipeline set on demand at startup.

## Examples

The Cornell box is a compact GI scene with raw capture and GPU timing support.
Sponza loads the included courtyard, curtains, and ivy assets. The lighting
ownership example checks that GI view state is restored when GI is disabled.
The capture profiles retain 64 history frames: Cornell blends at 0.97 and
Sponza at 0.985, averaging sparse samples without adding per-frame dispatches.

```sh
cargo run --example cornell_box
cargo run --example cornell_box -- --capture
cargo run --example cornell_box -- --capture --capture-output=tmp/cornell_after.png
cargo run --example cornell_box -- --capture-raw --raw-output=screenshots/cornell_src.pfm
cargo run --example cornell_box -- --live-capture --width=1280 --height=720 --output=captures/cornell_1280x720.png
cargo run --example cornell_box -- --live-capture --capture-frame-count=8 --output=captures/cornell_flicker.png
cargo run --example cornell_box -- --benchmark --width=1280 --height=720
cargo run --example sponza
cargo run --example sponza -- --capture
cargo run --example sponza -- --capture-motion
cargo run --example sponza -- --capture-stability
cargo run --example sponza -- --capture-motion
cargo run --example lighting_ownership
```

The Cornell window uses Split at a requested logical size of 960×720. It accepts
`--width=N --height=N` for other sizes. Interactive windows use the monitor's
DPI scale factor, so at 125% DPI a 960×720 request becomes a 1200×900 physical
target. `--live-capture` uses that same DPI scale, controls, and diagnostics
overlay; it automates saving the interactive window after warmup. The receiver
grid follows the physical
window size and raises the divisor as needed to stay within
`--max-receiver-cells=N`; renderer device and per-view memory limits can lower
the active grid further. The renderer reserves the full-resolution receiver
texture before choosing GI allocations and can raise the GI resolve divisor
through 8 when a large target would exceed the memory budget. Reconstruction
still produces the full-resolution image. Allocation adjustments are logged
with the viewport, effective grid, resolve divisor, and memory budget.
The F3 diagnostics
show the physical target, the grid allowed by the configured cell cap, the
renderer-reported active candidate count, and successful updates alongside the
scheduled update budget. The warmup estimate uses that configured budget, not
the phase-varying successful-update count. The interactive Cornell demo defaults
to 2,048 receiver updates per frame and an 8,192-cell receiver cap. The active
grid adapts its divisor
to the physical window size and may be lowered further by renderer memory
limits. `--max-rays=N` and `--max-receiver-cells=N` can lower those budgets for
slower GPUs. `--resolution-divisor=N`, `--max-receiver-cells=N`, `--max-rays=N`,
and `--resolve-scale=N` also apply to interactive runs. The window title includes
the initial scene and requested size; use `--window-title=NAME` to supply a
recording-friendly title.

`--live-capture` waits for the live window's GI warmup, captures its rendered
primary window, and exits after saving. It defaults to
`captures/cornell_live.png`; use `--output=PATH` to choose another file. Pair it
with `--capture-frame-count=N` to save a numbered PNG sequence, sampling once
per fresh GI render result while earlier PNG writes finish asynchronously. A raw
capture writes a little-endian RGB PFM file and a JSON sidecar with its dimensions
and settings. Benchmarks report completed GI GPU samples; they do not represent
total application frame time.

GI scene updates filter image and material asset events to the assets referenced
by visible GI geometry. Debug text and unrelated UI atlas updates preserve GI
history. Idle controls leave the ceiling emitter's transform untouched, and
identical geometry transforms do not trigger a rebuild. The warmup counter shows
rendered GI frames, rather than independent samples per receiver.
`RUST_LOG=bevy_lux::render=debug,bevy_lux::scene=debug` logs scene rebuild
and sparse-history reset causes.
`--capture-warmup-frames=N` can require a longer live observation before saving.

The Sponza example expects the supplied files under `assets/`. See
[the asset inventory](docs/sponza-assets.md) for scene contents. Its demo limits
material texture images to 512 pixels and uses `--max-rays=N` to adjust the
per-view work budget. To admit those assets and replacement overlap, the
benchmark allows a 2 GiB scene and a 768 MiB material atlas; this is separate
from the library's 4 GB low-end memory profile.

Split traces selected receivers with compact dispatch. Budgets below 8,192
updates space each receiver over up to eight lanes so more warps hide traversal
latency; at 2,048 updates this cut the trace pass from 12.4 to 10.6 ms (p50).
Primary transport is then split into four compute passes: closest hits, a
recording run of hit shading that stores its shadow queries, those queries
traced one per lane, and a replay that shades with the stored answers and
deposits the intervals. Each pass holds at most one copy of the hit-shading
code, and the shadow-ray pass is a small kernel. Up to 8,192 receivers per
frame use this path; any beyond that trace in a single pass as before.
The compact path's midnight raw-fidelity comparison remains unresolved. On the measured RTX 4070 Laptop GPU, the 1080p
Vulkan screen did not reach 60 FPS; see [validation notes](docs/validation.md)
for timings and quality limitations.

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
