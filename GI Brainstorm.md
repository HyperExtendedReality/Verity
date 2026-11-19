The future of real-time global illumination (GI) is moving decisively away from both **pure static baking** (which explodes storage and iteration times—for instance, id Software noted at SIGGRAPH 2025 that baking *DOOM: The Dark Ages* with older tech would have required up to 110 GB of data and 68 days of processing) and **brute-force per-pixel path tracing** (which relies heavily on aggressive temporal accumulation and AI/spatiotemporal denoising that introduce smearing and ghosting).

Instead, cutting-edge GI architectures from 2025–2026 converge on one overarching principle: **decoupling ray tracing from screen pixels via adaptive, multi-scale world-space radiance caching**, paired with **asymmetric spatial/angular resolution** and **ultra-cheap geometry proxies**.

Below is a breakdown of the exact "superpower" (and core weakness) of each technique you listed, followed by a blueprint for a novel hybrid architecture that combines the best of all ten.

---

### 1. Extracting the Best (and Worst) of Each Inspiration

| Technique | Core "Superpower" to Steal | Primary Weakness to Avoid |
| :--- | :--- | :--- |
| **Irradiance Field Gather** *(UE 5.8 Lumen Lite)* | Places world-space clipmap radiance probes around visible pixels with probe occlusion and shares the cache across opaque and translucent/volumetric passes—roughly doubling Lumen's GI speed (~1–1.8 ms territory) with minimal camera-motion noise. | Loses fine local occlusion (behind pillars/small corners) and shows temporal lag or cell jitter when lights change rapidly. |
| **XFSDK Hybrid SDF GI** *(AK Plugins / UE 5.8)* | Combines DDGI-style volumes with a G-buffer-seeded surfel cache, per-volume update budgets (*Responsive / Delayed / Baked* + *"Start From Bake"*), and a clean final gather inspired by *DOOM: The Dark Ages*—delivering crisp GI without needing TAA/TSR or HWRT. | Relies on Global Distance Fields, which struggle with very thin geometry, foliage, or deforming meshes without extra proxy work. |
| **Surface Cache GI (SCGI)** *(Unity 6.7)* | Discretizes the 5D space of **(3D position + 2D normal)** into camera-distance-adaptive surface patches with zero UV unwrapping or baking, plus a compute SWRT fallback for GPUs without hardware RT. | Needs several frames to accumulate when scenes/patches appear; currently struggles with alpha cutouts, animated surfaces, and high quality settings can get costly. |
| **Hash Cache** *(SHaRC, GI-1.x World Hash, VisCache)* | $O(1)$ GPU spatial hashing quantized by `(pos, normal, LOD)` avoids dense 3D grids or UV atlases. Variants like **VisCache** also store variance/visibility predictions as *control variates* to skip redundant rays entirely. | Hash collisions if poorly tuned, and raw flat hash cells appear blocky or drop to black on sudden camera cuts/disocclusions without interpolation. |
| **Two-Level Radiance Caching** *(AMD GI-1.0/1.2, On-Surface 2L)* | Separates a **Primary Cache** (screen/surface probes capturing *incoming* directional radiance with ReSTIR) from a **Secondary Cache** (world-space hash grid storing *outgoing* radiance, extended to 2nd-bounce cells in GI-1.2). | Primary screen probes suffer on disocclusions; naive secondary caches lose energy if they only cache direct light. |
| **Split Radiance Cascades** *(Freeman & Sannikov, 2026)* | Solves 3D Radiance Cascades' memory explosion by storing world-space probes sparsely in a hashmap near visible surfaces and **splitting rays** traced from surfaces into distance-based radiance/transmittance ($\beta$) intervals. | Probe hashing overhead and trilinear cascade merging can still get bandwidth-heavy across wide open vistas if cascade LODs aren't tightly bounded. |
| **Holographic Radiance Cascades (HRC)** *(2025)* | Maintains tight probe spacing *perpendicular* to ray segments at every level and merges short segments anisotropically—resolving razor-sharp hard shadows and volumetrics deterministically with zero stochastic noise and no SDF. | Currently formulated for 2D/heightfield transfer; full naive 3D extension explodes memory exponentially unless made sparse. |
| **PXTGI** *(PhysX5 Proxy Traced GI, 2026)* | **Geometry Proxy Unification**: traces compute SWRT rays directly against existing PhysX5 physics collider primitives (boxes, capsules, convex hulls) plus explicit emissive candidates, bypassing SDF cascade marshalling at ~half the cost of Godot's SDFGI. | Physics colliders often omit visual-only details (trim, foliage, thin decals), requiring screen-space/probe blending to fill small-scale gaps. |
| **Hierarchical Voxel Irradiance Caching (HVIC)** *(2026)* | Stores an irradiance hierarchy (octree) where parents pull converged samples from children. On disocclusion, low-sample leaf nodes blend from **converged parent nodes**—producing momentarily softer resolution rather than noise or ghosting—while 1-ray hits multiply by cached albedo for infinite diffuse bounce. | Uniform voxel/octree boundaries look blocky without careful trilinear neighbor filtering and struggle with non-voxel arbitrary triangle meshes unless mapped to a surface/hash hierarchy. |
| **Image Space Photon Mapping (ISPM)** *(McGuire & Luebke)* | **Light-to-Eye Bidirectional Injection**: rasterizes a light-space bounce map for initial photon emission, traces bounces, and scatters bounded filter volumes into the camera G-buffer. Effortlessly captures **caustics** and intense indirect bounce that camera rays almost never find. | Traditional CPU-side 3D photon bouncing bottlenecks on dynamic geometry; pure scattering can cause overdraw if photon filter bounds aren't clamped. |

---

### 2. A Novel Architecture: **Split-Proxy Hierarchical Hash Cascades (SPH-GI)**

If you combine the strongest mechanisms above while canceling out each method's blind spots, you arrive at a novel, hardware-agnostic GI pipeline designed to hit **< 1.5–2.5 ms on consoles/mid-range GPUs** while approaching offline path-traced fidelity and native-resolution sharpness.

Here is how the 5 stages work together:

#### Stage 1: Unified Proxy-Hybrid Ray Tracing *(PXTGI + Screen Traces + HWRT/SDF)*
Never trace raw multi-million-triangle meshes for diffuse GI when you don't have to:
* **Near-field (0–1.5 m):** Trace Screen-Space (Hi-Z) rays directly against the G-buffer for sub-centimeter contact occlusion and thin decals.
* **Mid-field (1.5–30 m):** On compute-only GPUs (or handhelds), trace against **existing physics/simulation proxy primitives** (PXTGI's insight: convex hulls, capsules, boxes already uploaded for physics) augmented with a coarse Global SDF for terrain. If HWRT is available, swap only this mid/far step to a simplified RT-LOD BVH.
* **Result:** Dynamic doors, destructibles, and moving characters automatically cast accurate indirect shadows and color bounce with near-zero extra acceleration-structure build cost.

#### Stage 2: Hierarchical 5D Surface Hash Tree *(SCGI + Hash Cache + HVIC + XFSDK)*
Instead of storing irradiance in a flat 3D grid (which leaks) or a dense voxel octree (which wastes memory), store lighting in a **Multi-Level 5D Surface Hash Cache** keyed by quantized `(worldPos, normal, LOD_level)`:
* **Surface-snapped patches (SCGI):** Hash entries only exist on actual surfaces seen by camera rays or bounce rays, eliminating empty-air memory waste.
* **Parent-Child Mip Linking (HVIC):** Every fine-level patch (`LOD 0`, e.g., 10 cm) deterministically hashes to a coarser parent cell (`LOD 1`, 40 cm; `LOD 2`, 1.6 m). Each frame, child patches push weighted radiance up to their parent.
* **Zero-Noise Disocclusion Fallback (HVIC):** When the camera whips around and reveals a surface whose `LOD 0` hash cell has `sampleCount < threshold`, the shader smoothly blends up to `LOD 1` or `LOD 2`. You get **zero temporal ghosting and zero Monte Carlo noise**—newly uncovered corners simply sharpen from soft GI into crisp GI over 3–4 frames.
* **Optional Static Prior (XFSDK):** Coarse `LOD 2+` parent cells for static architecture can be pre-seeded ("Start From Bake") on disk, giving instant multi-bounce warmth on frame 1 while dynamic runtime updates overwrite them seamlessly.

#### Stage 3: Variance-Guided Ray Splitting & Anisotropic Cascades *(Split RC + HRC + VisCache + IFG)*
Instead of shooting uniform rays everywhere every frame:
* **Delta & Variance Scheduling (VisCache + IFG):** Each hash cell tracks a running variance and temporal delta score. Static, fully converged areas receive only 1–2 maintenance rays per several frames (control variates skip shadow/bounce rays where predicted visibility variance is near zero). Moving lights, opening doors, and disoccluded patches automatically steal 80%+ of the frame's ray budget.
* **Surface-to-Cascade Ray Splitting (Split RC + HRC):** From active surface patches/screen tiles, cast sparse low-discrepancy rays. As a ray travels distance $t$, **split its radiance and transmittance ($\beta$)** into distance-binned cascade intervals stored in the sparse hash structure.
  * Following **Holographic Radiance Cascades**, keep cascade `0` and `1` (near-field) spatially dense *perpendicular* to incoming directions with coarse angular bins, while distant cascades trade spatial density for high angular resolution. This eliminates light leaking through thin walls and preserves sharp indirect shadows without increasing total ray count.
* **Infinite Diffuse Bounce in 1 Ray (HVIC + GI-1.2):** When any ray hits a surface at distance $t$, it does not trace a secondary ray. It evaluates `directLight + (hitPatchHashCache.outgoingRadiance * hitAlbedo)`, feeding multi-bounce energy back into the cascade intervals.

#### Stage 4: Targeted Light-Space Photon Injection *(ISPM + PXTGI Emissives)*
Pure camera-driven caches struggle with **caustics** (sunlight bouncing off water/glass or a bright spotlight hitting a colored floor into a dark ceiling) because camera bounce rays rarely hit tiny bright spots.
* Run a **GPU-only micro-ISPM pass** exclusively for the top $N$ strongest analytic lights or bright emissive candidates (tracked like PXTGI): rasterize a tiny $256\times256$ light-space bounce map (capturing refractive/reflective/bright bounce origins), trace 1 short proxy ray per bright texel on the GPU, and **splat the resulting energy directly into the 5D Surface Hash Cache** rather than screen-space photon volumes.
* Because the hash cache is persistent in world space, caustics and bright bounce spots accumulate smoothly across frames and affect both diffuse GI and glossy reflections automatically.

#### Stage 5: Two-Level Specular + Leak-Free Final Gather *(Two-Level RC + idTech 8 + XFSDK)*
* **Primary Screen/Final Gather (idTech 8 / GI-1.0):** For every pixel (or quarter-res bilateral tile), query the Split Hash Cascades using **normal-aligned, depth-plane-weighted trilinear interpolation**—ignoring any probe/patch behind the pixel's tangent plane to kill light leaking.
* **Glossy Reflections (PXTGI + Two-Level):** Smooth surfaces ($roughness < 0.25$) trace 1 reflection ray against the unified proxy/HWRT scene and evaluate outgoing radiance directly from the 5D Hash Cache at the hit point (handling off-screen multi-bounce reflections cleanly like GI-1.2). Rougher surfaces blend directly into the high-angular upper cascades of the Hash Cache.
* **TAA-Independent Output (XFSDK):** Because variance is absorbed inside the hierarchical world-space hash cascades and near-field splitting rather than smeared across screen history, the final image needs only a lightweight spatial edge-aware filter (like idTech 8) and optional SMAA 2TX, staying razor-sharp in motion.

---

### Why This Is the Sweet Spot for Quality & Performance

1. **No TAA/TSR Dependency:** Unlike traditional path-traced RTGI, moving objects don't leave dark trails or noisy disocclusion fizz—the HVIC-inspired parent-hash fallback turns disocclusion noise into clean, soft lighting that sharpens in milliseconds.
2. **Hardware Agnostic:** By unifying PXTGI's collider/SDF compute proxies with SCGI's compute fallback, the exact same GI pipeline scales from Switch 2 / mobile / integrated GPUs up to high-end HWRT desktop cards (simply swapping the mid-field ray backend).
3. **True All-Frequency Lighting:** Split Radiance Cascades + HRC principles fix the blurry "muddy corners" flaw of DDGI and Irradiance Field Gather, while micro-ISPM injection solves the high-frequency caustic/bounce failure mode of camera-only ray tracing.
4. **Fixed, Budgetable GPU Time:** By driving ray allocation through VisCache-style variance/delta priority queues, you can lock GI to a strict **1.5 ms frame budget**—spending rays only where the world is actively changing.


Beyond the **cache-and-cascade** paradigm from the previous approach, graphics research (especially across SIGGRAPH, ECCV, and GDC 2025–2026) is actively pushing in **five completely different architectural directions**. Each trades memory, ray budgets, and AI compute differently to conquer the same quality-vs-performance wall.

---

### 1. Online Neural Radiance Caching + ReSTIR Path Guiding (*"The Self-Learning Path Tracer"*)
Instead of manually engineering hash grids, probe clipmaps, or cascade textures—all of which suffer from discretization seams, hash collisions, or leaking—let a tiny neural network *be* the radiance cache and let path reservoirs *teach* rays where to look.

* **How it works:**
  * **Neural Suffix Cache:** Trace only 1–2 short bounces per pixel, then terminate into a tiny, online-trained MLP (pioneered by NVIDIA's NRC, expanded to **Mobile NRC** by Samsung/Traverse at SIGGRAPH Asia 2025 and **AMD FSR Redstone**). A tiny fraction of pixels trace deeper paths each frame to continuously train the network live in VRAM as lights and doors move.
  * **ReSTIR Path Guiding (`ReSTIR-PG`) & `ReSTIR PT Enhanced` (2026):** Rather than shooting blind hemisphere rays, use **ReSTIR-PG** to extract directional guiding distributions from the previous frame's resampled paths so new rays aim straight toward bright doorways, cracks, and glossy bounces. Meanwhile, **2026's ReSTIR PT** unifies direct and indirect lighting into a single reservoir and cuts spatial reuse cost in half via reciprocal neighbor selection.
  * **Multi-Layer Reservoir Splatting (SIGGRAPH 2026):** To kill ReSTIR's notorious disocclusion noise when objects move, maintain lightweight hidden depth layers so previously occluded reservoirs splat cleanly into newly revealed pixels.
* **Secret Weapon (Asset-Level Neural Transport):** For complex assets like foliage, skin, or velvet—where internal scattering takes 10+ bounces—pre-bake **8D Neural Asset Light Transport (`8DNA`, SIGGRAPH 2026)** or **Object-Centric Neural Transfer (`TransGI`, 2025)** directly onto the asset so multi-scattering resolves in one inference call.
* **The Catch:** Requires dedicated matrix/tensor hardware (FP16/INT8 systolic arrays) to train and evaluate fast enough, and online neural caches can exhibit subtle "boiling" or lag during extreme instantaneous lighting cuts.

---

### 2. Anisotropic 3D Gaussian Surfels (*"No Grids, No Voxels, No Cubic Seams"*)
Voxel octrees (like HVIC), clipmaps (like Lumen IFG), and quantized spatial hashes all force curved, organic geometry into rigid, axis-aligned cells. A completely different path is representing indirect illumination with **oriented, anisotropic 3D Gaussian surfels**.

* **How it works:**
  * **Curvature-Adaptive Radiance Elements:** Spawn flattened, disc-like 3D Gaussians along world surfaces (similar to the surface light transport models emerging in **Dynamic 3D Gaussian GI** and **GHPT** in 2025–2026). Large flat walls are covered by just a handful of wide, stretched Gaussians; intricate cornices and doorways receive tiny, dense Gaussians.
  * **Spherical Gaussian (SG) Lobes per Splat:** Each Gaussian stores directional outgoing radiance using 2–3 compact Spherical Gaussian lobes (handling both diffuse and low-to-mid roughness glossy bounce).
  * **Rasterized or Cone-Traced Final Gather:** Because 3D Gaussians support ultra-fast hardware/compute splatting *and* stochastic ray-cloud tracing, you can evaluate near-field indirect lighting by splatting/accumulating nearby Gaussians directly over the G-buffer—or trace sparse rays against them without needing a triangle BVH or dense voxel grid.
* **Why it's compelling:** Zero UV unwrapping, zero grid-boundary popping, and optimal memory density (spending bytes on geometric detail rather than empty space or uniform walls).
* **The Catch:** Maintaining and splitting/merging Gaussians dynamically on deforming characters or destructible meshes adds structural management overhead compared to an $O(1)$ hash map.

---

### 3. Residual / Control-Variate GI (*"Analytic Proxy + Sparse Ray Correction"*)
Instead of asking ray tracing to solve 100% of the lighting integral from scratch, compute **85–90% of the answer deterministically**, and spend rays *only* on the mathematical difference between the cheap approximation and physical reality.

* **How it works:**
  * Express indirect radiance as a **control variate** decomposition:
    $$L_{\text{indirect}} = L_{\text{proxy}} + \underbrace{(L_{\text{traced}} - L_{\text{proxy}})}_{\text{low-variance residual}}$$
  * **The Analytic Proxy ($L_{\text{proxy}}$):** Compute a fast, leak-free baseline using **Many-Light ReSTIR-Sampled Virtual Point Lights / Shadow Maps** or a coarse clipmap field (like UE5.8 Irradiance Field Gather).
  * **The Residual Tracer:** Shoot 0.5–1 ray per pixel (paired with **Spatio-Temporal Control Variates with ReSTIR**, SIGGRAPH 2026) to measure how much the true scene deviates from the proxy—such as sharp contact shadows, glossy reflections, or a door swinging shut.
* **Why it's mathematically brilliant:** If your analytic proxy already gets the smooth global color bleed right, the *variance* of the residual term collapses dramatically. One stochastic ray per pixel converges with the cleanliness of 16+ unguided rays—and if you drop to low settings on a handheld, you just turn down the residual ray count and keep the deterministic proxy intact!
* **The Catch:** Designing a proxy $L_{\text{proxy}}$ that is cheap, smooth, *and* strongly correlated with the true bounced light (especially for glossy materials) is nontrivial; poorly correlated proxies can actually double noise.

---

### 4. Sparse-Hint Generative & Attention GI (*"Physical Anchors + 1-Step Neural Synthesis"*)
This is the boldest departure from classical transport: replacing heavy multi-pass integration and denoising with **learned perceptual lighting models conditioned on ultra-sparse physical ray hints**.

* **How it works:**
  * **Ultra-Sparse Physical Anchors:** Trace only ~1 ray per $8\times8$ or $16\times16$ pixel tile (or pull a tiny proxy bounce buffer) strictly to provide *physically accurate color, intensity, and shadow hints*.
  * **Single-Step Latent Synthesis / Spatial Attention:** Feed the G-buffer intrinsics (albedo, normals, roughness, direct light) plus those sparse physical GI hints into a **1-step distilled latent diffusion UNet** or **Lightweight Attention-Based Indirect Illumination** network (pioneered by AMD Research at SIGGRAPH/ECCV 2026).
  * **Temporal VAE Decoder:** Instead of expensive multi-frame Monte Carlo accumulation that smears moving objects, a Temporal VAE decoder reprojects the *previous frame's output* via motion vectors directly into the final decoding step to lock down temporal stability. (Similar path-free inference is also emerging for **volumetric light transport**.)
* **Why it's disruptive:** Inference cost is **completely decoupled from scene triangle count, light count, and bounce depth**. A room with 500 colored emissive strips and 8-bounce glossy walls costs the exact same milliseconds as an empty box.
* **The Catch:** Artistic control and physical determinism. Because the network synthesizes missing structure from learned priors, thin dynamic off-screen light sources (e.g., a red laser behind the camera) won't bounce properly unless your sparse physical ray hints catch them first.

---

### 5. GPU Work-Graph Wavefront GI (*"The Zero-Waste Dynamic Pipeline"*)
Sometimes the bottleneck isn't the math—it's how GPUs schedule GI. Standard engines run rigid, barrier-separated compute/RT passes (`TracePass -> CachePass -> GatherPass -> DenoisePass`) where half the GPU warp lanes sit idle waiting on divergent rays or empty pixels.

* **How it works:**
  * Using **DirectX 12 GPU Work Graphs** combined with **Shader Execution Reordering (SER)**, eliminate static full-screen GI dispatches entirely.
  * **Demand-Driven Producer-Consumer Nodes:** A lightweight screen analyzer node inspects G-buffer gradients, roughness, and temporal error, then *dynamically spawns* work graph tasks only where needed:
    * Smooth mirror pixel? Emit directly to a `SpecularSER_Trace` node.
    * Flat matte wall matching last frame's irradiance? Emit **0 rays**; read cache and finish.
    * Ray hit a dynamic translucent/subsurface mesh? The hit shader itself enqueues a continuation node rather than writing intermediate hit buffers to VRAM.
* **Why it matters:** This isn't a new lighting equation—it's a *multiplier* on almost any GI algorithm (Split Radiance Cascades, Two-Level Caching, or ReSTIR PT). By eliminating intermediate VRAM round-trips and empty threads, it unlocks **RTX Mega Geometry**-scale scenes at console frame budgets.
* **The Catch:** Requires modern Work Graph + SER driver/hardware support and fundamentally changes engine profiling and memory budgeting.

---

### Which Future Wins Where?

| Target Use Case | Best Paradigm | Why |
| :--- | :--- | :--- |
| **AAA Path-Traced PC / Next-Gen Consoles** | **Online NRC + ReSTIR-PG + Work Graphs** | Scales cleanly toward true ground-truth light transport (caustics, multi-bounce glossy, `8DNA` assets) without grid artifacts. |
| **60–120 FPS Cross-Platform / Switch 2 / Mobile** | **Control-Variate Residual GI** (or **SPH-GI** with physics proxies) | Predictable sub-2 ms ceiling; graceful fallback to the deterministic proxy when ray budgets tighten. |
| **Stylized / Dynamic / Procedural Worlds** | **Anisotropic 3D Gaussian Surfels** | No baking, no UVs, adapts naturally to arbitrary procedural or destructible shapes without voxel stair-stepping. |
| **Heavy NPU / Tensor-First Future GPUs** | **Sparse-Hint One-Step Generative GI** | Fixed $O(1)$ inference time regardless of geometric or light complexity, turning tiny ray budgets into filmic GI. |

1. Anchor screen-space probes at the depth of the current fragment (a surface point on a mesh), forming a probe lattice; do this for all cascades.
2. Shoot rays from each probe (octahedral map, into the TLAS). The ray direction oscillates over 48 frames (deterministically), basically shooting approx. 432 well-distributed sub-directions per probe over 48 frames that get accumulated (see below).
 2.1. At the hitpoint, shoot shadow ray.
 2.2. At the hitpoint, read the (world-space) radiance cache and cosine-convolve it to get irradiance. The radiance cache is a hashgrid. Radiance is stored projected onto SH.
 2.3. Shade the ray as in the final composite shader (diffuse only), taking the irradiance computed from the radiance cache and shadowing into account (this simulates multiple bounces).
3. Merge cascades down to c0.
4. For each c0 probe, project the merged radiance into SH, then reproject the probe from last frame and EMA-accumulate the SH coefficients into a (screen-space) radiance map. I use 3x3 directions, enough if temporal accumulation is used.
5. Finally, in the composite shader, read the radiance (SH) at the fragment, convolve it to irradiance, and do the usual shading.
6. Scatter the radiance from the radiance map into the ws radiance hashgrid, making it persistent (that's actually an issue with lighting changes - it means only the current view supplies information (that we anyway already have), but it also means old lighting data gets stale after light changes, requiring a good cache eviction strategy).
7. Hashgrid SH cache eviction (can also come before 6). The eviction strategy is very dumb currently: I evict cells that are far away if they are not looked at.

**NILE (*Nested Interleaving of Low-Dimensional Elements*, Ahmed, Pharr, Ostromoukhov & Huang, 2026)** is a **drop-in modular meta-sampler**. Rather than changing your lighting math or adding neural passes, it upgrades **how you generate random/quasi-Monte Carlo numbers** across pixels and GI bounces—directly improving image quality, memory usage, and shader speed at low sample counts.

In a **mobile-ready GI pipeline** (PTGI, ReSTIR, SSGI, DDGI, or Radiance Cascades), NILE provides six concrete wins:

---

### 1. Eliminates Multi-Bounce Checkerboarding & Aliasing at Low SPP
Standard global samplers (like `Sobol'`) break down after screen-space inversion: the effective generator matrices lose their upper-triangular structure, leaving half the domain empty in later dimensions and causing **checkerboard artifacts on indirect bounces and glossy reflections**. Meanwhile, standard `Z-Sobol` gets noisy in high-dimensional indirect regions, and `SZ` can alias in reflections.
* **How NILE helps:** By associating each 2D/3D subspace (e.g., bounce 1 BRDF, light selection, bounce 2 BRDF) with a single dimension of an `SZ` low-discrepancy sequence and distributing samples via Z-indexing, NILE keeps **every bounce well-stratified while still preserving global high-dimensional coverage**.
* **Mobile payoff:** At **0.25–4 spp**, the `O2` and `BN` (Blue Noise) NILE variants produce clean, blue-noise-like screen-space error instead of structured checkerboards or clumpy aliasing—making **TAA, SVGF, and spatial denoisers** converge noticeably faster.

### 2. Zero-Table (`O2`) & Tiny-LUT Memory Footprint + Cheaper High Dimensions
Mobile GPUs have tiny L1/constant caches and tight register budgets. NILE actually **reduces** sampler cost compared to high-dimensional Sobol/SZ:
* **`O2` NILE is completely tableless:** Using diagonal factoring, it evaluates via fast bitwise arithmetic with **zero lookup tables** and wins on both speed and memory.
* **`StART` breaks even at 16D:** It needs only a **1 kB lookup table** (equivalent to 8 dimensions of 32-bit Sobol tables). Because each NILE element packs a *2D* subspace into *1* dimension of `SZ`, it breaks even at `8 × 2D` (16D) and **saves memory for every GI dimension beyond that**.
* **Stays inside the fast 16D core of `SZ`:** The paper notes that `SZ`'s compute cost rises sharply after the first 16 dimensions. Because NILE nests 2D/3D elements inside single `SZ` dimensions, a **32D–48D multi-bounce path tracer stays inside `SZ`'s cheap ≤16D regime**.

### 3. Native 3D Sequences (`O2` / "One-Three") for Every GI Bounce
Real-time GI decisions are often **3D**, not 2D—for example:
* `(u, v)` outgoing ray direction + `w` (BSDF lobe choice / Russian roulette / MIS), or
* `(u, v)` position on an area light + `w` (light index selection).

The paper shows that nesting a compliant 2D `(0,2)`-sequence inside an `SZ` pair automatically produces a **3D Sobol `(1,3)`-sequence** (arguably the best 3D sampling sequence available) for **every 3D subspace in your pipeline**—fixing the 3D-sample alignment issue of native `SZ`.

### 4. Super-Pixel Complementarity (Built for ReSTIR Spatial Reuse & Quarter-Res GI)
Because NILE distributes samples using Morton Z-indexing over power-of-two intervals, **every 2×2 or 4×4 block of adjacent pixels automatically forms a complementary stratified super-pixel** covering the entire domain.
* **Mobile payoff:** If you run **quarter-res PTGI/SSGI** or **ReSTIR spatial reuse** (sharing candidate paths across 2×2 or 3×3 neighbors), NILE ensures neighboring pixels did *not* wastefully trace rays into the same strata. Every spatial neighbor brings genuinely new directional/light coverage.

### 5. Mix-and-Match Domain Elements per GI Subspace
NILE lets you plug custom low-dimensional distributions into specific GI dimensions without building a new global sampler:
* **Emissive Triangles & Area Lights:** Plug in native **Triangle** (Basu–Owen) or **Disc** (Shirley–Chiu / Christensen) NILE elements directly for emissive mesh triangles or spherical/disc lights—avoiding square-to-triangle/disc warping distortion while staying globally interleaved with bounce rays.
* **Smooth vs. Complex Integrands:** You can assign **Blue Noise (`BN`)** or **Jittered Grid (`JG`)** to smooth diffuse/AO/SSGI subspaces, and **`O2`** or **`StART`** to high-variance specular reflections and multi-bounce paths.

### 6. Traceable Indexed Samples for Neural GI / Path Guiding
In Section 6, the authors specifically highlight that NILE’s traceable indexed samples align naturally with ML pipelines for learning-aware sampling. For example, if you feed candidate rays into **Mobile NRC** or **ReSTIR-PG**, the clean stratification across pixel neighborhoods gives smoother, lower-variance training suffixes per frame.

---

### Practical Mobile Caveats (How to Use It Right)
1. **Pick `O2` first on mobile GPU shaders:** Recursive ART tree traversal (`StART`/`BN`) is slightly less GPU-friendly than pure matrix bit-ops. The **tableless `O2` NILE** variant is pure bitwise math (fastest on mobile ALUs, zero LUT cache misses), while `StART`/`BN` is best saved for subspaces where extra stochasticity is needed to break severe aliasing.
2. **Pair NILE with Temporal Scrambling (`Z++`):** The NILE paper evaluates progressive sampling (`1, 2, 4, ... spp`) across static frames. In a real-time 1 spp mobile engine, combine NILE's spatial/subspace interleaving with **temporal negative correlation (`Z++`)** or **frame-index Z-offsetting** so consecutive frames don't repeat the exact same sample offsets during TAA/temporal accumulation.
3. **Avoid `JG` (Jittered Grid) for bounce 2+:** As shown in Section 5.2–5.3, `JG` works fine for smooth primary/background regions but degrades badly in reflections and complex multi-bounce GI. Use **`O2`**, **`BN`**, or **`StART`** for GI bounces.

A. Online Neural Radiance Caching (Mobile NRC) — Best for Fully Dynamic Scenes
Model requirement: Zero pre-training. The network initializes with random weights and continuously self-trains on the GPU while the game runs[9][10].
What you have to build/provide:
A Ray/Path Tracer (HW RT, ROMA, or SDF): Every frame, ~95%+ of pixels trace short 1-bounce paths and terminate into the MLP. A small fraction (e.g., 2–4% of pixels) trace 1–2 extra bounces to generate "ground truth" training targets for the MLP's backpropagation step[9][23].
Input Feature Buffers: World position x, outgoing/scatter direction ω, surface normal n, roughness, and diffuse/specular albedo[9][11].
Feature Encoders: Frequency encoding (sine/cosine or triangle waves) or a Multiresolution Hash Grid for position, plus One-Blob encoding for angles/roughness[9][12][24].
Fused FP16 Compute Shaders: Both the forward pass (inference) and backward pass (Adam/SGD weight update) must be written as fused GPU shaders (using shared memory / cooperative vectors / tensor ops where supported) so weights stay out of slow system DRAM[10][25][26].
Stability Tricks: Use the techniques from Taming Optimization Variance (2026)—like proper HDR loss scaling and activation choices—so the tiny MLP doesn't flicker or diverge under bright lights.

Drawing from **NVIDIA’s Real-Time Graphics Research publications (2019–2026)** alongside recent mobile and non-NVIDIA GI breakthroughs (such as **Split Radiance Cascades**, **Radiance Cascade Path Guiding**, and **Mobile NRC**), this compilation covers specific algorithmic improvements and how to combine them into **mobile-ready Global Illumination (GI)** pipelines.

Because mobile GPUs are severely constrained by **memory bandwidth, thermal budgets, ray-traversal divergence, and register pressure**, the biggest wins come from reducing rays per pixel, keeping warps coherent, eliminating BVH overhead where possible, and unifying passes.

---

## Part 1: GI Improvements from NVIDIA RTR Research (Applicable to Mobile)

### 1. Low-Cost & Software-Friendly Visibility (No Heavy BVH Required)
* **Ray-Aligned Occupancy Map Array (ROMA)** *(Zeng et al., EGSR 2023)*
  * **What it does:** Rasterizes dynamic scenes once per frame into a ray-aligned occupancy map array, enabling **constant-time, low-divergence visibility queries** without building or traversing a BVH (2.5×–10× faster than Distance Fields at equal storage).
  * **Mobile GI benefit:** Mobile GPUs excel at rasterization and suffer heavily from BVH divergence and dynamic BLAS/TLAS rebuilds. ROMA is ideal for driving short-to-medium GI bounce visibility or probe updates on devices without hardware RT (or to offload secondary rays on hardware-RT mobile chips).
* **Tight Bounding Boxes for SDF Bricks & Voxels** *(Hansson-Söderlund & Akenine-Möller, EG 2023)*
  * **What it does:** Computes tight AABBs around active SDF voxels/bricks, cutting real-time path tracer frame times by **20–31%**.
  * **Mobile GI benefit:** If you use signed distance fields for software GI tracing (e.g., Lumen-style or Radiance Cascades), tight brick bounds dramatically cut empty-space ray-marching ALU and texture fetches.
* **ReSTIR-Sampled Shadow Maps + Imperfect Shadow Maps (ISMs)** *(Zhang et al., EG 2025)*
  * **What it does:** Uses ReSTIR to select only the highest-contributing lights per frame for full-resolution shadow maps, while rendering all remaining lights with ultra-cheap Imperfect Shadow Maps.
  * **Mobile GI benefit:** Lets you drive multi-light direct illumination and Virtual Point Light (VPL) indirect bounces on mobile without tracing per-light shadow rays.

### 2. Making Path-Traced GI (PTGI / ReSTIR) Lean Enough for Mobile
* **Unifying DI + GI & Halving Spatial Reuse Cost (*ReSTIR PT Enhanced*, Lin et al., I3D 2026)**
  * **What it does:** Achieves a **2–3× speedup** and lower error over baseline ReSTIR PT by:
    1. **Unifying Direct Illumination and GI into the same reservoir** (cuts memory footprint and DRAM bandwidth in half).
    2. **Reciprocal neighbor selection** (halves spatial reuse cost).
    3. **Footprint-based reconnection criteria** (prevents fireflies) and **duplication maps** (reduces temporal correlation).
  * **Mobile GI benefit:** Bandwidth and reservoir read/write traffic are the #1 bottleneck for ReSTIR on tile-based mobile GPUs; unified reservoirs + reciprocal reuse directly attack that bottleneck.
* **Compatibility-Guided Neighbor Selection** *(Junkins et al., HPG 2026 Best Paper)*
  * **What it does:** Smarter spatial neighbor selection that cuts SMAPE error by **6–29%** and temporal covariance by **22–49%** for only a **2–5% cost**.
  * **Mobile GI benefit:** When mobile budgets restrict you to 1–2 spatial reuse taps per pixel, picking *compatible* neighbors gives massive quality gains almost for free.
* **Stochastic Pairwise MIS for Large-Kernel Reuse** *(Hedstrom et al., EG 2026)*
  * **What it does:** Enables unbiased reuse across large spatial neighborhoods without the quadratic cost of standard MIS weights.
  * **Mobile GI benefit:** Crucial when running PTGI at **quarter-res or 0.25–0.5 spp**, allowing sparse hits to propagate smoothly over wide areas.
* **Multi-Layer Reservoir Splatting for Disocclusions** *(Hong et al., SIGGRAPH 2026)*
  * **What it does:** Tracks a sparse set of active screen-space depth layers and splats reservoirs across them using depth-range checks—eliminating most extra visibility rays during disocclusion.
  * **Mobile GI benefit:** Keeps low-spp PTGI temporally stable during fast camera movement without expensive disocclusion ray bursts.
* **LoD-Aware ReSTIR** *(Wang et al., SIGGRAPH 2026)*
  * **What it does:** Maps surface points across differing mesh topologies so ReSTIR temporal reuse doesn't invalidate when geometric LoDs switch.
  * **Mobile GI benefit:** Mobile renderers rely heavily on aggressive LoD transitions; this prevents temporal GI from flickering or resetting every time an asset swaps LoD.
* **ReSTIR Path Guiding (ReSTIR-PG)** *(Zeng et al., SIGGRAPH Asia 2025)*
  * **What it does:** Fits lightweight guiding distributions from the previous frame’s accepted ReSTIR paths to generate far better initial candidate rays on the next frame.
  * **Mobile GI benefit:** When you can only afford 1 bounce ray per pixel (or per 2×2 block), wasting that ray on dark geometry is fatal. Closing the ReSTIR → guiding loop boosts effective sample efficiency.

### 3. Neural Caching, Embeddings & Basis Functions (Amortizing Multi-Bounce GI)
* **Real-Time Neural Radiance Caching (NRC)** *(Müller et al., SIGGRAPH 2021)* + **Taming Optimization Variance in Compact Neural Shading Networks** *(Bitterli et al., SIGGRAPH 2026)*
  * **What it does:** Terminates paths after 1–2 bounces into a tiny online-trained MLP cache to represent infinite-bounce dynamic GI. Bitterli et al. (2026) specifically solve training instability in **small/compact neural networks** under HDR lighting via multi-instance optimization, better input parameterizations, and activations.
  * **Mobile GI validation:** Samsung and Traverse Research already proved this works in production mobile silicon (*Neural Radiance Cache Implementation on Mobile GPU*, SIGGRAPH Asia 2025 on Xclipse 950; *Towards Real-time Dynamic GI on Mobile*, SIGGRAPH 2026).
* **A Generalizable Light Transport 3D Embedding** *(Xu et al., SIGGRAPH 2026)*
  * **What it does:** Encodes scene point clouds (geometry + materials) into latent neural primitives; at render time, queries nearby primitives via nearest-neighbor + cross-attention to predict diffuse GI (or spatial-directional radiance for path guiding) **with zero per-scene training and zero ray-traced illumination cues**.
  * **Mobile GI benefit:** A compelling blueprint for a **zero-ray mobile GI tier** or a lightweight multi-bounce guiding prior on devices without ray tracing hardware.
* **Lightweight Neural Basis Functions** *(Xu et al., SIGGRAPH Asia 2022)*
  * **What it does:** Replaces Spherical Harmonics (too blurry) and Wavelets (too heavy/non-rotatable) with compact learned basis coefficients that support fast product integrals (BRDF × lighting × visibility) and rotation (10–40× faster than wavelets).
  * **Mobile GI benefit:** Upgrading DDGI/probe grids from L1/L2 SH to compact neural basis coefficients gives sharper indirect shadows and glossy bounce without blowing up probe texture memory.

### 4. "Free" Convergence via Sampling & Filtering
* **Z++: Temporal, Non-Uniform, and 3D Z Sampling** *(Ahmed, Pharr, Huang, SIGGRAPH Asia 2026)* & **Spatiotemporal Blue Noise Masks** *(Wolfe et al., EGSR 2022)*
  * **What it does:** Z++ adds **temporal negative correlation** across consecutive frames at **zero extra memory or runtime cost**, supports non-uniform importance-sampled distributions, and introduces base-4 3D low-discrepancy sequences.
  * **Mobile GI benefit:** Pure math/indexing upgrade—gives cleaner temporal accumulation for SSGI, DDGI ray directions, and PTGI at no extra GPU cycle cost.
* **Stochastic Texture Filtering (STF)** *(Pharr et al., I3D 2024)*
  * **What it does:** Swaps filtering to happen *after* shading/reconstruction (1 stochastic tap per pixel instead of 4–8 bilinear/trilinear taps during shading).
  * **Mobile GI benefit:** Cuts texture-unit pressure when sampling compressed neural textures, lightmaps, or multi-level probe volumes inside ray/path shaders.

---

## Part 2: Combining Methods into Hybrid Mobile GI Architectures

No single GI algorithm wins across all mobile tiers and scenes. Below are **three practical hybrid combinations** linking NVIDIA's research with **PTGI**, **Radiance Cascades**, **DDGI**, and **SSGI**.

### Hybrid A: **Split Radiance Cascades + PTGI-Lite (ReSTIR + NRC)**
*Target: Flagship Mobile with Hardware RT (e.g., Snapdragon/Adreno, Dimensity/Immortalis, Samsung Xclipse)*
* **Near-Field & Path Guiding via Radiance Cascades:**
  * Use **Split Radiance Cascades** *(Freeman & Sannikov, 2026)*—which adapts 3D Radiance Cascades to world-space using a **sparse probe hashmap** and **ray splitting**—or **Screen-Space Radiance Cascades** for noise-free contact GI and near-field bounce.
  * Combine with **RCPG (Radiance Cascade Path Guiding)** *(Ikkala et al., 2026)*: because Radiance Cascades are hierarchical, you can sample the **product of incident radiance × BSDF in a single draw**. That cuts PTGI ray budgets in half compared to separate BSDF + light importance sampling, and plugs straight into unbiased ReSTIR!
* **1-Bounce ReSTIR PTGI + Mobile NRC:**
  * Run quarter- or half-res **1-bounce ReSTIR PT Enhanced** (unified DI+GI reservoirs, reciprocal + compatibility-guided neighbor selection, Z++ temporal sampling).
  * Terminate the first bounce directly into a **Mobile NRC** (tiny FP16 MLP trained online with *Taming Optimization Variance* stabilization) for infinite multi-bounce GI.
* **Reconstruction:**
  * Use **Multi-Layer Reservoir Splatting** (or depth-aware bilateral SVGF) + **Neural Super Sampling/Denoising** to upsample cleanly from 0.25–0.5 spp.

### Hybrid B: **ROMA/SDF Software Tracing + Split RC / DDGI + SSGI**
*Target: Mid-Range Mobile Without Hardware RT (Pure Compute + Raster)*
* **Software Visibility Engine:**
  * Trace rays against a per-frame **ROMA** (for dynamic meshes) and/or **Tight-AABB Brick SDFs** (for static world geometry) instead of building a BVH.
* **World-Space Multi-Bounce Cache:**
  * Maintain an asynchronously updated **DDGI** volume or **Split Radiance Cascades** sparse probe hashmap (updating a subset of probes/cascades per frame with **Z++ / Spatiotemporal Blue Noise** ray directions).
  * *Optional quality upgrade:* Encode probes using **Lightweight Neural Basis Functions** (2022) or a **Neural Irradiance Volume** *(Coomans et al., 2026 — ~1–5 MB, ~1 ms inference)* if standard SH probes leak or look too flat.
* **Near-Field Detail:**
  * Run a cheap half-res **Screen-Space GI (SSGI)** or **2D/Screen-Space Radiance Cascades** pass for high-frequency contact shadows and small-prop color bleeding, falling back seamlessly to the world-space probes for off-screen / occluded lighting.

### Hybrid C: **Tile-Stratified Direct + Zero-Ray Neural GI**
*Target: Entry-Level / Strict 60–120 FPS Battery-Constrained Mobile*
* **Direct Lighting:**
  * Use **Tile-Based Stratified Reservoir Sampling** (as used in HypeHype's mobile pipeline) paired with **ReSTIR-Sampled Shadow Maps + ISMs** for fixed-cost dynamic many-light direct illumination.
* **Indirect GI:**
  * Skip multi-bounce path tracing completely. Evaluate either:
    * A lightweight **baked/runtime Neural Irradiance Volume** or **Generalizable 3D Light Transport Embedding** queried directly from G-buffer positions/normals, **or**
    * A low-probe-count **DDGI / Radiance Cascade** probe grid updated at low frequency + **Screen-Space Bent Normals / SSAO** for local occlusion.

---

## Summary Checklist for Mobile Implementation
1. **Unify reservoirs & passes:** Merge DI + GI reservoirs (*ReSTIR PT Enhanced*) to fit tile/cache budgets and avoid extra G-buffer passes.
2. **Replace BVH rays where possible:** Use *ROMA* or *Screen-Space / Split Radiance Cascades* so visibility checks stay warp-coherent.
3. **Never waste a candidate ray:** Use *ReSTIR-PG* or *RCPG (Radiance Cascade Path Guiding)* + *Z++ Temporal Z-Sampling* so 0.25–1 spp hits high-contribution paths.
4. **Keep temporal history alive:** Apply *LoD-aware ReSTIR mappings* and *Multi-Layer Reservoir Splatting* so LoD pops and camera pans don't cause GI noise explosions.
5. **Amortize bounces neurally:** Terminate rays at bounce 1 into a *FP16 Mobile NRC* (stabilized via *Bitterli et al. 2026*) instead of tracing bounce 2+.