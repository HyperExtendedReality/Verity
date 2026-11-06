# CPU physics experiments

Keep this log when trying performance changes. Record the workload, build,
hardware, result, behavior checks, and reason for keeping or rejecting each
idea. A result applies only to the recorded conditions; retry when the solver,
compiler, hardware, or workload changes enough to invalidate it.

## Comparison method

Run `cargo run --release -p verity-physics --example compare` with
`RAYON_NUM_THREADS=1`. The Box3D bridge uses one worker. Each scene alternates
five runs per engine and reports median and range for the first 60 steps and
all 300 steps. The benchmark checks body counts, finite transforms and linear
velocities, and body heights. After the timed steps it reports final maximum
floor penetration, accounting for cube rotation, RMS linear speed, and maximum
absolute horizontal position. The heavy-stack scene also reports maximum
body-to-body cube penetration using an oriented-box separating-axis test.
These snapshots do not measure transient errors, angular jitter, joint error,
or energy; the other scenes do not check body-to-body penetration. It
includes world synchronization. Active and
full-run timing should both be considered because settling and sleeping can
change the result.

The results below were collected on 2026-11-04 with an AMD Family 25 Model 117
CPU, Rust 1.99.0, release defaults, and one Rayon thread. The checkout was based
on `e9733b6` with substantial uncommitted changes; the tested solver file had
SHA-256 `349a6d61283c11443a96d8b5714036fdec08f666f2d9fd0a4cc3f1bde99110fb`.
The Box3D and Verity configurations use different solvers, so their timings
are not a quality-matched comparison. Machine load was not controlled.

## Attempts

### Four-wide instead of eight-wide body batches

In an instrumented temporary copy, forced the existing four-wide solver path
on an AVX2 CPU. For the 820-body pyramid, the active median was 1.282 ms
versus 1.357 ms for the earlier instrumented eight-wide run. The individual
run ranges overlapped, the runs were not alternated, and instrumentation adds
overhead. The final height checks matched. **Inconclusive; not adopted.**

### Force inline contact evaluation

In an isolated copy, added `#[inline(always)]` to `Manifold::constraint` and
`Manifold::eval_constraint`. Five alternating invocations of each executable
ran the pyramid scenes; each invocation itself reported the median of five
trials. For 820 bodies, the median of those reported active medians was
1.324 ms baseline and 1.303 ms with inlining. Their per-trial ranges overlapped;
the 210-body medians were 0.236 ms and 0.237 ms. Final reported heights
matched. **No reliable win; not adopted.**

An initial comparison accidentally used an instrumented baseline due to a
stale build artifact. It was discarded, and the clean baseline was rebuilt
before the alternating runs above.

### Force inline manifold stamping and dual updates

In an isolated copy, added `#[inline(always)]` to `Manifold::stamp` and
`Manifold::dual`. Five alternating invocations of each executable ran the
pyramid scenes, with five trials per invocation. For 210 bodies, the median
of the reported active medians was 0.239 ms baseline (0.234–0.240 ms) and
0.253 ms with inlining (0.249–0.263 ms). For 820 bodies, it was 1.378 ms
(1.351–1.426 ms) and 1.456 ms (1.416–1.495 ms). Parenthesized ranges here
are across the five reported medians, not individual trials. Final reported
heights matched. **Rejected:** the candidate was slower in both scenes.

### Four AVBD iterations instead of six

Set `VERITY_ITERATIONS=4` without changing the default. One invocation per
scene family ran five trials per engine. In the 820-body pyramid, active
median was 0.985 ms for Verity versus 0.928 ms for Box3D, while the full-run
median was 0.608 ms versus 0.333 ms. The prior six-iteration run measured
1.359 ms active and 0.319 ms full-run for Verity, but was not paired with
this experiment. For 2,744 rain bodies, four iterations measured 2.233 ms
active and 6.243 ms full-run versus Box3D's 4.483 ms and 8.394 ms. Verity's
lowest body height was 0.45 versus Box3D's 0.50; height alone does not
measure penetration for every orientation. **Not adopted:** iteration count
changes simulation behavior, and the tall pyramid's full-run cost increased.
Measure stability and penetration and run paired trials before reconsidering.

### Profiling the pyramid

Temporary per-stage timers attributed most measured active solver time to
the iteration and body-solve section rather than broadphase/narrowphase or
color setup. This was an instrumented diagnostic, not a benchmark result.
Further profiling should split contact assembly, dual updates, and block
solves before choosing the next optimization.

### Contact assembly diagnostic

Instrumented an isolated copy to time each `assemble_body` call, the dual
updates inside it, and each block solve. In the pyramid scenes, accumulated
assembly time dominated block-solve time; dual time was included in assembly.
The timers and atomic counters roughly doubled active-step time, so their
absolute timings and proportions are distorted. Use this only to guide further
profiling, not to claim a speedup.

### Independent SIMD lane reductions

Replaced `sum4`'s matrix transpose and vector additions with four individual
`Vec4::element_sum` calls in a temporary copy. Five alternating invocations per
version ran the pyramid benchmark. For 210 bodies the median of the reported
active medians was 0.237 ms baseline (0.234–0.241 ms) and 0.245 ms candidate
(0.242–0.247 ms). For 820 bodies it was 1.349 ms (1.287–1.416 ms) and
1.413 ms (1.365–1.462 ms). Ranges are across invocation medians; final reported
heights matched. **Rejected:** the candidate was slower on these workloads.

### Native CPU compilation

Built the same Rust code with `RUSTFLAGS="-C target-cpu=native"` and compared
it with the generic build in five alternating invocations per version. For
210 bodies, the median of the reported active medians was 0.239 ms generic
(0.234–0.240 ms) and 0.221 ms native (0.218–0.230 ms). For 820 bodies it was
1.386 ms (1.248–1.411 ms) and 1.222 ms (1.206–1.343 ms). Ranges are across
invocation medians; final reported heights matched. Rust flags also affect
other Rust code in the comparison; they do not establish which instructions
caused the difference. The native binary may not run on other machines.
**Diagnostic only:** do not set this flag for portable releases.

### AVX2 compilation of the body batch

In a temporary copy, marked `solve_batch` with an x86-64 AVX2 target feature
and ran it on an AVX2 machine. This experiment did not add the dispatch needed
for safe execution on other x86-64 CPUs and was never suitable to ship as-is.
Five alternating invocations per version gave 210-body active medians of
0.239 ms baseline (0.235–0.247 ms) and 0.241 ms candidate (0.236–0.242 ms).
For 820 bodies they were 1.382 ms (1.367–1.481 ms) and 1.368 ms
(1.354–1.384 ms). Ranges are across invocation medians; final reported heights
matched. **Inconclusive; not adopted:** the ranges overlap and a portable
dispatch would add complexity.

### Final-state quality checks and new scenes

Added rotation-aware floor penetration and RMS linear speed to the benchmark,
calculated after timing. Added a sliding cube and a stack with a top cube
100 times the density of the others. These scenes use the same initial geometry,
velocity, and material density in both engines. The checks do not establish
equal solver quality or cover joint behavior.

In one five-trial invocation at six AVBD iterations, the 820-body pyramid's
final maximum floor penetration was 0.0057 m in Verity and 0.0023 m in Box3D;
both had zero reported final RMS linear speed. A separate four-iteration run
reported 0.0066 m and 0.0024 m respectively. These are final snapshots, not
maximum penetration over the whole simulation.

The sliding cube stopped at x=3.8816 m in Verity and x=2.1131 m in Box3D;
both reported zero final linear speed. The heavy stack's final top heights
were 5.46 m and 5.24 m. In a subsequent invocation, maximum final overlap
between cubes was 0.0074 m and 0.0590 m, respectively. These results identify
behavior differences, not which engine is closer to a reference solution
overall. Timing for the tiny scenes was only
microseconds and should not be treated as a dependable speed comparison.

On 2026-11-05, a five-trial invocation for each `BOX3D_SUBSTEPS` setting of
4, 8, 16, and 32 reported final maximum cube overlap of 0.0596, 0.0589,
0.0585, and 0.0586 m in Box3D. Verity, still at six AVBD iterations, reported
0.0074 m in each invocation. Increasing substeps alone did not match this
particular final-state measure. The runs do not sweep other solver settings or
compare the two engines against a reference trajectory.

### Frictionless sliding reference

Added `free_slide`, a cube moving at 5 m/s on a horizontal floor with zero
friction. After 300 steps of 1/60 s its reference horizontal displacement is
25 m. A five-trial invocation reported 25.0001 m in Verity and 25.0000 m in
Box3D, with both still moving at 5.0000 m/s. This checks horizontal momentum
in one simple case. It does not establish the cause of the friction-enabled
sliding difference or validate a general friction model.

### Skip zero multi-contact reductions

In a temporary copy, tracked whether a body touched any multi-contact
manifolds and skipped reduction of the unused multi-contact accumulators when
it did not. Five alternating benchmark invocations per version, with five
trials per invocation, gave active medians of medians of 0.236 ms baseline
and 0.236 ms candidate for the 210-body pyramid, and 1.273 ms and 1.288 ms
for the 820-body pyramid. For the 512-body rain scene the active medians were
0.335 ms and 0.336 ms; for 2,744 bodies they were 2.466 ms and 2.830 ms,
with overlapping invocation ranges. The baseline used the older benchmark
executable; the candidate calculated the newer quality metrics after the
timed region. **Rejected:** no repeatable improvement, with a possible
regression on the larger rain scene.

### CPU sampling profile

Collected a Windows Diagnostics Hub CPU sampling trace of the pyramid
comparison with one Rayon thread and an optimized Rust build with debug symbols.
The largest named Verity entry was `solve_batch::closure$1`, the body assembly
closure. `Manifold::dual` and `Manifold::prepare` also appeared. This trace
includes both engines, setup, active steps, and sleeping steps; inlining can
attribute work to the enclosing function. It does not establish a per-stage
percentage or prove which operation inside the closure dominates. Trace and
symbol output are local build artifacts under `target/physics-profile`.

A follow-up profile forced contact constraint evaluation, force evaluation,
stamping, and dual update out of line in a temporary build. Stamping and
force evaluation had the largest named Verity sample weights, followed by
constraint evaluation. Forced calls change optimization and execution costs,
so this diagnostic ranks areas for experiments; it does not quantify the
production build's time distribution.

### Symmetric linear block assembly

In a temporary copy, replaced the matrix multiplication used to assemble the
linear contact block with six unique entries of the symmetric result. Five
alternating invocations per version, each containing five trials, gave
210-body pyramid active medians of medians of 0.241 ms baseline and 0.244 ms
candidate. For 820 bodies they were 1.261 ms and 1.373 ms; invocation ranges
overlapped (1.232–1.425 ms and 1.238–1.478 ms). Reported final heights matched,
but floating-point operation order changed. **Not adopted:** no reliable
speedup, with a possible regression on the larger pyramid.

### AVX2 and FMA compilation of the body batch

In a temporary copy, marked `solve_batch` with x86-64 AVX2 and FMA target
features and ran it on a CPU confirmed to support both. This test did not
include the runtime dispatch needed for portable execution. Five alternating
invocations per version produced 210-body active medians of medians of
0.238 ms baseline (0.233–0.241 ms) and 0.240 ms candidate (0.238–0.241 ms).
For 820 bodies they were 1.355 ms (1.316–1.385 ms) and 1.349 ms
(1.330–1.399 ms). Ranges are across invocation medians. Reported final heights
matched. **Not adopted:** no reliable speedup, and portable runtime dispatch
would be required.

### Prevent contact stamping inlining

In a temporary copy, marked `Manifold::stamp` as `#[inline(never)]` to test
whether a smaller assembly loop outweighed function-call overhead. Five
alternating benchmark invocations per version, each with five trials, gave
210-body pyramid active medians of medians of 0.235 ms baseline
(0.232–0.239 ms) and 0.241 ms candidate (0.238–0.243 ms). For 820 bodies
they were 1.250 ms (1.228–1.321 ms) and 1.342 ms (1.285–1.428 ms).
Ranges are across invocation medians. **Not adopted:** the candidate was
slower on both scenes; individual run ranges overlap.

### Skip unused friction-state masks during assembly

In an isolated copy, specialized contact evaluation so body assembly skipped
the static-friction state mask when static and kinetic coefficients matched;
the dual update still calculated it. Five alternating benchmark invocations
per version, each with five trials, gave 210-body active medians of medians
of 0.232 ms baseline (0.230–0.237 ms) and 0.231 ms candidate
(0.228–0.235 ms). For 820 bodies they were 1.227 ms (1.212–1.257 ms) and
1.217 ms (1.211–1.239 ms). Ranges are across invocation medians. **Not
adopted:** the differences were smaller than variation; other material
combinations and targets were not benchmarked.

### Contact angular-motion distribution

Counted exactly zero angular displacement for each endpoint whenever a
temporary instrumented copy constructed contact motion in the pyramid scene.
Some early 100,000-call intervals had many zeros, but later intervals in the
820-body pyramid commonly had about 1,300 zero A endpoints and zero B
endpoints. The instrumentation changed the workload's timing and its counters
aggregate across runs. **No optimization attempted:** skipping angular terms
based on zero motion would add branches to a path where most measured visits
had nonzero angular displacement.

### Reuse the current body during contact assembly

In a temporary copy, passed the already loaded body to `Manifold::stamp`
instead of looking it up again through the body pointer. Five alternating
invocations per version, each with five trials, produced 210-body active
medians of medians of 0.237 ms for both versions (invocation ranges
0.235–0.238 ms baseline and 0.234–0.239 ms candidate). For 820 bodies they
were 1.290 ms (1.282–1.292 ms) and 1.281 ms (1.253–1.305 ms). **Not adopted:**
the difference overlaps the measured variation.

### Cache tangential linear stiffness

An instrumented pyramid run found unchanged complete stiffness vectors in
roughly half of consecutive visits to a manifold, with some early groups
higher. This included visits across iterations and steps and was not a direct
cache-hit measurement. Based on that signal, a temporary solver cached the
tangential portion of the linear contact matrix whenever penalties were
prepared or updated, and added the current normal contribution during body
assembly. This changed floating-point association and added per-manifold data.

Five alternating pyramid invocations per version, each with five trials,
gave 210-body active medians of medians of 0.233 ms baseline
(0.233–0.235 ms) and 0.238 ms candidate (0.236–0.246 ms). For 820 bodies
they were 1.267 ms (1.236–1.272 ms) and 1.276 ms (1.271–1.337 ms).
Ranges are across invocation medians. Heavy-stack and slide final snapshots
matched the previously recorded values, but full correctness was not proven.
**Not adopted:** no speedup; the extra cache and update work may cost more
than the saved matrix operations.

### Four-contact assembly workload and temporary layout

An instrumented pyramid run counted contact stamps in groups of 100,000.
More than 99,000 stamps in the sampled groups had four contacts. Equal
tangential penalties were common in some early groups but nearly absent in
later groups, so a shortcut requiring equality would miss much of this path.
Counters and workload timing were diagnostic only.

In a separate temporary build, directly constructed the weighted angular-row
array instead of zero-initializing it and filling it in the accumulation loop.
Five alternating invocations per version, each containing five trials, gave
210-body active medians of medians of 0.222 ms baseline (0.220–0.228 ms) and
0.224 ms candidate (0.223–0.225 ms). For 820 bodies they were 1.159 ms
(1.155–1.189 ms) and 1.179 ms (1.175–1.239 ms). Ranges are across invocation
medians; final reported heights matched. **Not adopted:** there was no
reliable speedup and the larger scene may have regressed.

### Reuse the friction-cone membership mask

In a temporary copy, derived the sliding mask from the already computed
inside-cone mask, with an explicit NaN-bound guard, instead of repeating the
comparison against the squared bound. Five alternating pyramid invocations
per version, with five trials each, gave 210-body active medians of medians
of 0.232 ms baseline (0.230–0.233 ms) and 0.235 ms candidate
(0.233–0.238 ms). For 820 bodies they were 1.239 ms (1.223–1.249 ms) and
1.241 ms (1.232–1.305 ms). Ranges are across invocation medians; final
reported heights matched. **Not adopted:** no reliable speedup, and the
algebraic rewrite was not exhaustively validated for exceptional floats.

### Diagonal approximation of the linear contact block

In a temporary copy, replaced the full linear contact Hessian block for
multi-contact manifolds with its diagonal. This deliberately changes the
solver approximation and would require broad stability and accuracy testing
before any adoption. Five alternating pyramid invocations per version, each
with five trials, gave 210-body active medians of medians of 0.236 ms baseline
(0.234–0.238 ms) and 0.239 ms candidate (0.236–0.242 ms). For 820 bodies
they were 1.296 ms (1.282–1.304 ms) and 1.322 ms (1.302–1.329 ms). Ranges
are across invocation medians. Final pyramid heights and separate slide and
heavy-stack snapshots matched the previously recorded values. **Not
adopted:** the candidate did not improve speed, and those snapshots do not
prove equivalent physics behavior.

### Five AVBD iterations

Compared five and six iterations using a symbol-enabled release executable,
five alternating invocations per setting and five trials per invocation. In
the 820-body pyramid the median of active medians was 2.583 ms at six
iterations and 2.121 ms at five; full-step medians were 0.620 ms and
0.518 ms. The machine was noticeably slower than earlier sessions, so these
timings are not directly comparable to earlier runs. Final floor penetration
was 0.0057 m at six iterations and 0.0061 m at five. In separate sliding
scene runs, the final horizontal position changed from 3.8816 m to 4.3144 m.
**Not adopted:** this changes friction behavior and penetration, and no
agreed quality threshold or reference validates the tradeoff.

### Sliding-friction branch frequency

Counted force evaluations with at least one sliding contact lane in a
temporary instrumented 210-body pyramid run. Successive 100,000-evaluation
groups reported roughly 1,800–16,100 groups requiring sliding-friction
scaling. Counters aggregate across trials and perturb timing. **No
optimization attempted:** replacing the conditional norm and division with
unconditional work would add that cost to most measured evaluations.

### SSE3 horizontal reductions

Symbolized sampled instruction addresses inside the previously recorded
body-assembly closure. Some samples mapped to `sum4`, which reduces four SIMD
vectors; sample attribution includes inlining and does not by itself measure
the reduction's end-to-end cost. In a temporary copy, replaced the transpose
and additions with three SSE3 horizontal-add instructions and runtime CPU
detection, retaining the existing fallback. Five alternating invocations per
version, each with five trials, gave 210-body active medians of medians of
0.237 ms baseline (0.234–0.240 ms) and 0.251 ms candidate (0.246–0.255 ms).
For 820 bodies they were 1.319 ms (1.278–1.333 ms) and 1.391 ms
(1.364–1.403 ms). Ranges are across invocation medians.

A second temporary build enabled SSE3 globally and inlined the reduction to
test without per-call dispatch. In five alternating invocations, 210-body
active medians of medians were 0.239 ms baseline and 0.245 ms candidate;
820-body medians were 1.300 ms and 1.387 ms. Reported final heights matched.
**Not adopted:** both variants were slower, and globally requiring SSE3 would
also exclude CPUs without it. Horizontal additions change floating-point
association, so full behavior validation would be needed before adoption.

### Add or subtract the contact matrix by endpoint

In a temporary copy, replaced multiplication of the cross block and linear
gradient by a sign with an endpoint-dependent addition or subtraction. Five
alternating pyramid invocations per version, each containing five trials,
gave 210-body active medians of medians of 0.236 ms baseline
(0.233–0.246 ms) and 0.240 ms candidate (0.230–0.246 ms). For 820 bodies
they were 1.271 ms (1.256–1.296 ms) and 1.316 ms (1.298–1.326 ms). Ranges
are across invocation medians; final reported heights matched. **Not
adopted:** no reliable win, with a regression on the larger pyramid.

### Defer all contact reductions until body assembly completes

In a temporary copy, accumulated linear, cross, and gradient terms in SIMD
lanes across all of a body's contacts, then reduced them once per body. This
eliminated repeated per-manifold reductions at the cost of more vector
arithmetic and larger accumulators. Five alternating pyramid invocations per
version, each with five trials, gave 210-body active medians of medians of
0.240 ms baseline (0.229–0.246 ms) and 0.261 ms candidate (0.259–0.352 ms).
For 820 bodies they were 1.281 ms (1.250–1.389 ms) and 1.433 ms
(1.377–1.569 ms). Ranges are across invocation medians. Slide and heavy-stack
final snapshots matched the previously recorded values; the operation order
changed and was not exhaustively validated. **Not adopted:** it was slower.

### Pre-broadcast angular motion for contact evaluation

In a temporary copy, broadcast each angular-motion component to a SIMD vector
once before evaluating the three constraint rows, rather than letting each
vector-scalar multiplication perform its own broadcast. Five alternating
invocations per version, with five trials each, gave 210-body active medians
of medians of 0.233 ms baseline (0.226–0.242 ms) and 0.234 ms candidate
(0.229–0.242 ms). For 820 bodies they were 1.240 ms (1.228–1.327 ms) and
1.237 ms (1.231–1.252 ms). Ranges are across invocation medians; reported
final heights matched. **Not adopted:** the differences are within variation.

### Four-iteration alpha and damping sweep

In an isolated benchmark copy, tested all nine combinations of four AVBD
iterations, alpha 0.90/0.95/0.98, and contact damping 0.0003/0.0007/0.0015.
Each configuration ran one invocation of the pyramid benchmark (five trials
per engine); the grid was not alternated. On the 820-body pyramid, two of
these invocations produced slightly lower active medians than Box3D, but their
per-trial ranges overlapped. At alpha 0.95 and damping 0.0015, Verity measured
0.963 ms active versus Box3D's 0.988 ms and final floor penetration was
0.0069 m versus the six-iteration default's 0.0057 m from an earlier run.

Separate five-trial slide and heavy-stack invocations at damping 0.0015 and
all three alpha values gave final sliding positions between 4.9232 and
4.9458 m; the six-iteration default stopped at 3.8816 m. Heavy-stack final
cube overlap ranged from 0.0081 to 0.0141 m versus the default's 0.0074 m.
**Not adopted:** the settings change friction and stack behavior, and the
small apparent pyramid timing wins were not established beyond noise.

### Separate normal and tangent penalty-growth rates

In an isolated copy, used four iterations, alpha 0.90, normal beta 400,000,
damping 0.0015, and tangent beta ratios 0.05, 0.125, 0.20, 0.30, 0.50, and
1.0. One five-trial invocation per setting covered pyramid, slide, and heavy
scenes. Ratio 0.125 produced an 820-body active median of 0.883 ms versus
Box3D's 0.927 ms, while the sliding cube stopped at 3.4122 m versus the
six-iteration baseline's 3.8816 m. Ratios 0.14 through 0.17 were then tested
on slide and heavy scenes; ratio 0.16 stopped at 3.8413 m and the heavy stack
had 0.0057 m final cube overlap. These sweeps were not alternated.

At ratio 0.16, the initial candidate failed the existing static-friction and
dropped-pile regression tests. A second temporary candidate retained the full
tangent growth rate for contacts classified as sticking, and used the reduced
rate for other contacts. It passed all 32 existing physics integration tests.
Five alternating pyramid invocations per version, each with five trials,
gave candidate active medians of medians of 0.163 ms versus Box3D's 0.203 ms
for 210 bodies, and 0.963 ms versus 1.027 ms for 820 bodies. The larger-scene
invocation ranges overlapped and baseline and candidate builds differed.

One five-trial rain invocation exposed a major behavior difference. At 512
bodies the candidate's final maximum horizontal position was 40.1018 m and
RMS speed 1.3001 m/s, versus 16.9359 m and 0.9361 m/s in a separate
six-iteration baseline run. At 2,744 bodies these were 51.3799 m and
2.0208 m/s versus 27.5190 m and 1.0585 m/s. **Not adopted:** the existing
suite missed a large mixed-pile behavior regression, and the speed result is
not established at matched quality. The two rain invocations were not
alternated and their Box3D results varied as well.

### Motion-gated adaptive solver

An isolated prototype used the reduced-iteration settings only when all
bodies' linear and angular speeds were below fixed thresholds, and retained
the original settings for faster motion. An initial comparison accidentally
checked the wrong default angular beta, so the fast path never ran; those
numbers were discarded. After fixing the check and preserving explicitly
changed settings, the four-iteration version passed the 32 integration tests
and preserved the recorded mixed-pile final snapshots. Its 820-body active
timings were close to Box3D with overlapping ranges.

Three iterations for all low-speed bodies failed the static-friction test.
An exploratory variant used four iterations when any body had different
static and kinetic friction, otherwise three. It passed the 32 tests and an
additional equal-friction slope check. In seven alternating invocations, each
with five trials, the candidate's 820-body active median of medians was
0.802 ms (0.794–0.854 ms across invocation medians) and Box3D's was 0.860 ms
(0.848–0.898 ms). Mixed-pile final snapshots matched the prior baseline in
one separate invocation. **Not adopted:** the selection thresholds and exact
material comparison are workload-specific heuristics; they alter the meaning
of the public iteration setting and do not prove consistent accuracy across
timestep, force, or material changes. This is not a quality-matched win.

### Four iterations with different penalty-growth rates

In an isolated benchmark, used four iterations, alpha 0.90, damping 0.0015,
and tried beta values 50,000; 100,000; 200,000; 400,000; 1,000,000; and
5,000,000. Each setting ran one five-trial invocation for pyramid, slide, and
heavy scenes; settings were not alternated. Higher beta improved final stack
overlap, but the sliding cube traveled 3.5201, 4.9251, 7.1295, 10.2040,
15.4850, and 22.7629 m respectively. The six-iteration default traveled
3.8816 m. At beta 5,000,000 the cube was still moving at the final snapshot.
One 820-body run at beta 400,000 reported 0.895 ms versus Box3D's 0.901 ms,
but was not demonstrated beyond timing variation. **Not adopted:** high beta
substantially changes sliding behavior, and there was no repeatable win.

### Thin LTO with one code generation unit

Built an isolated copy using release `lto=thin` and `codegen-units=1`, and
alternated five pyramid benchmark invocations per build, each with five
trials. The 210-body active medians of medians were 0.238 ms baseline
(0.235–0.241 ms) and 0.233 ms with LTO (0.232–0.241 ms). For 820 bodies
they were 1.241 ms (1.236–1.247 ms) and 1.237 ms (1.226–1.265 ms); Box3D
medians in the same runs were 0.893 ms and 0.876 ms respectively. Ranges are
across invocation medians. Final reported heights matched. **Not adopted:**
the difference is within variation and does not close the active-scene gap.

### Profile-guided Rust compilation

Installed the Rust toolchain's matching `llvm-tools` component after the
available LLVM 22 profiler rejected Rust's LLVM 23 raw profiles. Trained an
instrumented isolated Rust build on the benchmark's pyramid, rain, sliding,
and heavy-stack scenes, then rebuilt it using the merged profile. Box3D's C
code was compiled normally; Rust code, including both bridges, used the
profile. Five alternating pyramid invocations per version, with five trials
each, gave 210-body active medians of medians of 0.239 ms baseline
(0.235–0.243 ms) and 0.228 ms candidate (0.225–0.231 ms). For 820 bodies
they were 1.303 ms (1.238–1.391 ms) and 1.280 ms (1.222–1.335 ms). Box3D's
corresponding medians in the candidate runs were 0.180 ms and 0.891 ms.
Ranges are across invocation medians; final reported heights matched. **Not
adopted:** it did not close the gap, the larger-scene ranges overlap, and
training and measurement used the same scenes. A general-purpose build would
need representative training, holdout scenes, and cross-platform validation.

### Pre-scale inertia once per step

In a temporary copy, moved multiplication of each dynamic body's inertia by
the timestep scale from every body assembly to initialization once per step.
This experiment mutated the internal solver body and would need a safer data
representation before adoption. Five alternating pyramid invocations per
version, each containing five trials, gave 210-body active medians of medians
of 0.237 ms baseline (0.226–0.244 ms) and 0.234 ms candidate
(0.231–0.258 ms). For 820 bodies they were 1.278 ms (1.208–1.390 ms) and
1.395 ms (1.202–1.453 ms). Ranges are across invocation medians; reported
final heights matched. **Not adopted:** the larger scene may have regressed,
and both timing ranges overlap.

## Paired stamping and a smaller manifold (adopted)

Collected on 2026-11-05 on the same CPU and toolchain, with one Rayon thread.
Comparisons alternated separate executables; figures are medians of invocation
medians over 5 to 11 rounds. Box3D's own median moved by a few percent between
sessions, and repeated 820-body ratios spanned roughly 0.95x to 1.05x, so treat
differences of that size as noise. The final solver file has SHA-256
`76d48427a389ee29cfdebe964cda1ebd7fd2d9f01b1970bb6046ca637c0c9bb8`.

Verity versus Box3D, milliseconds per step, 11 alternating rounds:

| Scene | Active before | Active after | Box3D | Ratio before | Ratio after |
| --- | --- | --- | --- | --- | --- |
| Pyramid, 210 | 0.220 | 0.160 | 0.177 | 0.78x | 1.11x |
| Pyramid, 820 | 1.161 | 0.836 | 0.825 | 0.70x | 0.99x |
| Rain, 512 | 0.319 | 0.304 | 0.481 | 1.50x | 1.58x |
| Rain, 2,744 | 2.231 | 2.082 | 3.518 | 1.55x | 1.69x |

| Scene | Full run before | Full run after | Box3D | Ratio before | Ratio after |
| --- | --- | --- | --- | --- | --- |
| Pyramid, 210 | 0.046 | 0.034 | 0.049 | 1.04x | 1.44x |
| Pyramid, 820 | 0.272 | 0.197 | 0.302 | 1.10x | 1.53x |
| Rain, 512 | 0.677 | 0.642 | 0.678 | 1.00x | 1.06x |
| Rain, 2,744 | 6.278 | 5.924 | 6.184 | 0.97x | 1.04x |

**The 820-body active phase reached parity, not a demonstrated win.** The
tiny slide and heavy scenes stayed ahead of Box3D but are too short to time.
A size sweep before the Jacobian change found Verity ahead up to about 600
pyramid bodies. Per-body cost rose from about 0.85 to 1.2 µs as the working set
outgrew the 1 MB L2 cache, while Box3D stayed near 1.0 µs. Larger pyramids
(1,176 and 1,596 bodies) measured 0.82x to 0.89x then and were not re-measured.

Two earlier conclusions in this log need qualifying. Timers showed the
iterations were about 74% of the active step, and an instruction-level sample
of the running solver found a flat profile with first-touch cache stalls and
store-forwarding failures from packed `Vec3` values through the stack, which
is why removing arithmetic alone rarely helped. Target features also only help
code compiled inside them: `solve_batch` alone left `dual` and other hot code
out of line, and closures passed to `from_fn` or `map` leave the feature
boundary and pass vectors through memory.

Adopted, in order, with the 820-body active median before and after each step:

- Stamp two of a body's multi-contact manifolds at once, one per 256-bit half
  (1.210 to 1.160 ms). Lanes repeat the scalar operations in order, and the
  scalar path pairs manifolds identically, so CPUs with and without AVX2 agree.
- Store `moved` and `turned` as `Vec3A`, and compile each batch inside one AVX2
  function with the hot path forced inline (1.207 to 1.112 ms).
- Fuse the dual update into the pair stamp (1.104 to 1.056 ms).
- Widen the linear block and coupling products to both halves (0.966 to 0.928 ms).
- Keep anchors in lanes between collisions, and reject new pairs whose tight
  boxes do not overlap before colliding them (0.935 to 0.900 ms).
- Widen `solve_eight`'s packing and the remaining pair products. No 820-body
  change (0.904 and 0.914 ms); 210 bodies improved from 0.172 to 0.167 ms.
- Store world anchors instead of angular Jacobians, using
  J θ = n · (θ × r) for the constraint and recomputing one body's Jacobian
  in the stamp (0.907 to 0.847 ms). This is the one step that changes rounding
  relative to earlier builds.
- Store stick and held as bits and move the pair key to the cold data, reducing
  the manifold from 640 to 368 bytes (0.850 to 0.842 ms, within noise, but it
  extends the range of scenes that fit in cache).
- Prefetch the first two lines of the next body's manifolds (0.855 to 0.829 ms).

Checks on the final code: final transforms hashed at six checkpoints in all
seven scenes (42 values) matched bit for bit between the AVX2 path and a build
forced onto the fallback, and between 1, 4 and 8 Rayon threads. A unit test
compares the paired stamp with the scalar path bit for bit over 4,096 random
states and fails for each of nine deliberately injected faults. All 32
integration tests pass.

Rounding changes alter chaotic pile trajectories, so the rain scenes were run
with eight jitter seeds before and after. Mean final floor penetration was
0.0403 and 0.0410 m at 512 bodies, and 0.0509 and 0.0563 m at 2,744, with
overlapping ranges (0.0347–0.0711 and 0.0410–0.1054 m at 2,744). RMS speed,
maximum |x| and top height were also within the seed spread. Eight seeds rule
out a large shift, not a small one. Box3D's mean penetration in the same runs
was 0.0081 and 0.0180 m, so these timings remain unmatched in quality.

## Rejected during that work

Each candidate was compared with the build it extended, on the 820-body active
median unless stated.

- **Prefetching every manifold line:** about 44 instructions per body; 0.991 ms
  with it and 0.981 ms without.
- **FTZ and DAZ:** tested whether denormals explained the arithmetic-resistant
  time. 1.108 to 1.071 ms, about 3%, so they do not. Changes results near zero.
- **FMA in the pair stamp:** 0.916 to 0.898 ms with 16 contractions. Too small
  to justify losing bit-identical results with the fallback.
- **Grouping first-visited manifolds in the adjacency:** intended to skip more
  dual updates. 0.900 and 0.904 ms.
- **Reversing alternate colors:** exact, and intended to shorten cache reuse
  distance. 0.914 and 0.896 ms before the Jacobian change, then 0.829 and
  0.859 ms with prefetching, so no reliable gain.
- **Moving `seen` into the hot manifold:** 0.928 and 0.930 ms; the miss moved
  to the refresh loop.
- **Aligning manifolds to 64 bytes:** at 384 bytes, 0.875 ms versus 0.842 ms
  unaligned at 368 bytes. Padding to 448 bytes measured 0.873 ms.
- **Storing inertia as `Mat3A`:** 0.856 and 0.862 ms.
- **Prefetching cold data during refresh:** 0.825 and 0.838 ms.

## Accuracy against closed-form references (adopted)

Collected on 2026-11-05 on the same CPU and toolchain, with one Rayon thread.
`examples/quality.rs` compares both engines with analytic results: Coulomb
stopping distances, sliding down a slope (gravity is tilted), a ball rolling
and going from sliding to rolling at 5/7 of its speed, stacks, impacts, and a
tumbling slab against RK4 integration of Euler's equations. Cubes and spheres
have no gyroscopic term, so the slab is needed.

A literature check found no published successor to AVBD for this use: other
2026 work targets GPU affine bodies, robotics-grade contact, or offline
simulation. The search did not cover a complete SIGGRAPH 2026 program. The
fixes below come from the authors' reference solver and from NVIDIA Newton's
later AVBD implementation, plus this log's measurements.

| Measure | Reference | Before | After | Box3D |
| --- | --- | --- | --- | --- |
| Slide stop, 2 m/s | 0.3398 m | +27.5% | -5.9% | -1.2% |
| Slide stop, 5 m/s | 2.1237 m | +82.8% | -2.5% | -0.5% |
| Slide stop, 10 m/s | 8.4947 m | +118.7% | -1.2% | -0.2% |
| Cube on 35 degree slope | 1.6105 m | +2.8% | +3.9% | +1.5% |
| Cube held at 30 degrees | 0 m | 0.0011 m | 0.0025 m | 0.0008 m |
| Ball slide to roll | 5 m/s | -11.9% | +0.1% | -0.1% |
| Ball rolls down 20 degrees | 4.7932 m | -0.2% | +0.9% | +0.2% |
| Stack of 20, top | 19.5 m | 19.4151 | 19.4312 | 19.3944 |
| 100x cube on 2, top | 2.5 m | 2.4820 | 2.4888 | 2.4098 |
| Drop 5 m, late penetration | 0 m | 0.0064 m | 0.0010 m | 0.0001 m |
| Drop 5 m, late speed | 0 m/s | 0.0836 | 0.0000 | 0.0000 |
| Tumble, momentum drift | 0 | not measured | 0.0000 | 0.0089 |
| Tumble, energy ratio | 1 | not measured | 1.0000 | 0.9830 |
| Tumble, final body spin | -3.8727 r/s | not measured | -0.1% | -0.8% |

The "before" drop penetration is a final snapshot; "after" is the mean of the
last 60 steps. The ball's -0.2% on the slope was partly the spin damping below
cancelling backward Euler's bias of n(n+1)/n², +0.8% at 120 steps. Over eight
jitter seeds, mean rain floor penetration fell from 0.0410 to 0.0159 m at 512
bodies and from 0.0563 to 0.0262 m at 2,744, with RMS speed essentially
unchanged. Box3D measured 0.0062 and 0.0140 m, so it remains better here. Peak
penetration on impact is also unchanged (0.1531 m versus Box3D's 0.0906 m): at
9.9 m/s a body moves 16 cm per step and contacts exist only once shapes overlap.

Single-thread timing against the previous commit, 11 alternating rounds, was
within noise: 820-body active 0.880 and 0.898 ms with Box3D at 0.869 and
0.889 ms (0.99x both); full run 0.208 and 0.201 ms; rain, 2,744 bodies, 2.266
and 2.310 ms active. The machine ran slower than in the previous section.

Adopted:

- **Secant tangential stiffness.** A cone-limited contact kept its full penalty,
  so each Newton step applied about k/(k + m/h²) of the capped force, and the
  error grew with speed. Tangential stiffness now takes the cone's scale factor.
  Two multiplies. An earlier comment cited the paper for the full penalty; the
  authors' demo uses it too, at 10 iterations.
- **Exact inverse rotation map.** `quat_add` is a Cayley map but velocities came
  from 2 sin(φ/2), multiplying spin by 1/√(1 + w²/4) each step: 0.35% per step
  at 10 rad/s. `quat_step` is the exact inverse. Joints keep the bounded
  `quat_sub`.
- **Post-stabilization, from the reference solver.** The main sweeps use α = 1,
  velocities are taken, and a last sweep at the `alpha` setting corrects
  step-start error without adding velocity. `iterations` still counts all sweeps,
  so the default is five plus one at the same cost. The default `alpha` is now 0.
- **Contact forces carry over whole,** as in the reference. Applying γ to λ left
  the 820-body pyramid unable to settle or sleep with five main sweeps: its top
  sank to 39.16 m and the full run cost 0.848 ms. Joints keep the decay; removing
  it failed `hard_joint_chain_holds_without_gaining_energy` (error 0.0060 m).
- **`COLLISION_MARGIN` of 1 mm instead of 3 mm.** Less sink at equal cost. At
  0.5 mm, 820-body active rose to 0.973 ms from about 0.87 ms.
- **Gyroscopic term.** Non-cubic bodies now conserve angular momentum. The next
  rotation is predicted by explicit midpoint, and the step's average velocity is
  converted back through the same mid-step inertia, which makes free flight exact
  by construction. Only non-cubic boxes pay for it.

Checks: five new integration tests each fail with the matching fix undone. Final
transforms stayed bit-identical between AVX2 and the fallback, and across 1, 4
and 8 Rayon threads.

## Rejected during that work

- **Gyroscopic update using the end-of-step inertia alone:** momentum exact, but
  tumble energy rose 8%.
- **The same with energy capped:** energy held, but momentum drifted 3.7% and the
  flip missed by 3.7%.
- **Pairing average velocities with mid-step inertias, without a prediction
  step:** energy rose 16%.
- **Midpoint prediction with `finish` rebuilding the inertia from the actual turn:**
  the two inertias differed, and energy rose 7.5%.
- **Larger `alpha` (0.3, 0.6):** no benefit to stacks, and heavy-stack overlap
  rose from 0.0105 to 0.0126 m at 0.6.

## Speculative contacts and momentum (adopted)

Collected on 2026-11-05 on the same CPU and toolchain, with one Rayon thread.
Compared with the previous section's build, and with Box3D, using
`examples/quality.rs`, which gained drops from 5, 20 and 40 m, free-body
collisions, and body-to-body overlap. Depths are metres. Box3D's single-run pile
figures vary by roughly 30% between runs.

| Measure | Before | After | Box3D |
| --- | --- | --- | --- |
| Cube dropped 5 m, peak depth | 0.1531 | 0.0000 | 0.0906 |
| Cube dropped 20 m, peak depth | not measured | 0.0001 | 0.0004 |
| Cube dropped 40 m, peak depth | not measured | 0.0004 | 0.0004 |
| Ball dropped 20 m, peak depth | not measured | 0.0011 | 0.0001 |
| Ball dropped 40 m, peak depth | not measured | 0.0011 | 0.0069 |
| Head-on at 20 m/s each, peak overlap | not measured | 0.0000 | 0.3333 |
| Pile, peak floor depth | 0.1679 | 0.0219 | 0.1395 |
| Pile, late floor depth | 0.0252 | 0.0012 | 0.0138 |
| Pile, peak body overlap | not measured | 0.0521 | 0.1068 |
| Pile, late body overlap | not measured | 0.0033 | 0.0455 |
| Rain, 512, floor (8-seed mean) | 0.0410 | 0.0010 | 0.0044 |
| Rain, 2,744, floor (8-seed mean) | 0.0563 | 0.0022 | 0.0201 |
| Stack of 20, top (19.5 m) | 19.4312 | 19.4670 | 19.3944 |

The rain "before" figures are from the commit before the accuracy work; the
section above reached 0.0159 and 0.0262 m. Box3D's depth at 5 m reflects where a
step happens to fall, not its usual result.

A cube moving 20 to 150 m/s into a 10 cm wall no longer passes through it in
any of seven phases; the previous build passed through in 3 to 6 of 7. A ball
passing a cube's corner with 5 cm to spare keeps its velocity.

Momentum was not conserved in collisions before this work. A cube striking a
resting cube at 5 m/s left 2.18 of 5 units of momentum; it now leaves 4.55.
Equal cubes meeting head-on at 5 m/s stop dead, and at 20 m/s retain 0.64 m/s of
total. Box3D is exact, because impulses are equal and opposite by construction.

Timing, median of alternating rounds, against the previous commit:

| Scene | Active before | Active after | Box3D | Full before | Full after | Box3D |
| --- | --- | --- | --- | --- | --- | --- |
| Pyramid, 210 | 0.163 | 0.158 | 0.191 | 0.035 | 0.034 | 0.053 |
| Pyramid, 820 | 0.846 | 0.781 | 0.837 | 0.200 | 0.165 | 0.308 |
| Rain, 512 | 0.303 | 0.378 | 0.474 | 0.641 | 0.785 | 0.661 |
| Rain, 2,744 | 1.628 | 2.138 | 2.700 | 4.596 | 5.637 | 4.633 |

**Rain's full run regressed from about 1.02x to 0.82x of Box3D's speed.** Its
active phase stays ahead. In one multi-threaded run the 820-body pyramid improved
(0.438 to 0.338 ms active) and 2,744-body rain slowed (0.862 to 1.039 ms).

Adopted:

- **Speculative contacts.** Contacts exist before shapes touch; the gap stays in
  the constraint in the main sweeps, and only overlap waits for stabilization.
  Bodies now rest at zero overlap: `COLLISION_MARGIN` became a detection range,
  `CONTACT_DISTANCE`, kept under the 1 cm gaps of neighbouring boxes.
- **Contacts built at the predicted first touch.** A contact plane from the start
  pose took 26% to 52% of the velocity of a ball missing a corner.
- **Penalty limit between free bodies,** at half the heavier body's m/h². Much
  stiffer, block descent locks the pair: whichever moves first satisfies the
  contact alone. Between 0.5 and 4 times that, 1 was the best balance: 0.5 raised
  pile body overlap to 0.187 m, and 2 lost more momentum (-15.8%).
- **Dual update at every visit,** so the second body feels the force the first
  used. Alone this added momentum (11.2 of 5) and made piles explode. With the
  limit it conserves momentum, and the 820-body pyramid's top rose to 39.44 m.
- **Rescaling only touching contacts.** A contact far from its surface alternated
  between overshooting and springing back, which let cubes through walls.
- **Edge axes must beat `0.95 * face + 0.01`, not `0.95 * edge > face + 0.01`.**
  With negative separations the old form, inherited from the demo, favoured edges
  past about 0.2 m of depth, giving flat cubes one contact point. This caused
  18 cm dives on hard landings and most of the head-on overlap.

Checks: final transforms stayed bit-identical between AVX2 and the fallback, and
across 1, 4 and 8 Rayon threads. Six deliberate reversions of these changes each
failed a regression test. Tolerances for ball depth (2 mm) and head-on momentum
(5% of speed) were set from these measurements.

## Open problems

- A box sliding over floor tiles stops too soon: 1.06 m instead of 2.12 m at
  5 m/s. This predates speculative contacts and is not understood. Preferring
  the nearest touching axis made it worse (0.98 m) and was rejected.
- Free collisions still lose 9% of a strike's momentum at five sweeps.
- Rain's cost. Narrowphase and manifold counts rose; much of this may be
  speculative contacts in rolling piles.

## Rejected during that work

- **Scoping swept boxes by relative motion alone:** the exact reject stayed in,
  but it did not explain rain's cost. Spheres padding contact range by their spin
  did; they no longer pad at all.
- **Testing overlap at the start pose as well as along the path:** one more SAT per
  moving pair, for pairs that are separating anyway.
- **FMA, reversal and the other ideas above** were not retried.

## Rain speed, tile seams and momentum (adopted)

Collected on 2026-11-05 with one Rayon thread. The machine ran about 40% slower
than earlier in the day for the larger scenes, so compare ratios between sessions,
not milliseconds. This section closes most of the open problems listed above.

Timing, alternating rounds, ratios to Box3D (higher is better):

| Scene | Active, last commit | Active, now | Full, last commit | Full, now |
| --- | --- | --- | --- | --- |
| Pyramid, 210 | 1.02x | 1.30x | 1.33x | 1.67x |
| Pyramid, 820 | 0.95x | 1.14x | 1.46x | 1.99x |
| Rain, 512 | 1.61x | 1.39x | 1.07x | 1.01x |
| Rain, 2,744 | 1.71x | 1.43x | 1.00x | 0.96x |

Rain's full run recovered from 0.82x to about parity. It is still slightly slower
than the last commit: its active phase pays for speculative contacts. In one
multi-threaded run, 820 pyramid bodies took 0.340 ms active against 0.460, and
2,744 rain bodies 1.448 against 1.275.

Adopted:

- **All manifolds stamp in pairs,** including single contacts, whose scalar branch
  is gone. A profile put 46% of rain in that batch code. With the next change this
  took 512-body rain from 0.799 to 0.667 ms in one A/B.
- **No separate overlap test when one sample covers the path:** the contact
  generator rejects the pair itself.
- **Velocities from contact forces in impacts.** Bodies in a manifold between free
  bodies closing above 1 m/s take velocity from the multipliers, which are equal
  and opposite before the solve converges. A strike keeps 4.93 of 5 units of
  momentum, up from 4.55, and 19.73 of 20. Using this everywhere gave exact
  momentum but made stacks unstable: a pyramid went non-finite and 100x-on-2 fell
  to 2.37 m, because motion-based velocity supplies the damping resting contacts
  need. A 0.25 m/s gate also broke the 100x stack; 0.5 m/s did not improve
  momentum. The remaining 1.4% was not traced.
- **Penalties start at a quarter of m/h²,** not PENALTY_MIN. Peak depth of a ball
  dropped 20 m fell from 1.1 to 0.6 mm, and a cube on a 30 degree slope no longer
  creeps (0.0000 m in 5 s; it was 1.6 mm, and Box3D 0.6 mm).

The tile problem listed above had already gone: it was the edge-axis bug. Tiled
slides are now within 5% of theory. Box3D, measured for the first time, catches on
seams: it stopped within 1 cm of its start at 2 and 5 m/s and lost 51% to 98% of the
distance at 10 and 20 m/s, varying between runs.

Over eight seeds, pile means are 0.0202 m peak floor depth (Box3D 0.1113), 0.0016 m
late floor depth (0.0124), 0.0447 m peak body overlap (0.1012), and 0.0048 m late
body overlap (0.0327).

Two remaining error sources are the integrator, not bugs. Slide stops fall short by
v·h/2, backward Euler's bias: -4.9% at 2 m/s and -1% at 10 m/s. Box3D's is the same
formula at its four substeps. A pendulum keeps its period to -0.11% and its rod to
1.3 mm, but its amplitude falls to 31% in 30 s, matching backward Euler's
exp(-ω²ht/2) = 0.29 (fixed below). No joint comparison exists: the Box3D bridge has
no joints.

Checks: 43 integration and 11 unit tests pass; each change above fails a test when
reverted; results are bit-identical between AVX2 and the fallback and across 1, 4
and 8 Rayon threads.

## Joint energy and AVX-512 (adopted)

Collected on 2026-11-05 on a Ryzen 7 8845HS (Zen 4) with one Rayon thread.

### Joints

The pendulum's lost swing was the integrator, but it is not inherent to AVBD. The
inertial target is already SHAKE's. The only difference is where the constraint
force points: backward Euler uses the step's end, SHAKE its start, and SHAKE
conserves energy. Hard joints now use the step-start arm in the torque.

| Measure | Before | After |
| --- | --- | --- |
| Pendulum swing after 30 s | 31.0% | 98.6% |
| Pendulum period error | -0.11% | -0.00% |
| Ten-bar chain, energy left after 30 s | 19% | 39% |

The rod's worst stretch stays 1.3 mm. A chain still loses 40% of its energy in the
first 5 s of whipping.

The start direction alone is unstable when solves are unconverged: a ten-bar chain
came apart by 8 m. Keeping the Hessian at the current pose made it worse (61 m). So
the force blends to the step-end direction by 1 - (turn / 0.07 rad)², where turn is
the faster body's rotation per step. A limit of 0.1 gave 99.1% but let harsh chains
gain 2% to 7%. From 0.15 chains stop settling, and from 0.25 they gain energy
without bound. Soft joints and springs keep backward Euler's damping.

This work also found a bug from the last round. The explicit gyroscopic step fed
long thin bodies energy without bound at large turns per step: 30 bars at 1/30 s
reached 800,000 times their energy, and a 60 m/s kick 18 million times, under
backward Euler too. Turning at constant momentum does no work, so both gyroscopic
steps now cap energy. The predictor does so only above 0.25 rad per step, because
below that a midpoint rightly differs by O(h), and capping it cost free tumbling
its exact momentum (drift 0.0087). A 1.25x slack instead of that gate compounded
each step and still exploded.

Limits: 30 bars at 1/30 s still gain up to 25%, and 3 bars kicked at 60 m/s reach
2.4 times their energy before settling. Neither grows without bound.

### SIMD

`fearless_simd` 0.4.1 was rejected: it stops at AVX2, with no AVX-512 level. The
kernels here are hand-laid shuffles that must match the fallback bit for bit, so its
portable types would replace nothing, and its dispatch is what the 20 lines of
`cpu_level` already do. A separate crate would have one user. glam supplies the
fallback's SSE2 or NEON.

AVX-512 is used for its registers, not its width. The same 256-bit kernels compiled
with AVX-512 VL have 32 registers and spill less. This needs no arithmetic change:

| Scene | AVX2 active | AVX-512 active | AVX2 full | AVX-512 full |
| --- | --- | --- | --- | --- |
| Pyramid, 210 | 0.142 | 0.140 | 0.030 | 0.030 |
| Pyramid, 820 | 0.710 | 0.687 | 0.149 | 0.145 |
| Rain, 512 | 0.343 | 0.338 | 0.702 | 0.681 |
| Rain, 2,744 | 1.998 | 1.893 | 5.194 | 5.131 |

Rejected: **512-bit stamping, four manifolds per register.** It was built in full
and stayed bit-identical, but took 0.716 ms on the 820 pyramid against 0.684 for the
recompile. Zen 4 splits 512-bit operations in two, so it saves only decoding, and
few bodies fill four slots. Its four-slot queue also slowed AVX2 by 6%. It may do
better on Intel cores with full 512-bit units; none was available to measure, so
400 lines of unmeasured unsafe code were not kept. A 16-body solve was not tried
for the same reason.

Checks: 46 integration and 11 unit tests pass; each joint and energy change fails a
test when reverted; results are bit-identical between AVX-512, AVX2 and the fallback
and across 1, 4 and 8 Rayon threads. The AVX-512 stamp is also tested bit for bit.
The crate checks for wasm32, but NEON has not been run.

## Breakable joints and rotation limits (adopted)

Collected on 2026-11-05. Speed is unchanged: all four benchmark scenes are within
1% of the previous build, as none has joints and the new pass is over links only.

Joints used to break only on angular torque, so a ball joint never could. Each
joint now has a `Strength` for its linear rows and one for its angular rows, which
follow a stress-strain curve:

- **Elastic** up to `yield_load`.
- **Plastic** beyond it. A row's force is capped, like friction, with the secant of
  that cap as its stiffness. Once per step the row's error becomes permanent: the
  anchor moves, the welded rest pose turns, or the rotation limit widens.
- **Hardening:** the cap rises from `yield_load` to `break_load` as strain reaches
  `ductility`, where the joint breaks. Without a yield load it is brittle.

Forces come straight from the multipliers, in newtons. A hanging 1000 kg cube reads
within 1% of 9810 N, and a hinged bar's stop reads 688.7 N·m against 689 by hand.
They are checked once per step on converged values and averaged over 30 ms: without
the average, the first steps of a load broke joints it could hold. An impact
therefore counts for less than its instantaneous peak.

Results match mechanics that were not put in by hand. A load applied at once
overshoots a hardening joint to twice its static stretch (0.08 m, static 0.037). A
bar reaching a yielding stop bends 1.0 rad past it, as energy balance predicts. A
chain of equal links breaks at the top, which carries the most.

Rotation limits are inequality rows per axis of B's rest frame, built like a
contact's normal: one multiplier per axis, its sign selecting the bound. Equal
bounds make a hinge. Stabilization matters here too: without it a stop threw a bar
back from 0.50 to 0.42 rad, and with it the bar stays at 0.49998.

Rubber is a soft joint whose stiffness grows as (1 + (stretch / stiffening)²), read
from the step's starting stretch. Soft joints keep backward Euler's damping.

Limits:

- A body has one Joint component, so a body joined to three others needs its
  partners to hold the other joints.
- Entities are referenced by name, and duplicates are refused at Play.
- Rotation limits are a box on the rotation vector, not a cone: past about 90
  degrees on two axes at once the box's corners are not an even range.
- Links in contact when a joint breaks collide at once, as they now may.
- Very heavy end loads still stretch hard joints, which affects when they break.
- Rotation stops are rigid and yielding uses the unaveraged force, so a body arriving
  fast loads a stop far above its static load: by hand, the demo elbow sees about 3,600 N·m
  on arrival and 140 N·m at rest, and needs a 60 cm² section to survive. A compliant
  stop would fix this.
- Load is not shown in the viewport yet, only in the inspector during Play.

Checks: 54 integration and 14 unit tests pass. Each behaviour fails a test when
removed: hardening, breaking on load and on strain, the anchor slide, limit
widening, stiffening, the limit force, its stabilization and the load average.

## Muscles, soft bodies, tearing and plastic metal (adopted)

Collected on 2026-11-05 on a Ryzen 7 8845HS with one thread. Rigid scenes are
unchanged: pyramids stay within noise of the previous build, as `step_soft` returns
at once with no soft bodies.

### Muscles and compliant stops

A joint may have a Hill muscle: section × specific tension × moment arm ×
activation × length factor × velocity factor. It is a capped drive toward a target
angle, and giving way harms it no more than letting go. What it cannot hold falls on
the limit rows, because the solver splits the load between them: no code subtracts
one from the other. A tensed arm holds what breaks a limp one, and relaxing a tensed
arm under load breaks it too. Stops can be compliant, which spreads an impact over
time and cuts the peak torque by at least 30% in the test.

The muscle's default numbers are round values, not checked against published data.

### Soft bodies

One `SoftBody` type, two solvers, chosen after beams failed for solids.

**Cloth** is a triangle mesh whose edges are beams, solved by vertex block descent
with alternating sweeps. Direct stiffness inferred large false forces from
unconverged stretch, and a hanging cloth tore itself, so each beam carries its force
as a multiplier and is penalized at its compression stiffness. Rejected penalties:
below that, cloth rang for ever (0.28 m/s average at rest; 0.002 at the value kept).
Stepping on the true stiffness everywhere converged too slowly and tore curtains.
Particles with fewer than five beams do use it: nothing balances their forces, and
the penalty alone overshot by its ratio to the stiffness and flung fragments at
tens of kilometres per second.

**Tearing:** an edge past its strength is cut, by its original particles, so it stays
cut through later splits. Around each particle, triangles still joined by uncut edges
form a fan, and each further fan gets a copy of the particle with its share of the
mass. A crack opens only when cuts separate a fan, so tears grow from their tips.
Triangles torn on two edges drop out, or slivers hang from the tear.

**Solids** are lattices solved by shape matching. Beams could not do this: a plate
over two supports was still 13 cm down after a second, against about 2 mm of true
elastic sag, because Gauss-Seidel moves bending slowly at any penalty. Overlapping
regions, each pulled to the best-fit rotation of its rest shape, stay stiff in
bending and are stable at any stiffness.

Plasticity updates the rest shape, as yielding joints do. When to measure strain took
four attempts. After a full step it is erased by shape sweeps, so nothing dented.
During late sweeps, a step's gravity across a thin plate read as yielding, and a
plate crept 2.8 cm under its own weight. Measuring after the first contact push,
with an allowance of `2gh²/thickness`, gives a plate that holds its shape, springs
back from a light blow and keeps a dent from a hard one.

**Rigid bodies** feel soft bodies a step late. Cloth passes on the momentum it did not
keep: the body's change of momentum, less gravity and pins, shared by contact.
Estimating force from push-out distance was unstable. Impulses are along contact
normals, or noise from pins throws balls sideways, and are capped at what cancels the
closing speed, so a contact never launches a body. Solids meet a body as two masses
colliding, the body and about one region, or a light ball sinks into a heavy plate
that barely moves.

A soft body ignores bodies it is pinned to: a shell collided with its own core, and
exploded. Damping applies to motion within a solid, or it acts as air resistance.

Time per step, one thread (`examples/soft_cost.rs`):

| Body | Particles | ms |
| --- | --- | --- |
| Cloth 20 × 20 | 441 | 0.40 |
| Cloth 40 × 40 | 1,681 | 1.84 |
| Cloth 64 × 64 | 4,225 | 5.04 |
| Plate 12 × 12 × 1 | 338 | 0.27 |
| Car shell 8 × 3 × 2 | 108 | 0.09 |
| Car shell 16 × 6 × 4 | 595 | 0.42 |

Limits:

- Soft bodies step serially, on one thread. Colouring particles would parallelize
  them, as for rigid bodies.
- No self-collision, and soft bodies do not collide with each other.
- Particles collide, not triangles: a small fast object can pass between them, and
  fast cloth can pass through thin walls.
- Solids do not tear or break; only cloth does. Their surface is a box.
- Rigid bodies feel soft bodies a step late, which suits bodies much heavier than the
  particles they touch, and has not been tested with light ones in stacks.
- A steel solid's strength is far below real steel's, and `crumple_zone` is tuned by
  eye: dent sizes are plausible, not predictions.

### Renderer

`DeformingMesh` vertices go through a per-frame buffer into a reserved tail of the
shared vertex buffer, between barriers, so no shader changed. They draw double-sided,
cast shadows and are not culled. They are not in Lux's GI scene.

Checks: `soft_bodies` ran on the GPU (Radeon 780M), which found a crash in the draw
loop the tests could not. Removing each behaviour above fails a test, except where
noted in this section's history: a reset of multipliers at tears proved unnecessary
once fragments used true stiffness, and a limit on the force through pins made no
difference, so both were deleted.

## Cloth elements, global solve, fracture and collision (adopted)

Collected on 2026-11-05 on a Ryzen 7 8845HS. This replaces the section above's cloth
solver, and closes most of its limits.

### Beams were the wrong element

`examples/cloth_accuracy.rs` hangs a 1 m sheet under its own weight, which stretches
by exactly ρgL²/(2Et). At the same 20 kN/m stiffness:

| Cells | Beams | Triangle elements |
| --- | --- | --- |
| 8 | -7.9% | +1.6% |
| 16 | +5.6% | +2.2% |
| 32 | +100% | +0.4% |
| 64 | came apart | +1.7% |

Edge beams have no shear or Poisson behaviour of their own, and depend on the mesh.
Cloth is now StVK membrane triangles with Bergou's quadratic bending, whose Hessian
is constant. A triangle keeps its rest data through mesh splits, so tearing no
longer rebuilds elements. Tears follow the largest principal stress, cutting the
edge most nearly across it, preferring edges that touch a cut: tears grow from their
tips. Only one element in a hundred may tear per step, worst first, or a uniform
load shreds a sheet in one step instead of relieving itself.

### Sweeps could not solve it

Element forces alone did not help: vertex block descent's error grew with resolution
(+10%, +141%, +645% at 8, 16, 32 cells), as a stiff sheet's large motions barely move
in a few sweeps. Chebyshev acceleration and adaptive starts changed the numbers
without fixing them. A sheet now takes an implicit Newton step solved by conjugate
gradients, matrix-free from the elements, in two passes that run in parallel and sum
in one order.

One Newton step needs a good start. Starting from last step's correction brought
solve work down, but it must be limited: whole, it fed itself; along its own
direction, it destabilized hanging; only the part opposing gravity, and only for
particles not on a surface, both settles a hanging sheet and lets one slide.

Mistakes worth keeping:

- Softening stiffness when a triangle shortens on average made steps 20 times too
  large: only a triangle with no tension at all is soft.
- Damping on a particle's own displacement dragged the whole sheet. It now acts on
  the stiffness applied to true strain rates.
- Contact springs at 10 times the sheet's stiffness left local strain the solver
  could not spread; the same stiffness allows millimetres of overlap.
- Friction after the solve stuck particles to a ball and tore the sheet around it. It
  is now a spring along the surface inside the solve, slipping at Coulomb's limit.
- A cap on reaction by closing speed read zero for cloth, which moves with what
  presses it. Cloth's cap is the body's own speed.
- Pins as firm as the sheet rang against rigid bodies, which feel them a step late,
  so their stiffness is bounded by the body's mass.

Cost is about four times beams' for far better results, on one thread and eight:

| Body | Particles | 1 thread | 8 threads |
| --- | --- | --- | --- |
| Cloth 20 × 20 | 441 | 1.4 ms | |
| Cloth 40 × 40 | 1,681 | 8.3 ms | 6.7 ms |
| Cloth 64 × 64 | 4,225 | 26 ms | 14 ms |

Warm iterations are capped at `20 + √particles / 2`, which at the 50 kN/m preset
leaves meshes above 16 cells about 4 mm long in the test: a real 0.4% strain error,
twice the expected stretch, that more iterations remove. Its cause was not fully
traced. Cloth this large is not a cheap per-frame effect.

### Solids

`SoftBody::voxels` builds any shape from an inside test. `embed` and `deformed` tie
detailed mesh vertices to the lattice. Fracture needs no mesh surgery: particles
are coupled only through regions, so a crack across a region's largest strain splits
the regions it crosses, within a region's reach, and grows as new tips overload.
Brittle materials crack at `break_load / stiffness`; ductile ones once plastic strain
reaches `ductility`. Crack faces are not capped: the surface just opens.

### Crushing

Calibration against crash data showed crush did not depend on strength at all: 1.2 m
at any yield, the distance to the rigid core reaching the wall. Position corrections
on light particles cannot stop a heavy core, and limiting the force through pins did
nothing. A carried body now feels crush strength × contact area, which is how a
crushing structure behaves. `crumple_zone` is set to crush a 1500 kg car about 0.55 m
at 56 km/h, within the 0.49 m NHTSA measured and the 0.65 m of a published barrier
test, and `a_car_at_56_kmh_crushes_about_as_far_as_a_real_one` holds it there.

### Collision

Pairs of particles about to meet become contact planes before the solve, for a sheet
and itself and between soft bodies, so the sheet carries the load: pushing them apart
afterwards tore sheets and stopped nothing. Fast particles sweep exactly against
obstacles. Bodies small enough to pass between particles press triangles along their
path.

### Elbow

`Muscle::elbow_flexors` uses areas of 3.6, 6.0 and 1.5 cm² and about 30 N/cm², from a
textbook-style source, not a primary study. It gives about 13 N·m, well under what
people lift, so it needs scaling. Murray et al. 2000 and Kawakami et al. 1994 have
the primary figures, which were not retrieved.

Limits that remain:

- Soft bodies are not in Lux's GI scene: it is a static structure built at load.
- A solid's crack faces are not capped, and its contacts are per particle.
- A rigid body feels a soft body a step late, so a 30 m/s ball passes through cloth,
  slowed, rather than stopping in it.
- A body has one Joint component.
- Solids run on one thread (they are cheap). Their landing hits yield more than
  real steel would: a steel ball dropped half a metre flattens by about 12%.
- Editor entities are cubes, so `embed` has no authoring path yet.

Checks: each behaviour here fails a test when removed, except two with no visible
effect: preferring the deepest contact, as a later one overwrites it anyway, and
keeping carried cloth awake, as it wakes next step.

## Many cars: riding, impacts and interest (adopted)

Collected on 2026-11-05 on a Ryzen 7 8845HS. A game needs many cars, so the question
stopped being how to make a car's lattice better and became how rarely to run it.
Thin shells per panel were rejected: 40 x 40 of sheet costs about 7 ms a step.

### Riding

A carried solid never slept, so every driving car paid for a full solve. Now a settled
solid with one carrier and no contacts rides: its particles follow the carrier and
nothing solves until something could reach it. `examples/soft_cost.rs`, 100 cars of
240 particles on rigid cores, driving:

| Threads | All solving | All riding |
| --- | --- | --- |
| 1 | 29.8 ms | 0.36 ms |
| 4 | 18.5 ms | 0.19 ms |
| 8 | 16.1 ms | 0.18 ms |

The same 100 rigid bodies alone cost 0.04 ms. Solving scales poorly, about 2x on eight
threads, not traced. It matters less now that only crashing cars solve.

Bodies now step in parallel, in order-independent work joined in body order, so a run
gives the same positions on any number of threads.

Mistakes worth keeping:

- A rider dropped out of its carrier's sums. Its weight and what the carrier's own
  acceleration took must still be applied, or a car speeds up when its shell settles.
- Regions stop tracking rotation while riding. Woken after half a turn, the warm-started
  rotation solve left the shape 0.10 m out, so regions catch up by the carrier's turn.
  At 108 degrees the solve recovers alone: only a half turn shows the fault.
- Unseen cars first had no crush force, so the core ran on and the shell ended up
  through the wall. `UNSEEN_CRUSH` stands in for the pins' share of stopping a car. It
  is 4 from one wall crash, and is not derived.

### Impacts and interest

A collision is recorded as one `Impact`: point, normal, energy and radius, in the
carrier's frame, from the worst of its steps. Under 100 J leaves no record. They are
what to send over a network, as a few bytes per crash; there is no network layer yet.

`set_interest` limits solving to spheres and frustums. Rigid bodies always step. Outside,
a carried body rides on and an uncarried one freezes, and impacts wait. On return they
press into the rest shape by `energy / (yield_load x area)`. Empty, the default, solves
everything, which suits a server: riding keeps it cheap, and a car's stopping force
comes from its solve.

A wall crash at 56 km/h replays to 0.63 m at the nose, inside the 0.49 to 0.65 m of the
crash data. A solved crash leaves only 0.21 m at the same point once it has sprung back,
though it peaks within that range, so seen and unseen cars do not end up alike.

Limits:

- Riding needs no contact at all, so a car whose shell rests on the ground keeps solving,
  as does one pressed against what it crashed into.
- Only a rigid core counts while unseen: one car's shell passes through another's until
  their cores meet.
- Spin is left out of a rider's momentum, and a body spinning fast may never settle.
- Only solids ride on their own. Cloth carried by one body rides only while unseen, and
  uncarried soft bodies pop to life when seen again.

### Solving far from the origin

Solved in world space, rounding acts as a steady force, because the same particle rounds
the same way each step. 64 cars driving in lanes, with no contact at all:

| Lanes | Riding after 5 s | Worst sideways speed |
| --- | --- | --- |
| Within 190 m of the origin | 64 of 64 | 0.05 m/s |
| Out to 378 m | 61 of 64 | 1.2 m/s, and they hit each other |
| From 2,000 m | 15 of 64 | 5.4 m/s |

So each body now solves around its own anchor, and publishes world positions at the end
of a step. Obstacles are shifted to it, and pins are taken from anchored poses, as a
world point is already rounded. A body is also built around its anchor: built in world
space, its rest shape and pins kept the rounding of wherever it was made, which alone
gave a car 0.12 m/s at 400 m. Cars now run straight at 20 km, and crush and cloth sag
match the origin's at 5 km and 3 km.

Riding's cost in the first measurements of this work, 1.27 ms, turned out to be this:
it was the cars that could never settle, not thread wake-ups as first supposed.

Rigid bodies still live in world space, and the anchor only protects the solve.

## Deforming cars on the GPU (adopted)

Collected on 2026-11-05 on a Ryzen 7 8845HS with its integrated GPU.

Soft bodies drew by rebuilding world-space vertices on the CPU each frame, untextured and
unculled. `EmbeddedModel` uploads a mesh once, with UVs, tangents and a material, and
instances share it: each owns only its output vertices. A lattice is about 4 KB against
megabytes of vertices. `embedded.slang` skins a car only when its shape changed, so a riding
car is a transform and an ordinary culled draw.

Instances share one copy of their indices: the renderer rewrites `frame_indices` every
frame, and a copy per car would cost 100 cars of this mesh 33 MB a frame.

Normals are exact, from the cofactors of each cell's deformation, which trilinear weights
already give: `(w0 + w1)` across an axis weighs that axis's edges. `embed` used to clamp
points to the lattice, flattening whatever stuck out, and now extrapolates.

A lattice cell is 35 to 40 cm, so dents alone look soft. `Dent`s add creased detail in the
shader from a short list, and need no per-vertex storage. Only a share of the crush goes
there, as the lattice already has the rest.

`examples/embedded_cars.rs` crashes real cars and fails unless the GPU's vertices match
`SoftBody::deformed`:

| Check | Result |
| --- | --- |
| GPU against CPU, over a 0.21 m crush | 0.012 mm |
| Dent detail | 0.08 m from 2 dents |
| Dispatches a frame, 4 cars crashing, 64 or 100 cars | 3.9 |
| Vulkan validation errors | 0 |

Frame time read 16.8 ms at both 64 and 100 cars, which is the display's refresh, so it
says nothing of the cost. This has no GPU timing.

Limits:

- Nothing loads a car: a model is built by hand, and a loaded glTF's vertices and textures
  do not reach `EmbeddedModel`. Scenes and the editor do not use this yet.
- Instances cannot be removed, and adding them waits for the GPU.
- Dents are in the rest frame and do not follow later crushing of that spot.
- Embedded meshes are not in Lux's GI scene, like other soft bodies.
- A cracked solid's weights still span the crack. Cars do not crack.

## Tearing: threads, tips and cohesive bonds (adopted)

Collected on 2026-11-05 on a Ryzen 7 8845HS, with `examples/tear_quality.rs`.

Tears followed mesh edges, branched from anywhere along their length, and a uniform load
shredded a sheet. An element now fails on its weakest plane given its threads, and a tear
grows only from its tips, cutting through triangles in the direction the material wants.

### What made tears run

The first measurements cut about 2,100 of a sheet's 3,000 edges. In order, what mattered:

- While any tear grows no other starts, as a tear shields the sheet around it, which a
  step's averaged load cannot show. Alone this changed nothing: every cut vertex still
  counted as a tip.
- Tips are particles where exactly one cut ends, and each makes one cut per pass, or a tear
  branches from every point on it. This took 2,100 edges to about 340.
- A tear runs at `TEAR_SPEED`, not an edge per step, which made finer cloth tear slower.
- Next to a tip an element needs only `TIP_LOAD`: stress gathers faster than a step's average
  shows, and a tip that stalled for one step let a new tear start elsewhere.

### Threads

Strength has to climb sharply off a thread's line. With Hill's smooth criterion the weakest
plane under a pull stayed within about 10 degrees of square to that pull, whatever the
thread, because any plane cutting yarns looked nearly as weak as one along them.

Direction from X, pulled along Y from a central nick, warp at 30 degrees:

| Material | 24 cells | 40 cells |
| --- | --- | --- |
| Felt (`weave` 0) | 1.5 | 2.5 |
| Plain weave | 12.8 | 18.5 |
| Strong grain (`grain` 0.3) | 19.3 | 32.8 |

The right trend, not a clean result: these tears are not straight, at 0.3 to 0.7 where 1 is a
line, and a weave's should be 30. `failure` itself picks the thread exactly, which tests hold.
What is lost is at tips, where the stress is the mesh's, and at clamped edges, below.

### Cohesive bonds

A bond's zone is `stiffness x toughness / strength^2`: 3.3 cm for `cloth`. Weakening strength
alone spread it to 3.3 m in a test, and no stress gathered at the tip at all, so keep them in
proportion. Secant stiffness goes into the solve, as true softening stiffness is negative.

Work per metre of tear was 30 J/m at 16 cells and 38 J/m at 32, so it holds across
resolution. That is about a quarter of `toughness`: length counts every cut, and an edge with
no particle split yet has no bond.

Speed through a sheet could not show any of this: a 50 kg ball carries far more energy than a
tear costs, and speed varied more with the mesh than with toughness.

### Pins

Pins to immovable bodies were weak: stiffness was capped by the holder's mass, which is 0 for
immovable. A kinematic bar stretched a sheet 100% before it tore, and now 20%.

Limits:

- A slow pull in `pull_apart` leaves about 13 m of tear in a 2 m sheet, 6 times what it needs.
  Fixed edges tear too: rows beside a moving grip overload long before load spreads, as one
  Newton step carries a fast edge load slowly.
- A heavy ball was sometimes caught at 96 cells and not at 32, so behaviour still depends on
  the mesh where one object makes a whole tear.
- A bond stands for one cell of tear, whatever the real length of the edges around it.
- Particles gain UVs, and `DeformingMesh` can take them with `textured`, but nothing in
  scenes or the editor sets either yet.
- A tear's fraying is not drawn.

## Tetrahedra for flesh (adopted)

Collected on 2026-11-06 on a Ryzen 7 8845HS.

`SoftBody::finite_elements` turns a voxel solid into six tetrahedra per cell, so shapes, pins
and `embed` carry over. Elements are stable neo-Hookean, in the implicit solve cloth uses,
with no decomposition. Cars stay on the lattice: this is for what must tear.

| Check | Result | Theory |
| --- | --- | --- |
| Cantilever droop, 40 x 4 x 4 cells | 0.1506 m | 0.1472 m |
| Same, 20 x 2 x 2 | 0.0916 m | 0.1472 m |

The hanging block test holds its stretch within 10% of rho g L^2 / 2E. The coarse beam is the
usual locking of linear tetrahedra: this needs four cells across for anything that bends.

| Body | Particles | Time per step, 8 threads |
| --- | --- | --- |
| Flesh 6 x 6 x 6 | 343 | 0.40 ms |
| Flesh 12 x 12 x 12 | 2,197 | 2.6 ms |

Mistakes worth keeping:

- Leaving out the change of the cofactors keeps the Hessian positive, and halves it in shear.
  Every Newton step then overshoots twofold, and a stiff block ran to infinity. It is all in
  now, and an indefinite result stops conjugate gradients early.
- The energy's constant is not Lame's: at rest it acts as itself less the shear. Given the plain
  value, Poisson 0 had negative volume stiffness.
- A 20 m/s hammer crushed elements until Newton's step ran to infinity. A step is limited to a
  cell (`TET_TRUST`), and if the first direction is already indefinite the solve still moves
  along it.
- Bonds took `break_load`, so one between fibres held three times what parted it.
- Comparing extreme points of a hanging block counted its pins' uneven sag as stretch: 11%.

Plasticity needs no decomposition either: Green strain's deviator moves each element's
gradients, so nothing else knows it happened. A block held at 30% keeps most of it. A free
block snaps back in about two frames, too fast to flow.

A face only one element holds is surface, so torn faces close themselves. Fracture tests each
element's four faces directly: their normals are its gradients, and they are the only planes
it can part on, so no direction is sampled.

Limits:

- Cracks follow element faces, so they are jagged at a cell's scale, and there is no tip
  splitting as in cloth. Flesh's torn edges ran 0.45 along a pull along its fibres, against
  0.31 for the same material without them: the right way, not a clean split.
- Pulled along its fibres, flesh does not last the three times longer its strength suggests:
  clamped ends give it sideways tension, and it splits lengthwise at about the same strain.
- Fibres affect strength only, not stiffness.
- Particles copied by a crack share mass evenly, not by volume.
- `embed` weights span a crack, so a detailed mesh over one stretches across it.
- A tetrahedral body does not ride, nor take `Impact`s, nor `nick`.
- A particle at a crack is pulled at up to its bond's stiffness until `opened` updates at
  the end of a step, so a body teleported apart closes up again.
