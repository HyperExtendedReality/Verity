# CPU physics experiments

Keep this log when trying performance changes. Record the workload, build,
hardware, result, behavior checks, and reason for keeping or rejecting each
idea. A result applies only to the recorded conditions; retry when the solver,
compiler, hardware, or workload changes enough to invalidate it.

## Comparison method

Run `cargo run --release -p hyperon-physics --example compare` with
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
The Box3D and Hyperon configurations use different solvers, so their timings
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

Set `HYPERON_ITERATIONS=4` without changing the default. One invocation per
scene family ran five trials per engine. In the 820-body pyramid, active
median was 0.985 ms for Hyperon versus 0.928 ms for Box3D, while the full-run
median was 0.608 ms versus 0.333 ms. The prior six-iteration run measured
1.359 ms active and 0.319 ms full-run for Hyperon, but was not paired with
this experiment. For 2,744 rain bodies, four iterations measured 2.233 ms
active and 6.243 ms full-run versus Box3D's 4.483 ms and 8.394 ms. Hyperon's
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
final maximum floor penetration was 0.0057 m in Hyperon and 0.0023 m in Box3D;
both had zero reported final RMS linear speed. A separate four-iteration run
reported 0.0066 m and 0.0024 m respectively. These are final snapshots, not
maximum penetration over the whole simulation.

The sliding cube stopped at x=3.8816 m in Hyperon and x=2.1131 m in Box3D;
both reported zero final linear speed. The heavy stack's final top heights
were 5.46 m and 5.24 m. In a subsequent invocation, maximum final overlap
between cubes was 0.0074 m and 0.0590 m, respectively. These results identify
behavior differences, not which engine is closer to a reference solution
overall. Timing for the tiny scenes was only
microseconds and should not be treated as a dependable speed comparison.

On 2026-11-05, a five-trial invocation for each `BOX3D_SUBSTEPS` setting of
4, 8, 16, and 32 reported final maximum cube overlap of 0.0596, 0.0589,
0.0585, and 0.0586 m in Box3D. Hyperon, still at six AVBD iterations, reported
0.0074 m in each invocation. Increasing substeps alone did not match this
particular final-state measure. The runs do not sweep other solver settings or
compare the two engines against a reference trajectory.

### Frictionless sliding reference

Added `free_slide`, a cube moving at 5 m/s on a horizontal floor with zero
friction. After 300 steps of 1/60 s its reference horizontal displacement is
25 m. A five-trial invocation reported 25.0001 m in Hyperon and 25.0000 m in
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
The largest named Hyperon entry was `solve_batch::closure$1`, the body assembly
closure. `Manifold::dual` and `Manifold::prepare` also appeared. This trace
includes both engines, setup, active steps, and sleeping steps; inlining can
attribute work to the enclosing function. It does not establish a per-stage
percentage or prove which operation inside the closure dominates. Trace and
symbol output are local build artifacts under `target/physics-profile`.

A follow-up profile forced contact constraint evaluation, force evaluation,
stamping, and dual update out of line in a temporary build. Stamping and
force evaluation had the largest named Hyperon sample weights, followed by
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
per-trial ranges overlapped. At alpha 0.95 and damping 0.0015, Hyperon measured
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

Hyperon versus Box3D, milliseconds per step, 11 alternating rounds:

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
A size sweep before the Jacobian change found Hyperon ahead up to about 600
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
exp(-ω²ht/2) = 0.29. No joint comparison exists: the Box3D bridge has no joints.

Checks: 43 integration and 11 unit tests pass; each change above fails a test when
reverted; results are bit-identical between AVX2 and the fallback and across 1, 4
and 8 Rayon threads.
