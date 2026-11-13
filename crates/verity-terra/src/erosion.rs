//! Landscape evolution on any graph of cells: tectonic uplift against
//! stream-power river incision, with hillslopes held at their critical angle.
//! Mountains are what this leaves standing, so ridges, spurs and valleys come
//! out connected and drained instead of being drawn by noise.
//!
//! Incision is implicit in the receiver's height (Braun & Willett 2013), so
//! steps can be large. Flow is routed over depressions with an epsilon priority
//! flood (Barnes et al. 2014) and the lakes it finds are filled, as sediment
//! would. Every quantity is in metres, so a coarse grid and a fine one agree,
//! which is what lets [`crate::Landscape`] and planets erode coarse to fine.

use crate::Random;
use rayon::prelude::*;
use std::{cmp::Reverse, collections::BinaryHeap};

/// Erodibility times the time step: `f = K·√A / L` per step, with A in m² and L in m.
const K: f32 = 0.1;
/// Drainage area in m² where a stream starts cutting; above it is hillslope.
const CHANNEL_HEAD: f32 = 1.0e4;
/// Metres of hillslope that soil creep holds under the critical slope.
const HILLSLOPE: f32 = 80.0;
/// Steps between reroutes; the implicit update stays valid on a slightly stale graph.
const REROUTE: usize = 2;

/// Cells and who neighbours whom.
pub(crate) trait Graph: Sync {
    fn len(&self) -> usize;
    /// Calls `visit(j, metres)` for each neighbour `j` of `i`.
    fn neighbours(&self, i: usize, visit: impl FnMut(usize, f32));
    /// Cells at the world's edge, where water leaves at their own height.
    fn is_edge(&self, i: usize) -> bool;
    /// Ground area per cell in m².
    fn area(&self) -> f32;
}

/// Evolve `heights` for `steps` steps of `uplift` metres each against river
/// incision, with slopes held under `talus` (rise over run). Cells below `sea`
/// and edge cells are base level. `hard(i, height)` in [0, 1] is rock that
/// erodes slower and stands steeper. Returns each cell's drainage area in m².
#[allow(clippy::too_many_arguments)] // Each is a distinct physical input; a struct would only rename them.
pub(crate) fn evolve(
    graph: &impl Graph,
    heights: &mut [f32],
    uplift: &[f32],
    sea: f32,
    talus: f32,
    hard: &(impl Fn(usize, f32) -> f32 + Sync),
    steps: usize,
    rng: &mut Random,
) -> Vec<f32> {
    let n = graph.len();
    let mut route = Route::new(n);
    // Soil creep in m² per step: a hillslope as long as `HILLSLOPE` stays under the
    // soil's critical slope at full uplift, so hilltops round over instead of meeting in knife edges.
    let diffusion = uplift.iter().fold(0.0f32, |m, u| m.max(*u)) * HILLSLOPE / 0.75;
    // Per cell: `f` of the implicit incision below, and the creep rate. Hardness changes
    // slowly, so these are per reroute.
    let mut incision = vec![0.0f32; n];
    let mut creep = vec![0.0f32; n];
    let (mut scratch, mut next) = (vec![0.0f32; n], vec![0.0f32; n]);
    for step in 0..steps {
        if step % REROUTE == 0 {
            route.update(graph, heights, sea, rng);
            // Between reroutes a couple of sweeps keep up with uplift; the exact pass ends the run.
            for _ in 0..2 {
                relax_slopes(graph, heights, &mut scratch, talus, hard);
            }
            let (h, r) = (&*heights, &route);
            (incision.par_iter_mut())
                .zip(creep.par_iter_mut())
                .enumerate()
                .for_each(|(i, (f, c))| {
                    let hardness = hard(i, h[i]);
                    // Above the channel head water gathers into a stream that cuts down.
                    *f = if r.area[i] < CHANNEL_HEAD {
                        0.0
                    } else {
                        K * (1.0 - 0.6 * hardness) * r.area[i].sqrt() / r.length[i]
                    };
                    *c = diffusion * (1.0 - 0.8 * hardness);
                });
        }
        heights.par_iter_mut().enumerate().for_each(|(i, h)| {
            if !graph.is_edge(i) {
                *h += uplift[i];
            }
        });
        // Downstream first, so each receiver already has its new height.
        for &i in &route.order {
            let r = route.receiver[i] as usize;
            // Base levels drain to themselves; lakes and flats don't incise.
            if r != i && heights[i] > heights[r] {
                let f = incision[i];
                heights[i] = (heights[i] + f * heights[r]) / (1.0 + f);
            }
        }
        // Implicit hillslope diffusion, by two Jacobi sweeps from the incised heights.
        // Few sweeps only smooth short wavelengths, and those are what the coarser grids left.
        scratch.copy_from_slice(heights);
        for _ in 0..2 {
            let (old, current, route) = (&scratch, &*heights, &route);
            next.par_iter_mut().enumerate().for_each(|(i, out)| {
                if route.receiver[i] as usize == i {
                    *out = current[i]; // Base level.
                    return;
                }
                let (mut sum, mut weight) = (old[i], 1.0);
                graph.neighbours(i, |j, metres| {
                    let a = creep[i] / (metres * metres);
                    // Soil creeps to the shore, not down the sea floor.
                    sum += a * current[j].max(sea);
                    weight += a;
                });
                *out = sum / weight;
            });
            heights.copy_from_slice(&next);
        }
    }
    route.update(graph, heights, sea, rng);
    limit_slopes(graph, heights, talus, hard);
    route.area
}

/// The drainage network: who drains into whom, in what order, and how much.
struct Route {
    receiver: Vec<u32>,
    /// Metres to the receiver.
    length: Vec<f32>,
    /// Drainage area in m².
    area: Vec<f32>,
    /// Downstream to upstream.
    order: Vec<usize>,
    filled: Vec<f32>,
    done: Vec<bool>,
}

impl Route {
    fn new(n: usize) -> Self {
        Self {
            receiver: (0..n as u32).collect(),
            length: vec![1.0; n],
            area: vec![0.0; n],
            order: Vec::with_capacity(n),
            filled: vec![0.0; n],
            done: vec![false; n],
        }
    }

    /// Route every cell to the sea or an edge, filling the lakes on the way.
    fn update(&mut self, graph: &impl Graph, heights: &mut [f32], sea: f32, rng: &mut Random) {
        let base = |i: usize, h: f32| h < sea || graph.is_edge(i);
        let (filled, done, order) = (&mut self.filled, &mut self.done, &mut self.order);
        done.fill(false);
        order.clear();
        let mut heap = BinaryHeap::new();
        for (i, &h) in heights.iter().enumerate() {
            if base(i, h) {
                done[i] = true;
                filled[i] = h;
                heap.push(Reverse(flood_key(h, i)));
            }
        }
        // A closed world with no sea drains to its lowest point.
        if heap.is_empty() {
            let lowest = (0..heights.len())
                .min_by(|a, b| heights[*a].total_cmp(&heights[*b]))
                .expect("a graph has cells");
            done[lowest] = true;
            filled[lowest] = heights[lowest];
            heap.push(Reverse(flood_key(heights[lowest], lowest)));
        }
        while let Some(Reverse(key)) = heap.pop() {
            let i = key as u32 as usize;
            let level = filled[i];
            order.push(i);
            graph.neighbours(i, |j, _| {
                if !done[j] {
                    done[j] = true;
                    // A random epsilon keeps filled lakes sloping to their spill point without
                    // the parallel straight channels a constant one carves across flats.
                    filled[j] = heights[j].max(level + 1e-3 * (0.1 + rng.unit()));
                    heap.push(Reverse(flood_key(filled[j], j)));
                }
            });
        }
        // Lakes fill with sediment up to their spill point.
        heights.copy_from_slice(filled);
        // Steepest descent on the filled surface. A lower neighbour always exists (the cell
        // that flooded it) and was popped earlier, so `order` runs downstream to upstream.
        let heights = &*heights;
        (self.receiver.par_iter_mut())
            .zip(self.length.par_iter_mut())
            .enumerate()
            .for_each(|(i, (r, length))| {
                *r = i as u32;
                if base(i, heights[i]) {
                    return;
                }
                let mut best = 0.0;
                graph.neighbours(i, |j, metres| {
                    let slope = (heights[i] - heights[j]) / metres;
                    if slope > best {
                        best = slope;
                        *r = j as u32;
                        *length = metres;
                    }
                });
            });
        self.area.fill(graph.area());
        for &i in order.iter().rev() {
            let r = self.receiver[i] as usize;
            if r != i {
                self.area[r] += self.area[i];
            }
        }
    }
}

/// Lower every cell to at most `talus × (1 + 3·hardness)` rise over run above
/// each neighbour: landslides, which the rivers then carry away. Exact, like a
/// distance transform: cells are settled lowest first, starting from those
/// that hold a slope over the limit.
fn limit_slopes(
    graph: &impl Graph,
    heights: &mut [f32],
    talus: f32,
    hard: &(impl Fn(usize, f32) -> f32 + Sync),
) {
    let limit = |j: usize, h: f32| talus * (1.0 + 3.0 * hard(j, h));
    let seeds: Vec<usize> = (0..graph.len())
        .into_par_iter()
        .filter(|&i| {
            let mut steep = false;
            graph.neighbours(i, |j, metres| {
                steep |= heights[j] > heights[i] + limit(j, heights[j]) * metres;
            });
            steep
        })
        .collect();
    let mut heap: BinaryHeap<_> = (seeds.iter())
        .map(|&i| Reverse(flood_key(heights[i], i)))
        .collect();
    while let Some(Reverse(key)) = heap.pop() {
        let i = key as u32 as usize;
        if (key >> 32) as u32 != ordered_bits(heights[i]) {
            continue; // Lowered since it was queued.
        }
        let h = heights[i];
        graph.neighbours(i, |j, metres| {
            let top = h + limit(j, heights[j]) * metres;
            if heights[j] > top {
                heights[j] = top;
                heap.push(Reverse(flood_key(top, j)));
            }
        });
    }
}

/// One parallel sweep of [`limit_slopes`]: each cell drops to the limit over
/// its lowest neighbour as that was. Approximate, as a slope can only fall
/// one cell further per sweep, and identical on any thread count.
fn relax_slopes(
    graph: &impl Graph,
    heights: &mut [f32],
    scratch: &mut [f32],
    talus: f32,
    hard: &(impl Fn(usize, f32) -> f32 + Sync),
) {
    let old = &*heights;
    scratch.par_iter_mut().enumerate().for_each(|(j, out)| {
        let limit = talus * (1.0 + 3.0 * hard(j, old[j]));
        let mut h = old[j];
        graph.neighbours(j, |i, metres| h = h.min(old[i] + limit * metres));
        *out = h;
    });
    heights.copy_from_slice(scratch);
}

/// Float bits that sort like the floats.
fn ordered_bits(height: f32) -> u32 {
    let bits = height.to_bits();
    if bits >> 31 == 1 {
        !bits
    } else {
        bits | 1 << 31
    }
}

/// Heap key: orders by height, then index for determinism.
fn flood_key(height: f32, i: usize) -> u64 {
    (ordered_bits(height) as u64) << 32 | i as u64
}

/// A square heightmap of `side` × `side` samples `cell` metres apart.
pub(crate) struct Grid {
    pub side: usize,
    pub cell: f32,
}

impl Graph for Grid {
    fn len(&self) -> usize {
        self.side * self.side
    }

    fn neighbours(&self, i: usize, mut visit: impl FnMut(usize, f32)) {
        let diagonal = self.cell * std::f32::consts::SQRT_2;
        crate::for_neighbours(self.side, i, |j, d| {
            visit(j, if d { diagonal } else { self.cell })
        });
    }

    fn is_edge(&self, i: usize) -> bool {
        crate::is_border(self.side, i)
    }

    fn area(&self) -> f32 {
        self.cell * self.cell
    }
}

/// Bilinear upsample of a `coarse`-sided map to `2·(coarse − 1) + 1` per side.
pub(crate) fn upsample(map: &[f32], coarse: usize) -> Vec<f32> {
    let fine = 2 * (coarse - 1) + 1;
    (0..fine * fine)
        .into_par_iter()
        .map(|i| {
            let (x, y) = (i % fine, i / fine);
            let (x0, y0) = (x / 2, y / 2);
            let (x1, y1) = (x.div_ceil(2), y.div_ceil(2));
            let at = |x: usize, y: usize| map[y * coarse + x];
            (at(x0, y0) + at(x1, y0) + at(x0, y1) + at(x1, y1)) * 0.25
        })
        .collect()
}
