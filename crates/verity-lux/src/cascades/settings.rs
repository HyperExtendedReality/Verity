//! Quality controls and the buffer sizing and work budgets derived from them,
//! ported from the Bevy implementation (`legacy/src/render.rs`). Sizes must
//! match what the shaders index, so these stay verbatim.

use bytemuck::{Pod, Zeroable};

/// Cascade levels; mirrors the shaders.
pub const LEVELS: usize = 4;
/// Bytes per interval cache entry.
pub const ENTRY_SIZE: u64 = 240;
/// Bytes per descriptor (`Descriptor` in common.slang).
pub const DESCRIPTOR_SIZE: u64 = 32;
/// Bytes per directional irradiance probe record.
pub const PROBE_SIZE: u64 = 1792;
pub const CACHE_WORK_GROUPS: u64 = 2;
pub const CACHE_HISTORY_PER_BUCKET: u64 = 4;
pub const SPARSE_HISTORY_RESET_FLAG: u32 = 1 << 31;
pub const SPARSE_UPDATES_DISABLED_FLAG: u32 = 1 << 30;
pub const TRACE_MAX_LANE_STRIDE: u32 = 8;
/// Words before the first merge list; `MERGE_LIST_HEADER` in cascades.slang.
pub const MERGE_LIST_HEADER: u64 = 16;
const COLD_RANK_PHASES: [usize; 3] = [1, 2, 3];
const PACK_PROBE_GROUPS_PER_ROW: u64 = 256;
const HISTORY_DIAGNOSTIC_WORDS: u64 = 26;
// Mirror DEFERRED_* in common.slang: deferred primary trace scratch.
pub const DEFERRED_TRACE_RECEIVERS: u64 = 8192;
pub const DEFERRED_VISIBILITY_QUERIES: u64 = 4;
const DEFERRED_SLOT_WORDS: u64 = 32;
const DEFERRED_QUERY_WORDS: u64 = 12;
/// The receiver-grid safety cap when `max_receiver_cells` is zero.
const MAX_RECEIVERS: u32 = 131_072;

/// Split Radiance Cascades quality and lighting controls. Distances are in
/// metres.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Settings {
    pub sampling: Sampling,
    /// Cascade-zero probe spacing.
    pub probe_spacing: f32,
    /// Cascade-zero interval length.
    pub interval_length: f32,
    /// Receiver-grid pixel divisor.
    pub resolution_divisor: u32,
    /// Receiver updates per frame, at most.
    pub max_rays: u32,
    pub normal_bias: f32,
    /// Composite gain on indirect diffuse light.
    pub intensity: f32,
    /// Keep the renderer's sky reflection, which Lux does not trace. It ignores
    /// occlusion, so leave it off indoors.
    pub sky_reflections: bool,
    pub temporal_blend: f32,
    pub split_transmittance_blend: f32,
    pub history_frames: u32,
    /// À-trous passes over the shaded image. The probes are already smooth:
    /// on Sponza, three passes instead of one changed neither error nor flicker
    /// once warm, and cut error by at most 3% in GI's first second, for 0.22 ms.
    pub spatial_filter_steps: u32,
    pub emissive_samples: u32,
    /// Carry light past the first bounce, through last frame's probes.
    pub multi_bounce: bool,
    /// Cache outgoing light on the scene's surfaces, in islands of nearly flat
    /// triangles packed into an atlas, and let ray hits read it: multi-bounce
    /// light then stays on the surface it belongs to instead of leaking through
    /// walls from world probes. Blended and double-sided surfaces stay out of
    /// the atlas; [`Self::surface_hash_cells`] holds their light instead.
    pub surface_cache: bool,
    /// The cache's texel size in metres. It grows when the atlas would pass
    /// two million texels.
    pub surface_texel: f32,
    /// Texels shaded per frame, each with direct light and bounce rays.
    pub surface_updates: u32,
    /// Bounce rays per texel update, at most 8.
    pub surface_bounce_rays: u32,
    /// With software tracing, bounce rays stop after this many metres, and
    /// those that reach no surface leave the texel's mean. A few rays across
    /// the whole scene otherwise set the update's time. Zero: no limit.
    /// Hardware rays always go on.
    pub surface_software_reach: f32,
    /// Hits in this cascade and beyond take their whole outgoing light from
    /// the cache instead of shading direct light; [`LEVELS`] keeps every hit
    /// shaded and uses the cache for further bounces only.
    pub surface_from_level: u32,
    /// Hashed cells for the surfaces the atlas leaves out (double-sided ones
    /// such as foliage, and retained meshes such as terrain), each side of a
    /// surface its own cell. Ray hits claim them; zero turns them off. Taken
    /// when a scene loads. Needs [`Self::surface_cache`].
    pub surface_hash_cells: u32,
    /// Hashed cells shaded per frame, as [`Self::surface_updates`] for texels.
    pub surface_hash_updates: u32,
    /// Trace with hardware ray queries where the GPU has them; off, rays walk
    /// the GI scene's own tree. Changing it recompiles the passes, and GI shows
    /// the hemisphere until they are ready, as when it starts.
    pub hardware_rays: bool,
    /// Path trace each shaded pixel's indirect light with the cache's own hit
    /// shading and no cache, probe or filter, averaging until the view or a
    /// light changes: a reference to measure the rest of these settings
    /// against. Use `resolve_scale` 1 for one per pixel. Hundreds of times the
    /// cost of the cache, and it takes hundreds of frames to settle.
    pub reference: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            sampling: Sampling::default(),
            probe_spacing: 0.25,
            interval_length: 0.4,
            resolution_divisor: 2,
            max_rays: 131_072,
            normal_bias: 0.005,
            intensity: 1.0,
            sky_reflections: false,
            temporal_blend: 0.88,
            split_transmittance_blend: 0.98,
            history_frames: 24,
            spatial_filter_steps: 1,
            emissive_samples: 16,
            multi_bounce: true,
            surface_cache: true,
            surface_texel: 0.3,
            surface_updates: 32_768,
            surface_bounce_rays: 2,
            surface_software_reach: 0.0,
            surface_from_level: 3,
            surface_hash_cells: 1 << 16,
            surface_hash_updates: 8_192,
            hardware_rays: true,
            reference: false,
        }
    }
}

impl Settings {
    /// For open ground hundreds of metres across.
    /// The defaults suit rooms. Outdoors, what they leave uncapped costs several
    /// times more, and with the budgets below their quarter-metre probes spread
    /// the samples too thin and turn noisy.
    pub fn landscape() -> Self {
        Self {
            probe_spacing: 4.0,
            interval_length: 6.0,
            sky_reflections: true,
            ..Self::budgeted()
        }
    }

    /// The benchmarked budget for a frame, with about 800 MB of world cache in
    /// place of the uncapped default. On Sponza a coarser grid and half-size
    /// resolve lost nothing, and what they save buys a cache that keeps probes:
    /// against a path-traced reference, a smaller one is what costs accuracy.
    /// Half-metre probes, with intervals scaled to match, keep the update budget
    /// from spreading thin: in Sponza's courtyard they took the error from 0.30
    /// to 0.26 and the frame from 9.9 ms to 4.9 ms (quarter-metre ones suit the
    /// uncapped default).
    pub fn budgeted() -> Self {
        Self {
            probe_spacing: 0.5,
            interval_length: 0.8,
            sampling: Sampling {
                grid_divisor: 4,
                resolve_scale: 2,
                max_cache_entries: 1_048_576,
                max_updates_per_frame: 2048,
                max_traversal_queries: 65_536,
                ..Sampling::default()
            },
            ..Self::default()
        }
    }
}

/// Work and memory caps. Zeros mean "derive from other settings" as noted.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Sampling {
    /// Candidate-grid pixel divisor; zero inherits `resolution_divisor`.
    pub grid_divisor: u32,
    /// Maximum screen-grid candidates; zero is the 131,072 safety cap.
    pub max_receiver_cells: u32,
    /// Receiver updates per frame; zero inherits `max_rays`.
    pub max_updates_per_frame: u32,
    /// Retained world-cache entries; zero derives capacity from the grid.
    pub max_cache_entries: u32,
    /// Traversal queries per frame across all passes; zero is uncapped.
    pub max_traversal_queries: u32,
    /// Shading resolution divisor, 1 to 8 (zero means 1).
    pub resolve_scale: u32,
}

/// The shaders' parameter block, in their std140 order (`Parameters` in
/// `shaders/common.slang`).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct Parameters {
    pub world_from_clip: [[f32; 4]; 4],
    pub clip_from_world: [[f32; 4]; 4],
    pub previous_clip_from_world: [[f32; 4]; 4],
    pub previous_world_from_clip: [[f32; 4]; 4],
    pub camera_position: [f32; 4],
    pub viewport: [u32; 4],
    pub size: [u32; 4],
    pub tuning: [f32; 4],
    pub environment: [f32; 4],
    /// The scene's ground radiance, under `environment`'s sky.
    pub ground: [f32; 4],
    pub history: [f32; 4],
    pub interval_history: [f32; 4],
    pub frame: [u32; 4],
    pub light_counts: [u32; 4],
    pub stability: [f32; 4],
    pub layers: [u32; 4],
    pub environment_info: [u32; 4],
    pub environment_rotation: [f32; 4],
    pub environment_scale: [f32; 4],
    pub geometry_pages: [u32; 4],
    pub scene_pages: [u32; 4],
    pub scene_overlay: [u32; 4],
    pub dirty_min: [f32; 4],
    pub dirty_max: [f32; 4],
    pub source_history: [u32; 4],
    pub sampling: [u32; 4],
    pub owner_sequence: [u32; 4],
    pub cache_time: [u32; 4],
    pub emissive_dirty_min: [f32; 4],
    pub emissive_dirty_max: [f32; 4],
    pub light_dirty_min: [f32; 4],
    pub light_dirty_max: [f32; 4],
    pub ray_traced: [u32; 4],
    /// Surface cache: texels (0 = off), texels updated per frame, the first
    /// cascade whose hits read the cache instead of shading, bounce rays per
    /// texel (bits 0-7) and software reach in centimetres (bits 8-31).
    pub surface: [u32; 4],
    /// Hashed surface cells: capacity (0 = off), cells updated per frame and
    /// frames a cell may go unread before it is dropped.
    pub surface_hash: [u32; 4],
    /// Elements in the cache entries, previous entries, head words and
    /// merge-list words: the shaders reach them through pointers, which have
    /// no length of their own.
    pub lengths: [u32; 4],
    /// Elements in the geometry and scene words.
    pub word_lengths: [u32; 4],
    /// `CASCADE_POINTERS` in cascades.slang, in order.
    pub buffers: [u64; 23],
}

pub fn probe_directory_capacity(samples: u64) -> u64 {
    samples.saturating_mul(4).max(1).next_power_of_two()
}

pub fn pack_probe_dispatch_groups(samples: u64) -> Option<(u32, u32)> {
    let descriptors = samples.checked_mul(2)?;
    if descriptors == 0 || descriptors > u64::from(u32::MAX) {
        return None;
    }
    let groups_x = descriptors.min(PACK_PROBE_GROUPS_PER_ROW);
    let groups_y =
        descriptors.checked_add(PACK_PROBE_GROUPS_PER_ROW - 1)? / PACK_PROBE_GROUPS_PER_ROW;
    Some((u32::try_from(groups_x).ok()?, u32::try_from(groups_y).ok()?))
}

/// Extra rank phases that prioritize dense probes while history is cold.
pub fn cold_rank_phases(history_frames: u32, history_was_valid: bool) -> &'static [usize] {
    if history_frames > 0 && !history_was_valid {
        &COLD_RANK_PHASES
    } else {
        &[]
    }
}

/// Words after the trace sample list: query count and dispatch arguments,
/// the query list, per-variant hit slots, and the queries themselves.
pub fn deferred_trace_words(samples: u64) -> u64 {
    let slots = samples.min(DEFERRED_TRACE_RECEIVERS) * 2;
    let queries = slots * DEFERRED_VISIBILITY_QUERIES;
    4 + queries + slots * DEFERRED_SLOT_WORDS + queries * DEFERRED_QUERY_WORDS
}

/// Cache heads and counters, per-cascade probe indices and ray counts, the
/// trace sample list and the deferred trace scratch.
pub fn cache_head_bytes(buckets: u64, samples: u64) -> u64 {
    let counters_and_diagnostics = 10 + HISTORY_DIAGNOSTIC_WORDS;
    let probe_directory = LEVELS as u64 * probe_directory_capacity(samples);
    (buckets * 2
        + counters_and_diagnostics
        + probe_directory * 2
        + 1
        + samples
        + 4
        + deferred_trace_words(samples))
        * 4
}

/// Single bounce only: the multi-bounce cache doubles this.
pub fn irradiance_probe_bytes(samples: u64) -> u64 {
    probe_directory_capacity(samples) * 2 * PROBE_SIZE
}

pub fn resolve_scale(settings: &Settings) -> u32 {
    settings.sampling.resolve_scale.clamp(1, 8)
}

/// Shading (resolve) grid for a viewport.
pub fn resolve_size(viewport: [u32; 2], settings: &Settings) -> [u32; 2] {
    let scale = resolve_scale(settings);
    viewport.map(|v| v.div_ceil(scale).max(1))
}

pub fn cache_work_slots(samples: u64) -> u64 {
    samples * LEVELS as u64 * CACHE_WORK_GROUPS
}

/// One descriptor per working slot that seed fills: sample, variant, level.
pub fn descriptor_bytes(samples: u64) -> u64 {
    cache_work_slots(samples) * DESCRIPTOR_SIZE
}

pub fn cache_entry_bytes(samples: u64, buckets: u64) -> u64 {
    (cache_work_slots(samples) + buckets * CACHE_HISTORY_PER_BUCKET) * ENTRY_SIZE
}

/// A dispatch and a count per level, then room for every entry on each level.
pub fn merge_list_bytes(samples: u64, buckets: u64) -> u64 {
    let entries = cache_work_slots(samples) + buckets * CACHE_HISTORY_PER_BUCKET;
    (MERGE_LIST_HEADER + LEVELS as u64 * entries) * 4
}

pub fn cache_bucket_count(samples: u64, settings: &Settings) -> u64 {
    let capacity = u64::from(settings.sampling.max_cache_entries);
    if capacity > 0 {
        // A world cache must keep its capacity when the screen grid shrinks.
        let maximum_buckets = (capacity / CACHE_HISTORY_PER_BUCKET).max(1);
        1u64 << (u64::BITS - 1 - maximum_buckets.leading_zeros())
    } else {
        (samples * 2).next_power_of_two().max(1024)
    }
}

pub fn cache_history_bytes(buckets: u64) -> u64 {
    buckets * CACHE_HISTORY_PER_BUCKET * ENTRY_SIZE
}

pub fn update_budget(settings: &Settings) -> u32 {
    let ceiling = settings.max_rays.max(1);
    let requested = match settings.sampling.max_updates_per_frame {
        0 => ceiling,
        requested => requested,
    };
    requested.min(ceiling).max(1)
}

/// Trace and next-event queries stop at half the cap; gather may reach it.
pub fn traversal_query_caps(query_limit: u32) -> (u32, u32) {
    if query_limit == 0 {
        (0, 0)
    } else {
        ((query_limit / 2).max(1), query_limit)
    }
}

/// The sparse receiver grid: the smallest pixel divisor (at least the
/// configured one) whose grid fits the receiver cap.
pub fn resolution(viewport: [u32; 2], settings: &Settings) -> [u32; 2] {
    let size_at = |divisor: u32| viewport.map(|v| v.div_ceil(divisor).max(1));
    let max_receivers = u64::from(match settings.sampling.max_receiver_cells {
        0 => MAX_RECEIVERS,
        cells => cells,
    });
    let divisor = match settings.sampling.grid_divisor {
        0 => settings.resolution_divisor,
        divisor => divisor,
    };
    let (mut low, mut high) = (
        divisor.max(1),
        viewport[0].max(viewport[1]).max(divisor.max(1)),
    );
    while low < high {
        let middle = low + (high - low) / 2;
        let [x, y] = size_at(middle);
        if u64::from(x) * u64::from(y) > max_receivers {
            low = middle + 1;
        } else {
            high = middle;
        }
    }
    size_at(low)
}

/// Which irradiance image holds the displayed result.
pub fn composite_display_index(spatial_filter_steps: u32, parity: usize) -> usize {
    if spatial_filter_steps == 0 {
        parity
    } else {
        2 + spatial_filter_steps as usize % 2
    }
}

pub fn sparse_candidate_phase_word(
    candidate_phase: u32,
    reset: bool,
    updates_disabled: bool,
) -> u32 {
    (candidate_phase & 0x03ff_ffff)
        | if reset { SPARSE_HISTORY_RESET_FLAG } else { 0 }
        | if updates_disabled {
            SPARSE_UPDATES_DISABLED_FLAG
        } else {
            0
        }
}

/// Fair round-robin over the receiver candidates: each frame updates up to
/// the budget, starting where the last finished.
#[derive(Default)]
pub struct CandidateSchedule {
    candidates: u32,
    cursor: u32,
}

impl CandidateSchedule {
    /// This frame's starting candidate and quota; advances for the next frame.
    pub fn next(&mut self, candidates: u32, budget: u32) -> (u32, u32) {
        if candidates != self.candidates {
            (self.candidates, self.cursor) = (candidates, 0);
        }
        let phase = if candidates > 0 {
            self.cursor % candidates
        } else {
            0
        };
        let quota = budget.min(candidates);
        if candidates > 0 {
            self.cursor = ((u64::from(phase) + u64::from(quota)) % u64::from(candidates)) as u32;
        }
        (phase, quota)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parameters_match_the_shader_block() {
        // 4 matrices, 33 vec4s and 23 pointers, no padding.
        assert_eq!(std::mem::size_of::<Parameters>(), 4 * 64 + 33 * 16 + 23 * 8);
    }

    #[test]
    fn grid_respects_the_receiver_cap() {
        let settings = Settings::default();
        // Divisor 2 would give 230,400 receivers, over the 131,072 cap.
        assert_eq!(resolution([1280, 720], &settings), [427, 240]);
        let capped = Settings {
            sampling: Sampling {
                max_receiver_cells: 10_000,
                ..Sampling::default()
            },
            ..settings
        };
        let [x, y] = resolution([1280, 720], &capped);
        assert!(x * y <= 10_000 && (x + 20) * (y + 12) > 10_000, "{x}x{y}");
    }

    #[test]
    fn buckets_and_dispatch_groups() {
        let settings = Settings::default();
        assert_eq!(cache_bucket_count(230_400, &settings), 524_288);
        let fixed = Settings {
            sampling: Sampling {
                max_cache_entries: 65_536,
                ..Sampling::default()
            },
            ..settings
        };
        assert_eq!(cache_bucket_count(230_400, &fixed), 16_384);
        assert_eq!(pack_probe_dispatch_groups(1000), Some((256, 8)));
        assert_eq!(pack_probe_dispatch_groups(0), None);
    }

    #[test]
    fn schedule_visits_every_candidate() {
        let mut schedule = CandidateSchedule::default();
        let mut seen = [0u32; 10];
        for _ in 0..5 {
            let (phase, quota) = schedule.next(10, 4);
            for i in 0..quota {
                seen[((phase + i) % 10) as usize] += 1;
            }
        }
        assert!(seen.iter().all(|&n| n == 2));
    }
}
