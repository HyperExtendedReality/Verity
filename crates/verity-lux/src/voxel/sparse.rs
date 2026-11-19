//! Packed occupancy alternatives for one finite cubic clip region.
//!
//! Upload the word slices as `StructuredBuffer<uint>`, with the region origin
//! and side supplied separately. All addresses below are word indices relative
//! to the start of that region's slice, not byte offsets. The payload carries no
//! voxel material, lighting, or pool indices. Use the GPU's storage buffer size
//! limit in addition to the format limits; this module does not allocate GPU memory.

use std::{
    collections::{HashMap, TryReserveError, hash_map::Entry},
    error::Error,
    fmt,
};

const BRICK_SHIFT: u32 = 2;
const BRICK_SIDE: u32 = 1 << BRICK_SHIFT;
const BRICK_AXIS_BITS: u32 = 10;
const MAX_CLIP_SIDE: u32 = BRICK_SIDE << BRICK_AXIS_BITS;
// Keep even the byte size representable by a u32.
const MAX_PACKED_WORDS: usize = (u32::MAX / 4) as usize;
const HASH_HEADER_WORDS: usize = 2;
const HASH_SLOT_WORDS: usize = 3;
const HASH_MAX_SLOTS: usize = 1 << ((MAX_PACKED_WORDS - HASH_HEADER_WORDS) / HASH_SLOT_WORDS).ilog2();

/// An axis-aligned clip cube addressed by signed integer cell positions.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ClipRegion {
    origin: [i32; 3],
    side: u32,
}

impl ClipRegion {
    /// Creates an inclusive origin with `side` cells in each positive direction.
    ///
    /// # Errors
    ///
    /// The side must be a power of two from 1 to 4096 inclusive, and the last
    /// included cell on each axis must be representable by `i32`.
    pub(crate) fn new(origin: [i32; 3], side: u32) -> Result<Self, SparseBuildError> {
        if !side.is_power_of_two() || side > MAX_CLIP_SIDE {
            return Err(SparseBuildError::InvalidSide(side));
        }
        if origin.iter().any(|&start| start > i32::MAX - (side - 1) as i32) {
            return Err(SparseBuildError::RegionOverflow { origin, side });
        }
        Ok(Self { origin, side })
    }

    fn local(self, cell: [i32; 3]) -> Option<[u32; 3]> {
        // Shader equivalent: `uint3(cell) - uint3(origin)` then `all(local < side)`.
        // The constructor rules out a valid region wrapping across i32::MAX.
        let local = std::array::from_fn(|axis| (cell[axis] as u32).wrapping_sub(self.origin[axis] as u32));
        local.iter().all(|&axis| axis < self.side).then_some(local)
    }
}

/// An invalid clip region, out-of-region cell, or unrepresentable packed build.
#[derive(Debug)]
pub(crate) enum SparseBuildError {
    InvalidSide(u32),
    RegionOverflow { origin: [i32; 3], side: u32 },
    CellOutsideRegion([i32; 3]),
    SizeLimit,
    Allocation(TryReserveError),
}

impl fmt::Display for SparseBuildError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidSide(side) => write!(f, "clip side {side} is not a power of two in 1..={MAX_CLIP_SIDE}"),
            Self::RegionOverflow { origin, side } => write!(f, "clip at {origin:?} with side {side} exceeds signed cell coordinates"),
            Self::CellOutsideRegion(cell) => write!(f, "occupied cell {cell:?} is outside the clip region"),
            Self::SizeLimit => f.write_str("sparse voxel build exceeds the packed format size limit"),
            Self::Allocation(_) => f.write_str("could not allocate sparse voxel storage"),
        }
    }
}

impl Error for SparseBuildError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Allocation(error) => Some(error),
            _ => None,
        }
    }
}

impl From<TryReserveError> for SparseBuildError {
    fn from(error: TryReserveError) -> Self {
        Self::Allocation(error)
    }
}

/// A sparse hash grid of 4×4×4 occupancy bricks.
///
/// Words 0 and 1 are the slot mask (slot count minus one) and maximum occupied
/// probe count. A slot starts at `2 + 3*slot`: its words are `(key, low, high)`.
/// The key is `1 + morton(local >> 2)` (see `morton`), so zero unambiguously
/// marks an unused slot, including when brick `(0,0,0)` exists. Start at
/// `hash_slot(key) & slot_mask`, and advance by `(slot + 1) & slot_mask`. Lookup
/// stops at an empty slot, a matching key, or the maximum probe count. If the
/// byte size permits, at most 2²⁸ slots and 2²⁷ distinct bricks are supported.
///
/// This build publishes a snapshot from authoritative cells. Replacing the mask
/// of an existing key does not alter its probe chain; adding or removing keys
/// requires rebuilding here. Clearing an individual key would break lookups.
#[derive(Debug)]
pub(crate) struct HashBricks {
    region: ClipRegion,
    words: Vec<u32>,
}

impl HashBricks {
    /// Builds the sparse grid; duplicate cells are folded into one bit.
    ///
    /// # Errors
    ///
    /// Rejects out-of-region cells, a table above the format limit at half load,
    /// or failed host allocations.
    pub(crate) fn from_cells(
        region: ClipRegion,
        occupied: impl IntoIterator<Item = [i32; 3]>,
    ) -> Result<Self, SparseBuildError> {
        let bricks = collect_bricks(region, occupied, HASH_MAX_SLOTS / 2)?;
        let slots = (2 * bricks.len()).next_power_of_two();
        let word_count = HASH_HEADER_WORDS + slots * HASH_SLOT_WORDS;
        let mut words = Vec::new();
        words.try_reserve_exact(word_count)?;
        words.resize(word_count, 0);
        let slot_mask = (slots - 1) as u32;
        words[0] = slot_mask;
        for (morton, bits) in bricks {
            let key = morton + 1;
            let mut slot = hash_slot(key) & slot_mask;
            let mut probes = 1;
            loop {
                let index = slot_index(slot);
                if words[index] == 0 {
                    let [low, high] = mask_words(bits);
                    words[index..index + HASH_SLOT_WORDS].copy_from_slice(&[key, low, high]);
                    words[1] = words[1].max(probes);
                    break;
                }
                // Half load guarantees an empty slot before the first slot repeats.
                slot = (slot + 1) & slot_mask;
                probes += 1;
            }
        }
        Ok(Self { region, words })
    }

    /// The packed grid, including a header and one empty slot for an empty region.
    pub(crate) fn words(&self) -> &[u32] {
        &self.words
    }

    /// Reads a cell directly from packed words; cells outside the clip miss.
    pub(crate) fn occupied(&self, cell: [i32; 3]) -> bool {
        self.region.local(cell).is_some_and(|local| self.occupied_local(local))
    }

    /// Traces the packed grid for the first occupied cell in `[0, max_t)`.
    ///
    /// Only positive-length cell visits count. The direction need not be
    /// normalized. Non-finite inputs, a zero direction, and `max_t <= 0` miss.
    pub(crate) fn trace(&self, origin: [f64; 3], direction: [f64; 3], max_t: f64) -> Option<VoxelHit> {
        trace_cells(self.region, origin, direction, max_t, |cell| self.occupied_local(cell))
    }

    fn occupied_local(&self, cell: [u32; 3]) -> bool {
        let key = brick_morton(cell) + 1;
        let slot_mask = self.words[0];
        let mut slot = hash_slot(key) & slot_mask;
        for _ in 0..self.words[1] {
            let index = slot_index(slot);
            let tag = self.words[index];
            if tag == key {
                return mask_contains(&self.words, index + 1, cell);
            }
            if tag == 0 {
                break;
            }
            slot = (slot + 1) & slot_mask;
        }
        false
    }
}

/// A canonical sparse voxel DAG for static occupancy.
///
/// Word 0 gives the root handle. Zero handles are empty; other handles are
/// absolute offsets within this slice. At extent greater than 4, read 8 child
/// handles, in `x | (y << 1) | (z << 2)` octant order using the next highest
/// coordinate bit, then halve the extent. At extent 4 or smaller, read a
/// `(low, high)` occupancy mask, with unused bits zero for extent 1 or 2. A mask
/// bit is `x + 4*y + 16*z`, where each coordinate is within its 4-cell brick;
/// bit 0 is the low word's LSB, bit 32 is the high word's LSB. Hash bricks use
/// the same mask layout. No shader 64-bit operations are required.
///
/// Identical leaves and child arrays share storage. References point strictly
/// backward; no record is overwritten. Indices are deterministic for the same
/// region-local occupancy regardless of source cell order, not across edits.
/// Rebuild the whole snapshot when its authoritative occupancy changes.
#[derive(Debug)]
pub(crate) struct VoxelDag {
    region: ClipRegion,
    words: Vec<u32>,
}

impl VoxelDag {
    /// Builds a canonical DAG; duplicate cells are folded into one bit.
    ///
    /// # Errors
    ///
    /// Rejects out-of-region cells, an arena exceeding `u32::MAX / 4` words,
    /// or failed host allocations.
    pub(crate) fn from_cells(
        region: ClipRegion,
        occupied: impl IntoIterator<Item = [i32; 3]>,
    ) -> Result<Self, SparseBuildError> {
        let bricks = collect_bricks(region, occupied, 1 << (3 * BRICK_AXIS_BITS))?;
        let mut frontier = Vec::new();
        frontier.try_reserve_exact(bricks.len())?;
        frontier.extend(bricks);
        frontier.sort_unstable_by_key(|&(morton, _)| morton);

        let mut words = Vec::new();
        words.try_reserve_exact(1)?;
        words.push(0);
        {
            let mut leaves = HashMap::new();
            for (_, payload) in &mut frontier {
                let handle = intern(&mut words, &mut leaves, mask_words(*payload))?;
                *payload = u64::from(handle);
            }
        }

        let mut branches = HashMap::new();
        for _ in BRICK_SHIFT..region.side.ilog2() {
            // Each nonempty array refers to the immediately preceding level, so
            // a child array cannot also occur at a different level.
            branches.clear();
            let mut read = 0;
            let mut write = 0;
            while read < frontier.len() {
                let parent = frontier[read].0 >> 3;
                let mut children = [0; 8];
                while let Some(&(morton, handle)) = frontier.get(read) {
                    if morton >> 3 != parent {
                        break;
                    }
                    children[(morton & 7) as usize] = handle as u32;
                    read += 1;
                }
                let handle = intern(&mut words, &mut branches, children)?;
                frontier[write] = (parent, u64::from(handle));
                write += 1;
            }
            frontier.truncate(write);
        }
        if let Some(&(_, root)) = frontier.first() {
            words[0] = root as u32;
        }
        Ok(Self { region, words })
    }

    /// The packed DAG, or the single zero root word for an empty region.
    pub(crate) fn words(&self) -> &[u32] {
        &self.words
    }

    /// Reads a cell directly from packed words; cells outside the clip miss.
    pub(crate) fn occupied(&self, cell: [i32; 3]) -> bool {
        self.region.local(cell).is_some_and(|local| self.occupied_local(local))
    }

    /// Traces the packed DAG for the first occupied cell in `[0, max_t)`.
    ///
    /// Uses the same finite ray and half-open cell contract as [`HashBricks::trace`].
    pub(crate) fn trace(&self, origin: [f64; 3], direction: [f64; 3], max_t: f64) -> Option<VoxelHit> {
        trace_cells(self.region, origin, direction, max_t, |cell| self.occupied_local(cell))
    }

    fn occupied_local(&self, cell: [u32; 3]) -> bool {
        let mut node = self.words[0] as usize;
        for bit in (BRICK_SHIFT..self.region.side.ilog2()).rev() {
            if node == 0 {
                return false;
            }
            let [x, y, z] = cell.map(|coordinate| (coordinate >> bit) & 1);
            node = self.words[node + (x | (y << 1) | (z << 2)) as usize] as usize;
        }
        node != 0 && mask_contains(&self.words, node, cell)
    }
}

fn collect_bricks(
    region: ClipRegion,
    occupied: impl IntoIterator<Item = [i32; 3]>,
    max_bricks: usize,
) -> Result<HashMap<u32, u64>, SparseBuildError> {
    let mut bricks = HashMap::new();
    for cell in occupied {
        let local = region.local(cell).ok_or(SparseBuildError::CellOutsideRegion(cell))?;
        let key = brick_morton(local);
        let bit = 1u64 << brick_bit(local);
        if bricks.len() == bricks.capacity() || bricks.len() == max_bricks {
            if let Some(mask) = bricks.get_mut(&key) {
                *mask |= bit;
                continue;
            }
            if bricks.len() == max_bricks {
                return Err(SparseBuildError::SizeLimit);
            }
            bricks.try_reserve(1)?;
        }
        *bricks.entry(key).or_default() |= bit;
    }
    Ok(bricks)
}

fn intern<const N: usize>(
    words: &mut Vec<u32>,
    nodes: &mut HashMap<[u32; N], u32>,
    record: [u32; N],
) -> Result<u32, SparseBuildError> {
    if nodes.len() == nodes.capacity() {
        if let Some(&handle) = nodes.get(&record) {
            return Ok(handle);
        }
        nodes.try_reserve(1)?;
    }
    match nodes.entry(record) {
        Entry::Occupied(entry) => Ok(*entry.get()),
        Entry::Vacant(entry) => {
            let handle = words.len();
            if handle > MAX_PACKED_WORDS - N {
                return Err(SparseBuildError::SizeLimit);
            }
            words.try_reserve(N)?;
            words.extend_from_slice(&record);
            entry.insert(handle as u32);
            Ok(handle as u32)
        }
    }
}

fn brick_morton(cell: [u32; 3]) -> u32 {
    morton(cell.map(|axis| axis >> BRICK_SHIFT))
}

// Interleave ten bits per axis, x in bit 0, y in bit 1, z in bit 2 of each triple.
fn morton([x, y, z]: [u32; 3]) -> u32 {
    fn spread(x: u32) -> u32 {
        let x = (x | (x << 16)) & 0x0300_00ff;
        let x = (x | (x << 8)) & 0x0300_f00f;
        let x = (x | (x << 4)) & 0x030c_30c3;
        (x | (x << 2)) & 0x0924_9249
    }
    spread(x) | (spread(y) << 1) | (spread(z) << 2)
}

// Hash the nonzero stored key with the same wrapping u32 operations in Slang.
fn hash_slot(mut key: u32) -> u32 {
    key ^= key >> 16;
    key = key.wrapping_mul(0x7feb_352d);
    key ^= key >> 15;
    key = key.wrapping_mul(0x846c_a68b);
    key ^ (key >> 16)
}

fn slot_index(slot: u32) -> usize {
    HASH_HEADER_WORDS + HASH_SLOT_WORDS * slot as usize
}

fn brick_bit(cell: [u32; 3]) -> u32 {
    let [x, y, z] = cell.map(|axis| axis & (BRICK_SIDE - 1));
    x | (y << BRICK_SHIFT) | (z << (2 * BRICK_SHIFT))
}

fn mask_words(bits: u64) -> [u32; 2] {
    [bits as u32, (bits >> 32) as u32]
}

fn mask_contains(words: &[u32], start: usize, cell: [u32; 3]) -> bool {
    let bit = brick_bit(cell);
    words[start + (bit >> 5) as usize] & (1 << (bit & 31)) != 0
}

/// The first occupied cell traversed by a reference ray.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct VoxelHit {
    /// Signed position of the hit cell, in clip cell coordinates.
    pub(crate) cell: [i32; 3],
    /// Entry parameter in `origin + direction * t`, clamped to the ray start.
    pub(crate) t: f64,
}

// Reference point-sampled occupancy DDA, not a hierarchical empty-space skip.
// Half-open cells and a finite parameter range; only positive-length visits
// count. Step all axes at a tied boundary to avoid testing point-only cells.
fn trace_cells(
    region: ClipRegion,
    origin: [f64; 3],
    direction: [f64; 3],
    max_t: f64,
    mut occupied: impl FnMut([u32; 3]) -> bool,
) -> Option<VoxelHit> {
    if !max_t.is_finite()
        || max_t <= 0.0
        || !origin.into_iter().chain(direction).all(f64::is_finite)
        || direction == [0.0; 3]
    {
        return None;
    }
    let origin: [f64; 3] = std::array::from_fn(|axis| origin[axis] - f64::from(region.origin[axis]));
    let side = region.side as i32;
    let side_f64 = f64::from(side);
    let mut step = [0; 3];
    let mut slab_entry = [f64::NEG_INFINITY; 3];
    let mut t: f64 = 0.0;
    let mut exit = max_t;
    for axis in 0..3 {
        let d = direction[axis];
        let o = origin[axis];
        if d == 0.0 {
            if !(0.0..side_f64).contains(&o) {
                return None;
            }
            continue;
        }
        let at_low = -o / d;
        let at_high = (side_f64 - o) / d;
        let (entry, leave) = if d > 0.0 {
            step[axis] = 1;
            (at_low, at_high)
        } else {
            step[axis] = -1;
            (at_high, at_low)
        };
        slab_entry[axis] = entry;
        t = t.max(entry);
        exit = exit.min(leave);
    }
    if t >= exit {
        return None;
    }
    let mut cell: [i32; 3] = std::array::from_fn(|axis| {
        if slab_entry[axis] == t {
            // Pin a clipping plane rather than rounding an entry point back
            // outside due to cancellation in the ray equation.
            return if step[axis] > 0 { 0 } else { side - 1 };
        }
        let point = direction[axis].mul_add(t, origin[axis]).clamp(0.0, side_f64);
        let floor = point.floor();
        let retreat = i32::from(step[axis] < 0 && point == floor);
        (floor as i32 - retreat).clamp(0, side - 1)
    });
    let next_plane = |axis: usize, cell: i32| {
        if step[axis] == 0 {
            f64::INFINITY
        } else {
            let plane = cell + i32::from(step[axis] > 0);
            // Use each plane directly so cumulative rounding cannot split a tie.
            (f64::from(plane) - origin[axis]) / direction[axis]
        }
    };
    let mut crossing: [f64; 3] = std::array::from_fn(|axis| next_plane(axis, cell[axis]));
    loop {
        let boundary = crossing[0].min(crossing[1]).min(crossing[2]);
        if t < boundary.min(exit) && occupied(cell.map(|axis| axis as u32)) {
            return Some(VoxelHit {
                cell: std::array::from_fn(|axis| region.origin[axis] + cell[axis]),
                t,
            });
        }
        if boundary >= exit {
            return None;
        }
        t = t.max(boundary);
        for axis in 0..3 {
            if crossing[axis] == boundary {
                cell[axis] += step[axis];
                if !(0..side).contains(&cell[axis]) {
                    return None;
                }
                crossing[axis] = next_plane(axis, cell[axis]);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dag_shares_repeated_branches_and_leaves_in_stable_order() {
        let region = ClipRegion::new([0; 3], 16).expect("valid clip region");
        let cells = [[1, 2, 3], [5, 2, 3], [9, 2, 3], [13, 2, 3]];
        let dag = VoxelDag::from_cells(region, cells).expect("pack repeated occupancy");
        let reordered = VoxelDag::from_cells(region, cells.into_iter().rev()).expect("repack reversed occupancy");
        let words = dag.words();
        assert_eq!(words, reordered.words());
        assert_eq!(words.len(), 1 + 2 + 8 + 8);
        let root = words[0] as usize;
        let branch = words[root] as usize;
        assert_eq!(&words[root..root + 8], &[branch as u32, branch as u32, 0, 0, 0, 0, 0, 0]);
        let leaf = words[branch] as usize;
        assert_eq!(&words[branch..branch + 8], &[leaf as u32, leaf as u32, 0, 0, 0, 0, 0, 0]);
        assert_eq!(&words[leaf..leaf + 2], &[0, 1 << 25]);
        for x in 0..16 {
            let cell = [x, 2, 3];
            assert_eq!(dag.occupied(cell), cells.contains(&cell), "cell {cell:?}");
        }
    }

    #[test]
    fn packed_consumers_lookup_and_trace_across_bricks() {
        let region = ClipRegion::new([-8, 10, -4], 16).expect("valid translated region");
        let cells = [[-5, 10, -2], [-4, 10, -2], [0, 11, -1], [-5, 10, -2]];
        let bricks = HashBricks::from_cells(region, cells).expect("pack hash bricks");
        let dag = VoxelDag::from_cells(region, cells).expect("pack DAG");
        for cell in cells {
            assert!(bricks.occupied(cell), "hash missed {cell:?}");
            assert!(dag.occupied(cell), "DAG missed {cell:?}");
        }
        for cell in [[-6, 10, -2], [0, 10, -1], [8, 10, -2]] {
            assert!(!bricks.occupied(cell), "hash hit {cell:?}");
            assert!(!dag.occupied(cell), "DAG hit {cell:?}");
        }
        let origin = [-10.5, 11.5, -0.5];
        let direction = [1.0, 0.0, 0.0];
        let hit = Some(VoxelHit { cell: [0, 11, -1], t: 10.5 });
        assert_eq!(bricks.trace(origin, direction, 20.0), hit);
        assert_eq!(dag.trace(origin, direction, 20.0), hit);
        let origin = [8.0, 10.5, -1.5];
        let direction = [-2.0, 0.0, 0.0];
        let hit = Some(VoxelHit { cell: [-4, 10, -2], t: 5.5 });
        assert_eq!(bricks.trace(origin, direction, 20.0), hit);
        assert_eq!(dag.trace(origin, direction, 20.0), hit);
    }

    #[test]
    fn hash_probes_all_colliding_keys_and_bounds_misses() {
        let region = ClipRegion::new([0; 3], 64).expect("valid clip region");
        let mut by_slot = [const { Vec::new() }; 4];
        for x in (0..64).step_by(4) {
            let cell = [x, 0, 0];
            let key = brick_morton(cell.map(|axis| axis as u32)) + 1;
            by_slot[(hash_slot(key) & 3) as usize].push(cell);
        }
        let colliding = by_slot.iter().max_by_key(|cells| cells.len()).expect("four home slots");
        let bricks = HashBricks::from_cells(region, [colliding[0], colliding[1]]).expect("pack colliding bricks");
        assert_eq!(&bricks.words()[..2], &[3, 2]);
        assert!(bricks.occupied(colliding[0]));
        assert!(bricks.occupied(colliding[1]));
        assert!(!bricks.occupied(colliding[2]));
    }

    #[test]
    fn corner_touches_and_terminal_points_are_not_ray_hits() {
        let region = ClipRegion::new([0; 3], 8).expect("valid clip region");
        let dag = VoxelDag::from_cells(region, [[1, 0, 0], [0, 1, 0], [2, 2, 0]]).expect("pack diagonal occupancy");
        let origin = [0.5; 3];
        let direction = [1.0, 1.0, 0.0];
        assert_eq!(dag.trace(origin, direction, 1.5), None);
        assert_eq!(dag.trace(origin, direction, 2.0), Some(VoxelHit { cell: [2, 2, 0], t: 1.5 }));
    }

    #[test]
    fn clip_limits_and_empty_snapshots_do_not_alias_keys() {
        assert!(matches!(ClipRegion::new([0; 3], 3), Err(SparseBuildError::InvalidSide(3))));
        assert!(matches!(ClipRegion::new([0; 3], 8192), Err(SparseBuildError::InvalidSide(8192))));
        assert!(matches!(ClipRegion::new([i32::MAX; 3], 2), Err(SparseBuildError::RegionOverflow { .. })));
        let region = ClipRegion::new([i32::MAX; 3], 1).expect("last representable cell is valid");
        assert!(matches!(HashBricks::from_cells(region, [[i32::MIN; 3]]), Err(SparseBuildError::CellOutsideRegion(_))));
        assert!(matches!(VoxelDag::from_cells(region, [[i32::MIN; 3]]), Err(SparseBuildError::CellOutsideRegion(_))));
        let bricks = HashBricks::from_cells(region, []).expect("pack empty hash grid");
        let dag = VoxelDag::from_cells(region, []).expect("pack empty DAG");
        assert_eq!(bricks.words(), &[0; 5]);
        assert_eq!(dag.words(), &[0]);
        assert!(!bricks.occupied([i32::MAX; 3]));
        assert!(!dag.occupied([i32::MAX; 3]));

        let region = ClipRegion::new([0; 3], MAX_CLIP_SIDE).expect("valid maximum clip side");
        let last = [(MAX_CLIP_SIDE - 1) as i32; 3];
        let bricks = HashBricks::from_cells(region, [[0; 3], last]).expect("pack both extreme keys");
        let dag = VoxelDag::from_cells(region, [[0; 3], last]).expect("pack deepest DAG");
        let slots = &bricks.words()[HASH_HEADER_WORDS..];
        assert!(slots.chunks_exact(HASH_SLOT_WORDS).any(|slot| slot == [1, 1, 0]));
        assert!(slots.chunks_exact(HASH_SLOT_WORDS).any(|slot| slot == [0x4000_0000, 0, 1 << 31]));
        for cell in [[0; 3], last] {
            assert!(bricks.occupied(cell));
            assert!(dag.occupied(cell));
        }
        assert!(!bricks.occupied([last[0], 0, 0]));
        assert!(!dag.occupied([last[0], 0, 0]));
    }
}
