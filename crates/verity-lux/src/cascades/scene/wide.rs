//! Immutable 4-wide software BLAS records, mirrored by `traversal.slang`.
//!
//! A mesh starts with two words: `(root min, record count | enhanced flag)`
//! and `(root max, root reference)`. References have a four-bit kind and a
//! 28-bit payload. The count's high bit enables extended bounds/leaf decoding. Zero
//! is unused; kind 0 points to a four-word record (one-based); kinds 1–8 give
//! the count of an inline triangle span, with a mesh-relative first triangle.
//! Kind 9 points to a record containing the full relative `(first, count)`.
//! Kinds 10–12 are branch records in the xy, xz and yz frames, respectively.
//! Kind 13 points to a record holding a reference count followed by up to
//! eight mesh-relative source triangle IDs; vertices and indices stay shared.
//!
//! An AABB record is `(frame min, ref0)`, `(frame max, ref1)`,
//! `(loX, hiX, loY, hiY)`, `(loZ, hiZ, ref2, ref3)`. Each coordinate field
//! holds one byte per child, least significant first. Bins 0 and 255 select
//! the frame endpoints; interior bins interpolate at q/256 of the grid span.

use std::{cell::Cell, ops::Range};

use glam::{UVec4, Vec3};

use super::{BvhNode, oriented, vec_word};

const LANES: usize = 4;
const RECORD_WORDS: usize = 4;
const HEADER_WORDS: usize = 2;
const KIND_SHIFT: u32 = 28;
const PAYLOAD_MASK: u32 = (1 << KIND_SHIFT) - 1;
const ENHANCED: u32 = 1 << 31;
const MAX_INLINE_COUNT: u32 = 8;
const FULL_SPAN: u32 = 9;
const ORIENTED_XY: u32 = 10;
const INDIRECT_SPAN: u32 = 13;
const TOP_BIN: u32 = 255;
const BINS: f32 = 256.0;
// Keep at most 16 significand bits so scaling by an 8-bit bin is exact,
// whether the shader contracts the multiply-add or evaluates it separately.
const GRID_MASK: u32 = 0xffff_ff00;

pub(super) fn pack(nodes: &[BvhNode]) -> Vec<UVec4> {
    pack_impl::<false, false>(nodes, &[], &[])
}

pub(super) fn pack_static(
    nodes: &[BvhNode],
    corners: Option<&[[Vec3; 3]]>,
    references: &[u32],
) -> Vec<UVec4> {
    match (corners, references.is_empty()) {
        (Some(corners), true) => pack_impl::<true, false>(nodes, corners, &[]),
        (Some(corners), false) => pack_impl::<true, true>(nodes, corners, references),
        (None, false) => pack_impl::<false, true>(nodes, &[], references),
        (None, true) => pack(nodes),
    }
}

fn pack_impl<const ORIENTED: bool, const SPATIAL: bool>(
    nodes: &[BvhNode],
    corners: &[[Vec3; 3]],
    references: &[u32],
) -> Vec<UVec4> {
    let Some(root) = nodes.first() else {
        return Vec::new();
    };
    // A represented inner node other than the root has a skipped binary
    // parent, which can supply at most two such nodes. Indirect leaves also
    // allocate their own records.
    let records = if SPATIAL {
        nodes.len()
    } else {
        nodes.len() / 3
    };
    let mut words = Vec::with_capacity(HEADER_WORDS + RECORD_WORDS * records);
    words.extend([UVec4::ZERO; HEADER_WORDS]);
    let mut spans = Vec::new();
    if ORIENTED {
        spans.resize(nodes.len(), 0..0);
        collect_spans(nodes, &mut spans, 0);
    }
    let inputs = PackInputs {
        nodes,
        corners,
        references,
        spans: &spans,
        used_orientation: Cell::new(false),
    };
    let reference = pack_reference::<ORIENTED, SPATIAL>(&mut words, &inputs, 0);
    let records = ((words.len() - HEADER_WORDS) / RECORD_WORDS) as u32;
    let enhanced = if SPATIAL || (ORIENTED && inputs.used_orientation.get()) {
        ENHANCED
    } else {
        0
    };
    words[0] = vec_word(root.min, records | enhanced);
    words[1] = vec_word(root.max, reference);
    words
}

fn reserve_record(words: &mut Vec<UVec4>) -> (usize, u32) {
    let at = words.len();
    let index = (at - HEADER_WORDS) / RECORD_WORDS + 1;
    assert!(
        index <= PAYLOAD_MASK as usize,
        "wide BLAS record index {index} exceeds its packed field"
    );
    words.extend([UVec4::ZERO; RECORD_WORDS]);
    (at, index as u32)
}

fn collect_spans(nodes: &[BvhNode], spans: &mut [Range<usize>], at: usize) -> Range<usize> {
    let node = nodes[at];
    let first = node.first as usize;
    let span = if node.count > 0 {
        first..first + node.count as usize
    } else {
        let left = collect_spans(nodes, spans, first);
        let right = collect_spans(nodes, spans, first + 1);
        // Object and spatial builders may emit their contiguous leaves in opposite orders.
        left.start.min(right.start)..left.end.max(right.end)
    };
    spans[at] = span.clone();
    span
}

struct PackInputs<'mesh> {
    nodes: &'mesh [BvhNode],
    corners: &'mesh [[Vec3; 3]],
    references: &'mesh [u32],
    spans: &'mesh [Range<usize>],
    used_orientation: Cell<bool>,
}

fn pack_reference<const ORIENTED: bool, const SPATIAL: bool>(
    words: &mut Vec<UVec4>,
    inputs: &PackInputs<'_>,
    node_index: usize,
) -> u32 {
    let node = &inputs.nodes[node_index];
    if node.count != 0 {
        if SPATIAL {
            let (at, index) = reserve_record(words);
            let first = node.first as usize;
            let ids = &inputs.references[first..first + node.count as usize];
            words[at].x = node.count;
            for (i, &id) in (1..).zip(ids) {
                words[at + i / 4][i % 4] = id;
            }
            return (INDIRECT_SPAN << KIND_SHIFT) | index;
        }
        if node.first <= PAYLOAD_MASK && node.count <= MAX_INLINE_COUNT {
            return (node.count << KIND_SHIFT) | node.first;
        }
        let (at, index) = reserve_record(words);
        words[at] = UVec4::new(node.first, node.count, 0, 0);
        return (FULL_SPAN << KIND_SHIFT) | index;
    }

    let (at, index) = reserve_record(words);
    let mut children = [0; LANES];
    let mut len = 0;
    let left = node.first as usize;
    // Expand both internal children, skipping two binary levels per wide level.
    for (lane, child) in inputs.nodes[left..left + 2].iter().enumerate() {
        if child.count == 0 {
            let first = child.first as usize;
            children[len..len + 2].copy_from_slice(&[first, first + 1]);
            len += 2;
        } else {
            children[len] = left + lane;
            len += 1;
        }
    }

    let mut bounds = [(Vec3::ZERO, Vec3::ZERO); LANES];
    for (lane, &child) in children[..len].iter().enumerate() {
        let node = inputs.nodes[child];
        bounds[lane] = (node.min, node.max);
    }
    let (mut min, mut max) = (node.min, node.max);
    let mut kind = 0;
    if ORIENTED {
        let spans = children.map(|index| inputs.spans[index].clone());
        if let Some(projected) = oriented::choose_indexed(
            inputs.corners,
            inputs.references,
            &spans[..len],
            &bounds[..len],
        ) {
            inputs.used_orientation.set(true);
            kind = ORIENTED_XY + projected.frame;
            bounds = projected.bounds;
            (min, max) = bounds[0];
            for &(lo, hi) in &bounds[1..len] {
                min = min.min(lo);
                max = max.max(hi);
            }
        }
    }
    let frame: [Axis; 3] = std::array::from_fn(|axis| Axis::new(min[axis], max[axis]));
    let mut refs = [0; LANES];
    let mut quantized = [[0; 2]; 3];
    for (lane, &child) in children[..len].iter().enumerate() {
        refs[lane] = pack_reference::<ORIENTED, SPATIAL>(words, inputs, child);
        let (min, max) = bounds[lane];
        for (axis, frame) in frame.iter().enumerate() {
            let [lo, hi] = frame.quantize(min[axis], max[axis]);
            let shift = lane * 8;
            quantized[axis][0] |= lo << shift;
            quantized[axis][1] |= hi << shift;
        }
    }
    words[at..at + RECORD_WORDS].copy_from_slice(&[
        vec_word(
            Vec3::from_array(frame.each_ref().map(|axis| axis.min)),
            refs[0],
        ),
        vec_word(
            Vec3::from_array(frame.each_ref().map(|axis| axis.max)),
            refs[1],
        ),
        UVec4::new(
            quantized[0][0],
            quantized[0][1],
            quantized[1][0],
            quantized[1][1],
        ),
        UVec4::new(quantized[2][0], quantized[2][1], refs[2], refs[3]),
    ]);
    (kind << KIND_SHIFT) | index
}

struct Axis {
    min: f32,
    max: f32,
    step: f32,
    bins_per_unit: f64,
}

impl Axis {
    fn new(min: f32, max: f32) -> Self {
        // Do not put subnormal operands into the shader's interpolation.
        // Expanding them to zero or the nearest normal also keeps endpoints
        // conservative on GPUs that flush subnormals to zero.
        let min = if !min.is_subnormal() {
            min
        } else if min.is_sign_negative() {
            -f32::MIN_POSITIVE
        } else {
            0.0
        };
        let max = if !max.is_subnormal() {
            max
        } else if max.is_sign_positive() {
            f32::MIN_POSITIVE
        } else {
            0.0
        };
        let step = grid_step(min, max);
        let bins_per_unit = if step.is_normal() {
            f64::from(step).recip()
        } else {
            0.0
        };
        Self {
            min,
            max,
            step,
            bins_per_unit,
        }
    }

    fn quantize(&self, lo: f32, hi: f32) -> [u32; 2] {
        if self.bins_per_unit == 0.0 {
            // A zero, tiny or overflowing span uses the exact frame on this
            // axis. The other axes are still independently compressed.
            return [0, TOP_BIN];
        }
        let bin = |value| (f64::from(value) - f64::from(self.min)) * self.bins_per_unit;
        let mut lower = (bin(lo) as u32).min(TOP_BIN);
        let mut upper = (bin(hi).ceil() as u32).min(TOP_BIN);
        // Verify against the rounded shader endpoints, not just the f64 bins.
        while lower != 0 {
            let decoded = decode_bound(self.min, self.max, self.step, lower);
            if decoded <= lo && !decoded.is_subnormal() {
                break;
            }
            lower -= 1;
        }
        while upper != TOP_BIN {
            let decoded = decode_bound(self.min, self.max, self.step, upper);
            if decoded >= hi && !decoded.is_subnormal() {
                break;
            }
            upper += 1;
        }
        [lower, upper]
    }
}

fn grid_step(min: f32, max: f32) -> f32 {
    f32::from_bits((max - min).to_bits() & GRID_MASK) / BINS
}

fn decode_bound(min: f32, max: f32, step: f32, bin: u32) -> f32 {
    match bin {
        0 => min,
        TOP_BIN => max,
        _ => min + step * bin as f32,
    }
}

#[cfg(test)]
pub(super) mod packed {
    use super::*;

    pub(in super::super) enum Reference {
        Branch(usize),
        Oriented(usize, u32),
        Indirect(usize, u32),
        Span(u32, u32),
    }

    pub(in super::super) fn root(words: &[UVec4], base: usize) -> (Vec3, Vec3, u32, &[UVec4]) {
        let [a, b] = [words[base], words[base + 1]];
        let begin = base + HEADER_WORDS;
        let end = begin + (a.w & !ENHANCED) as usize * RECORD_WORDS;
        (point(a), point(b), b.w, &words[begin..end])
    }

    pub(in super::super) fn reference(words: &[UVec4], reference: u32) -> Reference {
        let payload = reference & PAYLOAD_MASK;
        let kind = reference >> KIND_SHIFT;
        match kind {
            0 => Reference::Branch((payload as usize - 1) * RECORD_WORDS),
            ORIENTED_XY..=12 => {
                Reference::Oriented((payload as usize - 1) * RECORD_WORDS, kind - ORIENTED_XY)
            }
            INDIRECT_SPAN => {
                let at = (payload as usize - 1) * RECORD_WORDS;
                Reference::Indirect(at, words[at].x)
            }
            FULL_SPAN => {
                let span = words[(payload as usize - 1) * RECORD_WORDS];
                Reference::Span(span.x, span.y)
            }
            1..=MAX_INLINE_COUNT => Reference::Span(payload, kind),
            _ => panic!("unknown packed wide reference {reference:#x}"),
        }
    }

    pub(in super::super) fn source(words: &[UVec4], at: usize, index: u32) -> u32 {
        let index = index as usize + 1;
        words[at + index / 4][index % 4]
    }

    pub(in super::super) fn children(
        words: &[UVec4],
        at: usize,
    ) -> impl Iterator<Item = (Vec3, Vec3, u32)> {
        let (a, b, q, r) = (words[at], words[at + 1], words[at + 2], words[at + 3]);
        let (min, max) = (point(a), point(b));
        let step: [f32; 3] = std::array::from_fn(|axis| grid_step(min[axis], max[axis]));
        [a.w, b.w, r.z, r.w]
            .into_iter()
            .enumerate()
            .filter_map(move |(lane, reference)| {
                if reference == 0 {
                    return None;
                }
                let coord = |word| (word >> (lane * 8)) & TOP_BIN;
                let decode = |words: [u32; 3]| {
                    Vec3::from_array(std::array::from_fn(|axis| {
                        decode_bound(min[axis], max[axis], step[axis], coord(words[axis]))
                    }))
                };
                Some((decode([q.x, q.z, r.x]), decode([q.y, q.w, r.y]), reference))
            })
    }

    fn point(word: UVec4) -> Vec3 {
        Vec3::new(
            f32::from_bits(word.x),
            f32::from_bits(word.y),
            f32::from_bits(word.z),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::packed::{self, Reference};
    use super::*;

    #[test]
    fn decoded_child_bounds_enclose_extreme_and_degenerate_axes() {
        let large = 2.0_f32.powi(96);
        let tiny = f32::MIN_POSITIVE * 0.5;
        let children = [
            BvhNode {
                min: Vec3::new(-f32::MAX, -tiny, large),
                max: Vec3::new(-large, tiny, large),
                first: 0,
                count: 1,
            },
            BvhNode {
                min: Vec3::new(large, tiny, large),
                max: Vec3::new(f32::MAX, tiny, large),
                first: 1,
                count: 1,
            },
        ];
        let root = BvhNode {
            min: children[0].min.min(children[1].min),
            max: children[0].max.max(children[1].max),
            first: 1,
            count: 0,
        };
        let words = pack(&[root, children[0], children[1]]);
        let (min, max, reference, records) = packed::root(&words, 0);
        assert_eq!((min, max), (root.min, root.max));
        let Reference::Branch(at) = packed::reference(records, reference) else {
            panic!("expected a packed wide branch");
        };
        let decoded: Vec<_> = packed::children(records, at).collect();
        assert_eq!(decoded.len(), children.len());
        for ((lo, hi, reference), source) in decoded.into_iter().zip(children) {
            assert!(lo.cmple(source.min).all() && hi.cmpge(source.max).all());
            assert!(lo.is_finite() && hi.is_finite());
            assert!(
                matches!(packed::reference(records, reference), Reference::Span(first, 1) if first == source.first)
            );
        }
    }

    #[test]
    fn interior_bins_always_decode_outward_at_large_offsets() {
        let base = 2.0_f32.powi(40);
        for (min, max) in [(base, base + 2.0_f32.powi(28)), (-base, base), (-1.0, 0.3)] {
            let axis = Axis::new(min, max);
            for i in 0..1024 {
                let p = (f64::from(min) + (f64::from(max) - f64::from(min)) * f64::from(i) / 1024.0)
                    as f32;
                let [lo, hi] = axis.quantize(p, p);
                assert!(decode_bound(axis.min, axis.max, axis.step, lo) <= p);
                assert!(decode_bound(axis.min, axis.max, axis.step, hi) >= p);
            }
        }
    }

    #[test]
    fn full_span_preserves_offsets_and_counts_that_do_not_fit_inline() {
        let node = BvhNode {
            min: Vec3::ZERO,
            max: Vec3::ONE,
            first: PAYLOAD_MASK + 1,
            count: MAX_INLINE_COUNT + 1,
        };
        let words = pack(&[node]);
        let (_, _, reference, records) = packed::root(&words, 0);
        assert!(matches!(
            packed::reference(records, reference),
            Reference::Span(first, count) if (first, count) == (node.first, node.count)
        ));
    }
}
