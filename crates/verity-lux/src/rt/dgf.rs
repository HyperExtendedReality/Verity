//! Exact DGF1 for ray-query candidates that are re-tested against the original triangles.
//!
//! Layout and limits follow the [Vulkan DGF1 specification][spec] and AMD's
//! [DGFLib][sdk]. Each block is one contiguous run of input triangles. RESTART
//! topology preserves corner order and primitive IDs without a remap, but packs
//! fewer triangles than a clustered strip encoder. The encoder does not emit
//! palettes, opacity micromaps, or user data.
//!
//! [spec]: https://docs.vulkan.org/spec/latest/chapters/VK_AMDX_dense_geometry_format/dense_geometry_format.html
//! [sdk]: https://github.com/GPUOpen-LibrariesAndSDKs/DGF-SDK/tree/main/DGFLib

use std::fmt;

/// DGF1 block stride in bytes. Vulkan also requires this device-address alignment.
pub(crate) const BLOCK_SIZE: usize = 128;
const BLOCK_BITS: usize = BLOCK_SIZE * 8;
const HEADER_BITS: usize = 20 * 8;
const MAX_FRONT_BITS: usize = 96 * 8;
const MAX_REUSE_BITS: usize = 24 * 8;
const MAX_VERTICES: usize = 64;
const MAX_TRIANGLES: usize = 64;
const MAX_PRIMITIVE_INDEX: u32 = (1 << 29) - 1;
const MAGIC: u32 = 0x06;
const EXPONENT_BIAS: u64 = 127;
const MIN_EXPONENT: u8 = 1;
const MAX_EXPONENT: u8 = 232;
const ANCHOR_MIN: i32 = -(1 << 23);
const ANCHOR_MAX: i32 = (1 << 23) - 1;
const MAX_OFFSET_BITS: usize = 16;
const MAX_OFFSET: i32 = (1 << MAX_OFFSET_BITS) - 1;
const MIN_INDEX_BITS: usize = 3;

/// Contiguous DGF1 blocks and the corresponding Vulkan build metadata.
#[derive(Debug)]
pub(crate) struct DgfMesh {
    pub(crate) blocks: Vec<[u8; BLOCK_SIZE]>,
    pub(crate) num_triangles: u32,
    /// The sum of block vertex counts, including vertices shared between blocks.
    pub(crate) num_vertices: u32,
    /// The largest mesh-local index-triple ordinal. All geometry indices are zero.
    pub(crate) max_primitive_index: u32,
}

/// Input cannot be used as an exact DGF1 acceleration-structure source.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum EncodeError {
    NoTriangles,
    IncompleteTriangle { index_count: usize },
    TooManyTriangles { triangle_count: usize },
    IndexOutOfBounds { primitive: u32, vertex: u32 },
    NonFinitePosition { vertex: u32, axis: usize },
    Inexact { primitive: u32 },
}

impl fmt::Display for EncodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::NoTriangles => f.write_str("DGF1 requires at least one triangle"),
            Self::IncompleteTriangle { index_count } => {
                write!(
                    f,
                    "DGF1 index count {index_count} is not a multiple of three"
                )
            }
            Self::TooManyTriangles { triangle_count } => {
                write!(
                    f,
                    "DGF1 triangle count {triangle_count} exceeds the 29-bit primitive index limit"
                )
            }
            Self::IndexOutOfBounds { primitive, vertex } => {
                write!(
                    f,
                    "DGF1 triangle {primitive} references missing vertex {vertex}"
                )
            }
            Self::NonFinitePosition { vertex, axis } => {
                write!(
                    f,
                    "DGF1 vertex {vertex} has a non-finite coordinate on axis {axis}"
                )
            }
            Self::Inexact { primitive } => {
                write!(
                    f,
                    "DGF1 cannot encode triangle {primitive} with exact coordinates"
                )
            }
        }
    }
}

impl std::error::Error for EncodeError {}

/// Encodes indexed float3 positions with no rounding of any referenced coordinate.
///
/// Blocks use the coarsest legal power-of-two grid that represents all of their
/// vertices exactly. Different grids across blocks cannot introduce cracks
/// because the decoded coordinates are bit-identical to the source. All
/// triangles are non-opaque, so hardware ray queries expose them as candidates.
///
/// # Errors
///
/// Rejects invalid indices, non-finite positions, negative zero and other
/// coordinates DGF1 cannot represent exactly. Splits a block when the next
/// triangle doesn't fit, but errors if that triangle cannot fit on its own.
/// Unreferenced positions are not inspected.
pub(crate) fn encode_exact(
    positions: &[[f32; 3]],
    indices: &[u32],
) -> Result<DgfMesh, EncodeError> {
    let (triangles, remainder) = indices.as_chunks::<3>();
    if !remainder.is_empty() {
        return Err(EncodeError::IncompleteTriangle {
            index_count: indices.len(),
        });
    }
    let triangle_count = triangles.len();
    if triangle_count > MAX_PRIMITIVE_INDEX as usize + 1 {
        return Err(EncodeError::TooManyTriangles { triangle_count });
    }
    let mut triangles = triangles.iter();
    let first = triangles.next().ok_or(EncodeError::NoTriangles)?;
    let mut block = BlockDraft::new(0, positions, first)?;
    let mut blocks = Vec::with_capacity(triangle_count.div_ceil(MAX_TRIANGLES));
    let mut num_vertices = 0;
    for (primitive, triangle) in (1..).zip(triangles) {
        if !block.try_extend(primitive, positions, triangle)? {
            blocks.push(block.finish(positions));
            num_vertices += block.num_vertices as u32;
            block = BlockDraft::new(primitive, positions, triangle)?;
        }
    }
    blocks.push(block.finish(positions));
    num_vertices += block.num_vertices as u32;
    let num_triangles = triangle_count as u32;
    Ok(DgfMesh {
        blocks,
        num_triangles,
        num_vertices,
        max_primitive_index: num_triangles - 1,
    })
}

#[derive(Clone, Copy)]
struct GridBounds {
    min: [f32; 3],
    max: [f32; 3],
    exponent: u8,
}

impl GridBounds {
    fn empty() -> Self {
        Self {
            min: [f32::INFINITY; 3],
            max: [f32::NEG_INFINITY; 3],
            exponent: MAX_EXPONENT,
        }
    }

    fn include(
        &mut self,
        primitive: u32,
        vertex: u32,
        positions: &[[f32; 3]],
    ) -> Result<(), EncodeError> {
        let position = positions
            .get(vertex as usize)
            .ok_or(EncodeError::IndexOutOfBounds { primitive, vertex })?;
        for (axis, &coordinate) in position.iter().enumerate() {
            let bits = coordinate.to_bits();
            if bits != 0 {
                let biased_exponent = ((bits >> 23) & 0xff) as i32;
                if biased_exponent == 255 {
                    return Err(EncodeError::NonFinitePosition { vertex, axis });
                }
                if biased_exponent == 0 {
                    // Negative zero cannot round-trip its sign, and DGF1 bans subnormals.
                    return Err(EncodeError::Inexact { primitive });
                }
                // The least significant set bit bounds how coarse the exact grid can be.
                let significand = (bits & 0x7f_ffff) | (1 << 23);
                let exponent = biased_exponent - 23 + significand.trailing_zeros() as i32;
                if exponent < i32::from(MIN_EXPONENT) {
                    return Err(EncodeError::Inexact { primitive });
                }
                self.exponent = self.exponent.min(exponent as u8);
            }
            self.min[axis] = self.min[axis].min(coordinate);
            self.max[axis] = self.max[axis].max(coordinate);
        }
        Ok(())
    }

    fn quantization(self) -> Option<Quantization> {
        let inverse_scale = inverse_scale(self.exponent);
        let mut anchor = [0; 3];
        let mut widths = [0; 3];
        for axis in 0..3 {
            let min = f64::from(self.min[axis]) * inverse_scale;
            let max = f64::from(self.max[axis]) * inverse_scale;
            if min < f64::from(ANCHOR_MIN) || max > f64::from(ANCHOR_MAX + MAX_OFFSET) {
                return None;
            }
            // The largest legal anchor at or below the minimum minimizes the offsets.
            // A positive vertex can exceed the signed anchor range via its offset.
            anchor[axis] = (min as i32).min(ANCHOR_MAX);
            let span = max as i32 - anchor[axis];
            if span > MAX_OFFSET {
                return None;
            }
            widths[axis] = bits_needed(span as u32).max(1);
        }
        // A finer exact grid would only enlarge these integer bounds and spans.
        let width = widths.iter().sum::<usize>();
        let mut padding = width.next_multiple_of(4) - width;
        for axis_width in &mut widths {
            let extra = padding.min(MAX_OFFSET_BITS - *axis_width);
            *axis_width += extra;
            padding -= extra;
        }
        Some(Quantization {
            exponent: self.exponent,
            anchor,
            widths,
        })
    }
}

#[derive(Clone, Copy)]
struct Quantization {
    exponent: u8,
    anchor: [i32; 3],
    widths: [usize; 3],
}

impl Quantization {
    fn vertex_bits(self) -> usize {
        self.widths.iter().sum()
    }
}

struct BlockDraft {
    primitive_base: u32,
    bounds: GridBounds,
    quantization: Quantization,
    num_vertices: usize,
    num_triangles: usize,
    max_reused_index: u8,
    source_indices: [u32; MAX_VERTICES],
    local_indices: [[u8; 3]; MAX_TRIANGLES],
}

impl BlockDraft {
    fn new(primitive: u32, positions: &[[f32; 3]], triangle: &[u32]) -> Result<Self, EncodeError> {
        let mut bounds = GridBounds::empty();
        for &vertex in triangle {
            bounds.include(primitive, vertex, positions)?;
        }
        let quantization = bounds
            .quantization()
            .ok_or(EncodeError::Inexact { primitive })?;
        // The first triangle implicitly references 0, 1, 2. A repeated source
        // index here gets its own identical vertex slot to preserve the corner.
        let mut source_indices = [0; MAX_VERTICES];
        source_indices[..3].copy_from_slice(triangle);
        let mut local_indices = [[0; 3]; MAX_TRIANGLES];
        local_indices[0] = [0, 1, 2];
        Ok(Self {
            primitive_base: primitive,
            bounds,
            quantization,
            num_vertices: 3,
            num_triangles: 1,
            max_reused_index: (1 << MIN_INDEX_BITS) - 1,
            source_indices,
            local_indices,
        })
    }

    fn try_extend(
        &mut self,
        primitive: u32,
        positions: &[[f32; 3]],
        triangle: &[u32],
    ) -> Result<bool, EncodeError> {
        let num_triangles = self.num_triangles + 1;
        if num_triangles > MAX_TRIANGLES {
            return Ok(false);
        }
        let mut num_vertices = self.num_vertices;
        let mut bounds = self.bounds;
        let mut max_reused_index = self.max_reused_index;
        let mut local_indices = [0; 3];
        for (&vertex, local) in triangle.iter().zip(&mut local_indices) {
            if let Some(slot) = self.source_indices[..num_vertices]
                .iter()
                .position(|&x| x == vertex)
            {
                *local = slot as u8;
                max_reused_index = max_reused_index.max(*local);
            } else {
                let Some(slot) = self.source_indices.get_mut(num_vertices) else {
                    return Ok(false);
                };
                bounds.include(primitive, vertex, positions)?;
                // The new slots become visible to finish() only if the block fits.
                *slot = vertex;
                *local = num_vertices as u8;
                num_vertices += 1;
            }
        }
        let quantization = if num_vertices == self.num_vertices {
            self.quantization
        } else if let Some(quantization) = bounds.quantization() {
            quantization
        } else {
            return Ok(false);
        };
        let vertex_bits = (num_vertices * quantization.vertex_bits()).next_multiple_of(8);
        let reuse_bits =
            (3 * num_triangles - num_vertices) * bits_needed(u32::from(max_reused_index));
        if vertex_bits > MAX_FRONT_BITS
            || reuse_bits > MAX_REUSE_BITS
            || HEADER_BITS + vertex_bits + reuse_bits > first_index_start(num_triangles)
        {
            return Ok(false);
        }

        self.bounds = bounds;
        self.quantization = quantization;
        self.num_vertices = num_vertices;
        self.local_indices[self.num_triangles] = local_indices;
        self.num_triangles = num_triangles;
        self.max_reused_index = max_reused_index;
        Ok(true)
    }

    fn finish(&self, positions: &[[f32; 3]]) -> [u8; BLOCK_SIZE] {
        let mut block = [0; BLOCK_SIZE];
        let Quantization {
            exponent,
            anchor,
            widths,
        } = self.quantization;
        let index_bits = bits_needed(u32::from(self.max_reused_index));
        // Counts and component widths are stored minus one. The zero GeomID
        // metadata, palette mode, and OMM count select non-opaque geometry 0.
        let header = [
            MAGIC
                | ((index_bits as u32 - MIN_INDEX_BITS as u32) << 8)
                | ((self.num_vertices as u32 - 1) << 10)
                | ((self.num_triangles as u32 - 1) << 16),
            ((anchor[0] as u32) << 8) | u32::from(exponent),
            ((anchor[1] as u32) << 8) | ((widths[1] as u32 - 1) << 4) | (widths[0] as u32 - 1),
            ((anchor[2] as u32) << 8) | (widths[2] as u32 - 1),
            self.primitive_base,
        ];
        for (word, dst) in header.into_iter().zip(block.as_chunks_mut::<4>().0) {
            *dst = word.to_le_bytes();
        }

        let inverse_scale = inverse_scale(exponent);
        let mut bit = HEADER_BITS;
        let bits_per_vertex = self.quantization.vertex_bits();
        for &source in &self.source_indices[..self.num_vertices] {
            let position = positions[source as usize];
            let mut packed = 0;
            let mut shift = 0;
            for axis in 0..3 {
                let fixed = (f64::from(position[axis]) * inverse_scale) as i32;
                let offset = (fixed - anchor[axis]) as u64;
                packed |= offset << shift;
                shift += widths[axis];
            }
            write_bits(&mut block, bit, bits_per_vertex, packed);
            bit += bits_per_vertex;
        }

        let mut reuse_bit = bit.next_multiple_of(8);
        // The control bits are zero (RESTART). Starting just before them, the
        // is-first bits run backward; the reuse indices run forward.
        let mut first_bit = BLOCK_BITS - 2 * (self.num_triangles - 1);
        let mut next_vertex = 3;
        for &local in self.local_indices[1..self.num_triangles].iter().flatten() {
            first_bit -= 1;
            if local == next_vertex {
                block[first_bit / 8] |= 1 << (first_bit % 8);
                next_vertex += 1;
            } else {
                write_bits(&mut block, reuse_bit, index_bits, u64::from(local));
                reuse_bit += index_bits;
            }
        }
        block
    }
}

fn bits_needed(value: u32) -> usize {
    (u32::BITS - value.leading_zeros()) as usize
}

fn inverse_scale(exponent: u8) -> f64 {
    // Converting from f32 to f64 and multiplying by this exact power of two
    // preserves the integer bits, including when a rejected bound exceeds f32.
    f64::from_bits((1023 + EXPONENT_BIAS - u64::from(exponent)) << 52)
}

fn first_index_start(num_triangles: usize) -> usize {
    // Each RESTART beyond the first consumes two control and three is-first bits.
    BLOCK_BITS - 5 * (num_triangles - 1)
}

fn write_bits(block: &mut [u8; BLOCK_SIZE], start: usize, width: usize, value: u64) {
    let shift = start % 8;
    let mut bits = value << shift;
    for dst in &mut block[start / 8..(start + width).div_ceil(8)] {
        *dst |= bits as u8;
        bits >>= 8;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug)]
    struct DecodedBlock {
        primitive_base: u32,
        positions: Vec<[f32; 3]>,
        triangles: Vec<[usize; 3]>,
    }

    fn read_bits(block: &[u8; BLOCK_SIZE], start: usize, width: usize) -> u32 {
        let mut value = 0;
        for i in 0..width {
            let bit = start + i;
            value |= u32::from((block[bit / 8] >> (bit % 8)) & 1) << i;
        }
        value
    }

    // Independent reader for the emitted subset, including constraints that an
    // unchecked SDK decode assumes have already been validated by the encoder.
    fn decode_restart_block(block: &[u8; BLOCK_SIZE]) -> Option<DecodedBlock> {
        let [layout, x, y, z, primitive_base] =
            std::array::from_fn(|word| read_bits(block, word * 32, 32));
        let exponent = (x & 0xff) as u8;
        let num_vertices = ((layout >> 10) & 0x3f) as usize + 1;
        let num_triangles = ((layout >> 16) & 0x3f) as usize + 1;
        if layout & 0xff != MAGIC
            || layout >> 22 != 0
            || z & 0xf0 != 0
            || primitive_base >> 29 != 0
            || primitive_base + num_triangles as u32 - 1 > MAX_PRIMITIVE_INDEX
            || !(MIN_EXPONENT..=MAX_EXPONENT).contains(&exponent)
            || !(3..=3 * num_triangles).contains(&num_vertices)
        {
            return None;
        }

        let anchors = [x, y, z].map(|word| (word as i32) >> 8);
        let widths = [y & 0xf, (y >> 4) & 0xf, z & 0xf].map(|width| width as usize + 1);
        let bits_per_vertex = widths.iter().sum::<usize>();
        let vertex_end = HEADER_BITS + bits_per_vertex * num_vertices;
        let index_start = vertex_end.next_multiple_of(8);
        let index_width = ((layout >> 8) & 0x3) as usize + 3;
        let reuse_bits = index_width * (3 * num_triangles - num_vertices);
        let index_end = index_start + reuse_bits;
        let control_start = BLOCK_BITS - 2 * (num_triangles - 1);
        let first_start = control_start - 3 * (num_triangles - 1);
        if !bits_per_vertex.is_multiple_of(4)
            || index_start - HEADER_BITS > MAX_FRONT_BITS
            || reuse_bits > MAX_REUSE_BITS
            || index_end > first_start
            || read_bits(block, vertex_end, index_start - vertex_end) != 0
            || (control_start..BLOCK_BITS).any(|bit| read_bits(block, bit, 1) != 0)
        {
            return None;
        }

        let mut positions = Vec::with_capacity(num_vertices);
        let mut vertex_bit = HEADER_BITS;
        let scale = f32::from_bits(u32::from(exponent) << 23);
        for _ in 0..num_vertices {
            let position = std::array::from_fn(|axis| {
                let offset = read_bits(block, vertex_bit, widths[axis]) as i32;
                vertex_bit += widths[axis];
                (anchors[axis] + offset) as f32 * scale
            });
            if position.iter().any(|v| !v.is_normal() && *v != 0.0) {
                return None;
            }
            positions.push(position);
        }

        let mut triangles = Vec::with_capacity(num_triangles);
        triangles.push([0, 1, 2]);
        let mut next_vertex = 3;
        let mut reuse_bit = index_start;
        let mut first_bits = (first_start..control_start).rev();
        for _ in 1..num_triangles {
            let mut triangle = [0; 3];
            for corner in &mut triangle {
                if read_bits(block, first_bits.next()?, 1) == 1 {
                    if next_vertex >= num_vertices {
                        return None;
                    }
                    *corner = next_vertex;
                    next_vertex += 1;
                } else {
                    if reuse_bit >= index_end {
                        return None;
                    }
                    *corner = read_bits(block, reuse_bit, index_width) as usize;
                    reuse_bit += index_width;
                    if *corner >= next_vertex {
                        return None;
                    }
                }
            }
            triangles.push(triangle);
        }
        if next_vertex != num_vertices || reuse_bit != index_end {
            return None;
        }
        Some(DecodedBlock {
            primitive_base,
            positions,
            triangles,
        })
    }

    fn assert_round_trip(
        mesh: &DgfMesh,
        positions: &[[f32; 3]],
        indices: &[u32],
        triangles_per_block: &[usize],
    ) {
        assert_eq!(mesh.blocks.len(), triangles_per_block.len());
        let mut primitive = 0;
        let mut num_vertices = 0;
        for (block, &num_triangles) in mesh.blocks.iter().zip(triangles_per_block) {
            let block = decode_restart_block(block).expect("legal DGF1 RESTART block");
            assert_eq!(block.primitive_base as usize, primitive);
            assert_eq!(block.triangles.len(), num_triangles);
            num_vertices += block.positions.len();
            for triangle in block.triangles {
                for (local, &source) in triangle.into_iter().zip(&indices[3 * primitive..][..3]) {
                    assert_eq!(
                        block.positions[local].map(f32::to_bits),
                        positions[source as usize].map(f32::to_bits),
                        "primitive {primitive}, source vertex {source}"
                    );
                }
                primitive += 1;
            }
        }
        assert_eq!(mesh.num_triangles as usize, indices.len() / 3);
        assert_eq!(mesh.num_triangles as usize, primitive);
        assert_eq!(mesh.num_vertices as usize, num_vertices);
        assert_eq!(mesh.max_primitive_index as usize, primitive - 1);
    }

    #[test]
    fn matches_the_dgf1_layout_for_signed_anchor_and_mixed_first_indices() {
        let positions = [
            [-1.0, 0.0, 0.0],
            [0.0; 3],
            [-1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ];
        let indices = [0, 1, 2, 2, 1, 3];
        let mesh = encode_exact(&positions, &indices).expect("integer square");
        // Manually specified from the layout: anchor [-1, 0, 0], exponent 127,
        // widths [2, 1, 1]. Offsets are nibbles [0, 1, 4, 5]. The second
        // triangle is RESTART with first flags [0, 0, 1] and reuse entries [2, 1].
        let mut expected = [0; 128];
        expected[..23].copy_from_slice(&[
            0x06, 0x0c, 0x01, 0x00, 0x7f, 0xff, 0xff, 0xff, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x54, 0x0a,
        ]);
        expected[127] = 0x08;
        assert_eq!(mesh.blocks, [expected]);
        assert_round_trip(&mesh, &positions, &indices, &[2]);
    }

    #[test]
    fn reuse_limit_preserves_primitive_ordinals_and_corner_order() {
        let positions = [[0.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let indices: Vec<u32> = (0..70)
            .flat_map(|i| if i % 2 == 0 { [0, 1, 2] } else { [2, 1, 0] })
            .collect();
        let mesh = encode_exact(&positions, &indices).expect("repeated integer triangle");
        // Twenty-one restarts reuse 189 bits; a twenty-second would use 198.
        assert_round_trip(&mesh, &positions, &indices, &[22, 22, 22, 4]);
        assert_eq!(mesh.num_vertices, 12);
    }

    #[test]
    fn budgets_front_buffer_and_topology_separately() {
        let corners = [[0.0; 3], [65535.0, 65535.0, 0.0], [65535.0, 0.0, 65535.0]];
        let positions: Vec<_> = corners.into_iter().cycle().take(21).collect();
        let indices: Vec<u32> = (0..21).collect();
        let mesh = encode_exact(&positions, &indices).expect("sixteen-bit offset coordinates");
        // Fifteen 48-bit vertices fit in the front buffer, eighteen do not.
        assert_round_trip(&mesh, &positions, &indices, &[5, 2]);

        let indices: Vec<u32> = (0..15)
            .chain([0, 1, 2].into_iter().cycle().take(30))
            .collect();
        let mesh = encode_exact(&positions, &indices).expect("mixed new and reused vertices");
        // At triangle 14, 720 front + 81 reuse + 65 backward bits collide,
        // despite the front and reuse buffers independently being within budget.
        assert_round_trip(&mesh, &positions, &indices, &[13, 2]);
    }

    #[test]
    fn supports_sixty_four_vertices_and_six_bit_reuse_indices() {
        let corners = [[0.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let positions: Vec<_> = corners.into_iter().cycle().take(67).collect();
        let indices: Vec<u32> = (0..63).chain([63, 61, 62, 64, 65, 66]).collect();
        let mesh = encode_exact(&positions, &indices).expect("integer corner coordinates");
        assert_round_trip(&mesh, &positions, &indices, &[22, 1]);
        assert_eq!(mesh.num_vertices, 67);
        assert_eq!(read_bits(&mesh.blocks[0], 8, 2), 3);
    }

    #[test]
    fn splits_incompatible_exact_grids_but_rejects_indivisible_triangles() {
        let positions = [
            [1048576.0, 0.0, 0.0],
            [1048577.0, 0.0, 0.0],
            [1048576.0, 1.0, 0.0],
            [0.0; 3],
            [0.0625, 0.0, 0.0],
            [0.0, 0.0625, 0.0],
        ];
        let indices = [0, 1, 2, 3, 4, 5];
        let mesh = encode_exact(&positions, &indices).expect("two separately encodable grids");
        assert_round_trip(&mesh, &positions, &indices, &[1, 1]);
        assert_eq!(mesh.blocks[0][4], 127);
        assert_eq!(mesh.blocks[1][4], 123);

        let positions = [[0.0; 3], [65536.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        assert_eq!(
            encode_exact(&positions, &[0, 1, 0, 0, 1, 2]).unwrap_err(),
            EncodeError::Inexact { primitive: 1 }
        );
    }

    #[test]
    fn encodes_extreme_legal_anchors_and_exponents() {
        let indices = [0, 1, 2];
        let positions = [
            [8454141.0, -8388608.0, 0.0],
            [8454142.0, -8388608.0, 1.0],
            [8454140.0, -8388607.0, 0.0],
        ];
        let mesh =
            encode_exact(&positions, &indices).expect("positive fixed points beyond anchor range");
        assert_round_trip(&mesh, &positions, &indices, &[1]);
        assert_eq!(read_bits(&mesh.blocks[0], 40, 24), 0x7f_ffff);
        assert_eq!(read_bits(&mesh.blocks[0], 72, 24), 0x80_0000);

        for (unit, bound, exponent) in [
            (f32::MIN_POSITIVE, f32::MIN_POSITIVE, 1),
            (f32::from_bits(232 << 23), f32::from_bits(0x7f7f_fffe), 232),
        ] {
            let positions = [[bound, 0.0, 0.0], [bound, unit, 0.0], [bound, 0.0, unit]];
            let mesh =
                encode_exact(&positions, &indices).expect("legal endpoint of exponent range");
            assert_round_trip(&mesh, &positions, &indices, &[1]);
            assert_eq!(mesh.blocks[0][4], exponent);
        }
    }

    #[test]
    fn preserves_repeated_corners_and_rejects_invalid_inputs() {
        let positions = [[0.0; 3], [1.0, 0.0, 0.0], [f32::NAN; 3]];
        let indices = [0, 0, 1, 1, 1, 0];
        let mesh =
            encode_exact(&positions, &indices).expect("unreferenced positions do not matter");
        assert_round_trip(&mesh, &positions, &indices, &[2]);
        assert_eq!(mesh.num_vertices, 3);
        assert_eq!(
            encode_exact(&positions, &[]).unwrap_err(),
            EncodeError::NoTriangles
        );
        assert_eq!(
            encode_exact(&positions, &[0, 1]).unwrap_err(),
            EncodeError::IncompleteTriangle { index_count: 2 }
        );
        assert_eq!(
            encode_exact(&positions, &[0, 1, 3]).unwrap_err(),
            EncodeError::IndexOutOfBounds {
                primitive: 0,
                vertex: 3
            }
        );
        assert_eq!(
            encode_exact(&positions, &[0, 1, 2]).unwrap_err(),
            EncodeError::NonFinitePosition { vertex: 2, axis: 0 }
        );
        for invalid in [
            -0.0,
            f32::from_bits(1),
            f32::from_bits(0x0080_0001),
            f32::MAX,
            0.1,
        ] {
            let positions = [[0.0; 3], [1.0, 0.0, 0.0], [0.0, invalid, 0.0]];
            assert_eq!(
                encode_exact(&positions, &[0, 1, 2]).unwrap_err(),
                EncodeError::Inexact { primitive: 0 },
                "coordinate {invalid:?}"
            );
        }
    }

    #[test]
    fn subset_decoder_rejects_bad_padding_and_topology() {
        let positions = [[0.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let mesh = encode_exact(&positions, &[0, 1, 2, 2, 1, 0]).expect("valid triangles");
        for (byte, mask) in [
            (21, 0x10),  // Nonzero vertex padding.
            (22, 0x07),  // Reuse index 7 before its first use.
            (127, 0x80), // A non-RESTART control.
            (127, 0x20), // Too many first-use references.
        ] {
            let mut block = mesh.blocks[0];
            block[byte] |= mask;
            assert!(
                decode_restart_block(&block).is_none(),
                "byte {byte}, mask {mask}"
            );
        }
    }
}
