//! Whole-primitive index LODs for static triangle meshes.
//!
//! Each level indexes the original vertex and attribute buffers; vertices are
//! never synthesized. Generate levels separately for each material primitive.
//! Meshoptimizer locks topological borders, but independent primitives with
//! mismatched boundary topology are not guaranteed to stitch. Simplification
//! can change triangles, their order, and primitive IDs: visibility-buffer
//! resolve and primitive extensions must use the selected level's index buffer
//! and cannot treat its triangle IDs as IDs in the original primitive.
//!
//! The reported error is meshoptimizer's approximate position-only quadric
//! metric, not a Hausdorff bound or a guarantee of pixel accuracy. Attribute
//! values are retained at surviving vertices, but their interpolation and
//! appearance are not error-controlled. Hard seams should be represented by
//! split vertices. Skinning, morphs, and displacement invalidate these static
//! geometry estimates.

use std::fmt;

use meshopt::{SimplifyOptions, VertexDataAdapter, simplify, typed_to_bytes};

/// Parameters for repeatedly reducing the previous full-primitive level.
#[derive(Clone, Copy, Debug)]
pub struct LodOptions {
    /// Maximum number of levels including the original, at least one.
    pub max_levels: usize,
    /// Desired fraction of triangles retained at each step, strictly between 0 and 1.
    pub triangle_ratio: f32,
    /// Maximum meshoptimizer error per step in object-space units, finite and nonnegative.
    pub max_step_error: f32,
}

/// One drawable index buffer, ordered from the original to progressively coarser levels.
#[derive(Clone, Debug, PartialEq)]
pub struct LodLevel {
    /// Triangle-list indices into the original vertex and attribute buffers.
    pub indices: Vec<u32>,
    /// Sum of the per-step meshoptimizer errors, in object-space units.
    /// This is an accumulated estimate, not a strict geometric error bound.
    pub error: f64,
}

/// Invalid input or an unusable simplifier result.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LodError {
    /// Options must satisfy the documented ranges.
    InvalidOptions,
    /// The input is not a triangle list.
    IncompleteTriangle,
    /// The input exceeds the simplifier's 32-bit indexing limits.
    TooLarge,
    /// A source vertex index is out of bounds.
    IndexOutOfBounds { offset: usize, vertex: u32 },
    /// All positions, including unreferenced positions, must be finite.
    NonFinitePosition { vertex: usize },
    /// Position extents overflow the simplifier's floating-point arithmetic.
    PositionRangeTooLarge,
    /// Meshoptimizer returned a non-finite or negative error estimate.
    InvalidSimplifierError,
}

impl fmt::Display for LodError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidOptions => write!(f, "invalid LOD options"),
            Self::IncompleteTriangle => write!(f, "indices are not a triangle list"),
            Self::TooLarge => write!(f, "geometry exceeds 32-bit simplifier limits"),
            Self::IndexOutOfBounds { offset, vertex } => {
                write!(
                    f,
                    "vertex {vertex} at index offset {offset} is out of bounds"
                )
            }
            Self::NonFinitePosition { vertex } => write!(f, "vertex {vertex} is not finite"),
            Self::PositionRangeTooLarge => write!(f, "position extent exceeds f32 range"),
            Self::InvalidSimplifierError => write!(f, "simplifier returned an invalid error"),
        }
    }
}

impl std::error::Error for LodError {}

/// Builds successive simplifications of a complete material primitive.
///
/// The first level is an exact copy of `indices`; empty input produces one
/// empty level. Topological border vertices are locked. The requested ratio and
/// level count are goals: topology, borders, or the error limit can stop
/// simplification early. Empty and unchanged simplifier results are discarded.
/// Winding is retained by meshoptimizer's topology-preserving edge collapses.
pub fn build_lods(
    positions: &[[f32; 3]],
    indices: &[u32],
    options: LodOptions,
) -> Result<Vec<LodLevel>, LodError> {
    if options.max_levels == 0
        || !options.triangle_ratio.is_finite()
        || !(0.0..1.0).contains(&options.triangle_ratio)
        || options.triangle_ratio == 0.0
        || !options.max_step_error.is_finite()
        || options.max_step_error < 0.0
    {
        return Err(LodError::InvalidOptions);
    }
    if !indices.len().is_multiple_of(3) {
        return Err(LodError::IncompleteTriangle);
    }
    // The native implementation stores vertex IDs and adjacency offsets in u32.
    if positions.len() >= u32::MAX as usize || indices.len() >= u32::MAX as usize {
        return Err(LodError::TooLarge);
    }
    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    for (vertex, position) in positions.iter().enumerate() {
        for axis in 0..3 {
            let value = position[axis];
            if !value.is_finite() {
                return Err(LodError::NonFinitePosition { vertex });
            }
            min[axis] = min[axis].min(value);
            max[axis] = max[axis].max(value);
        }
    }
    if !positions.is_empty() && (0..3).any(|axis| !(max[axis] - min[axis]).is_finite()) {
        return Err(LodError::PositionRangeTooLarge);
    }
    for (offset, &vertex) in indices.iter().enumerate() {
        if vertex as usize >= positions.len() {
            return Err(LodError::IndexOutOfBounds { offset, vertex });
        }
    }
    let mut levels = vec![LodLevel {
        indices: indices.to_vec(),
        error: 0.0,
    }];
    if indices.is_empty() || options.max_levels == 1 {
        return Ok(levels);
    }
    // The adapter's stride and offset are fixed by the [f32; 3] representation.
    let vertices = VertexDataAdapter::new(typed_to_bytes(positions), 12, 0)
        .expect("packed three-float positions have a valid stride");
    while levels.len() < options.max_levels {
        let previous = levels.last().expect("the original level always exists");
        let triangles = previous.indices.len() / 3;
        if triangles <= 1 {
            break;
        }
        let target = ((triangles as f64 * f64::from(options.triangle_ratio)) as usize).max(1) * 3;
        let mut step_error = 0.0;
        let simplified = simplify(
            &previous.indices,
            &vertices,
            target,
            options.max_step_error,
            SimplifyOptions::LockBorder | SimplifyOptions::ErrorAbsolute,
            Some(&mut step_error),
        );
        if !step_error.is_finite() || step_error < 0.0 {
            return Err(LodError::InvalidSimplifierError);
        }
        if simplified.is_empty() || simplified.len() >= previous.indices.len() {
            break;
        }
        // Round upward, rather than understating the sum due to f64 rounding.
        let sum = previous.error + f64::from(step_error);
        let error = if sum > 0.0 {
            f64::from_bits(sum.to_bits() + 1)
        } else {
            0.0
        };
        levels.push(LodLevel {
            indices: simplified,
            error,
        });
    }
    Ok(levels)
}

/// Selects the coarsest level within an approximate screen-space error budget.
///
/// `pixels_per_object_unit` is a caller-supplied conservative projection scale
/// for the entire object, including its transform and nearest relevant depth.
/// A perspective bound must account for off-axis projection and near-plane
/// intersections; if no finite bound is known, pass infinity to select level 0.
/// For orthographic projection the scale is depth independent. The estimate is
/// the accumulated quadric metric times this scale, not a pixel-error bound.
/// Invalid inputs or malformed/non-monotonic level errors select level 0; an
/// empty slice returns `None`. Equal-to-budget estimates are accepted.
pub fn select_lod(
    levels: &[LodLevel],
    pixels_per_object_unit: f64,
    max_error_pixels: f64,
) -> Option<usize> {
    if levels.is_empty() {
        return None;
    }
    if !pixels_per_object_unit.is_finite()
        || pixels_per_object_unit < 0.0
        || !max_error_pixels.is_finite()
        || max_error_pixels < 0.0
        || levels[0].error != 0.0
    {
        return Some(0);
    }
    let mut last_error = 0.0;
    let mut selected = 0;
    for (index, level) in levels.iter().enumerate().skip(1) {
        if !level.error.is_finite() || level.error < last_error {
            return Some(0);
        }
        last_error = level.error;
        // Division avoids a product overflow and is conservative at zero scale.
        if pixels_per_object_unit == 0.0 || level.error <= max_error_pixels / pixels_per_object_unit
        {
            selected = index;
        }
    }
    Some(selected)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    fn options() -> LodOptions {
        LodOptions {
            max_levels: 5,
            triangle_ratio: 0.5,
            max_step_error: 10.0,
        }
    }

    fn grid(n: u32) -> (Vec<[f32; 3]>, Vec<u32>) {
        let positions = (0..=n)
            .flat_map(|y| {
                (0..=n).map(move |x| {
                    [
                        x as f32,
                        y as f32,
                        (x as f32 * 0.2).sin() * (y as f32 * 0.2).sin(),
                    ]
                })
            })
            .collect();
        let mut indices = Vec::new();
        for y in 0..n {
            for x in 0..n {
                let a = y * (n + 1) + x;
                indices.extend_from_slice(&[a, a + 1, a + n + 1, a + 1, a + n + 2, a + n + 1]);
            }
        }
        (positions, indices)
    }

    fn boundary(indices: &[u32]) -> BTreeSet<(u32, u32)> {
        let mut edges = std::collections::BTreeMap::new();
        for tri in indices.chunks_exact(3) {
            for (a, b) in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])] {
                *edges.entry((a.min(b), a.max(b))).or_insert(0usize) += 1;
            }
        }
        edges
            .into_iter()
            .filter_map(|(edge, count)| (count == 1).then_some(edge))
            .collect()
    }

    #[test]
    fn reduces_full_primitive_and_locks_border() {
        let (positions, indices) = grid(16);
        let levels = build_lods(&positions, &indices, options()).unwrap();
        assert!(levels.len() > 1);
        assert_eq!(levels[0].indices, indices);
        let border = boundary(&indices);
        for pair in levels.windows(2) {
            assert!(pair[1].indices.len() < pair[0].indices.len());
            assert!(pair[1].error >= pair[0].error);
            assert_eq!(boundary(&pair[1].indices), border);
            assert!(
                pair[1]
                    .indices
                    .iter()
                    .all(|&i| (i as usize) < positions.len())
            );
            // The grid faces consistently point toward positive Z.
            for t in pair[1].indices.chunks_exact(3) {
                let [a, b, c] = [
                    positions[t[0] as usize],
                    positions[t[1] as usize],
                    positions[t[2] as usize],
                ];
                assert!((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]) > 0.0);
            }
        }
    }

    #[test]
    fn validates_input_and_handles_empty() {
        assert_eq!(build_lods(&[], &[], options()).unwrap().len(), 1);
        assert_eq!(
            build_lods(&[], &[0], options()),
            Err(LodError::IncompleteTriangle)
        );
        assert!(matches!(
            build_lods(&[], &[0, 0, 0], options()),
            Err(LodError::IndexOutOfBounds { .. })
        ));
        assert!(matches!(
            build_lods(&[[f32::NAN, 0.0, 0.0]], &[], options()),
            Err(LodError::NonFinitePosition { .. })
        ));
        assert_eq!(
            build_lods(
                &[],
                &[],
                LodOptions {
                    triangle_ratio: 1.0,
                    ..options()
                }
            ),
            Err(LodError::InvalidOptions)
        );
        assert_eq!(
            build_lods(
                &[],
                &[],
                LodOptions {
                    max_step_error: f32::INFINITY,
                    ..options()
                }
            ),
            Err(LodError::InvalidOptions)
        );
        assert_eq!(
            build_lods(
                &[[-f32::MAX, 0.0, 0.0], [f32::MAX, 0.0, 0.0]],
                &[],
                options()
            ),
            Err(LodError::PositionRangeTooLarge)
        );
    }

    #[test]
    fn selects_coarsest_eligible_and_fails_safe() {
        let mut levels = vec![
            LodLevel {
                indices: vec![],
                error: 0.0,
            },
            LodLevel {
                indices: vec![],
                error: 0.25,
            },
            LodLevel {
                indices: vec![],
                error: 1.0,
            },
        ];
        assert_eq!(select_lod(&[], 1.0, 1.0), None);
        assert_eq!(select_lod(&levels, 4.0, 1.0), Some(1));
        assert_eq!(select_lod(&levels, 1.0, 1.0), Some(2));
        assert_eq!(select_lod(&levels, 0.0, 0.0), Some(2));
        assert_eq!(select_lod(&levels, f64::INFINITY, 1.0), Some(0));
        assert_eq!(select_lod(&levels, 1.0, f64::NAN), Some(0));
        levels[2].error = 0.1;
        assert_eq!(select_lod(&levels, 1.0, 1.0), Some(0));
    }
}
