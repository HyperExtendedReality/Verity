//! Order-preserving meshlets for indexed triangle lists.
//!
//! A meshlet contains a contiguous range of the source triangles, a table of
//! source vertex indices, and byte-sized indices into that table. Neither
//! triangle order, winding, duplicate triangles nor degenerates are changed.
//! Build each material primitive separately; `first_triangle` is relative to
//! the input slice, not a scene-wide index buffer.
//!
//! Construction uses a greedy scan, analogous to meshoptimizer's scan builder,
//! rather than its spatially optimized offline builder. Consequently meshlets
//! can be spatially loose if the input triangles are not already coherent.
//! Bounds describe static object-space positions only: they are not valid after
//! skinning, morphing, displacement, or any other vertex deformation unless the
//! caller supplies conservative bounds for the deformed geometry.
//! [`lod`] provides whole-primitive simplification; [`residency`] provides a
//! CPU page cache; and [`culling`] provides a CPU reverse-Z occlusion reference.
//! These components are not a GPU-driven streaming or continuous-LOD renderer.

pub mod culling;
pub mod lod;
pub mod residency;
pub mod virtual_geometry;

use std::fmt;

/// Maximum number of unique source vertices in one meshlet.
pub const MAX_VERTICES: usize = 64;
/// Maximum number of source triangles in one meshlet.
pub const MAX_TRIANGLES: usize = 124;

/// Object-space axis-aligned bounds. All components must be finite and ordered.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Bounds {
    pub min: [f32; 3],
    pub max: [f32; 3],
}

impl Bounds {
    /// Returns true only if the entire box is strictly outside an inward-facing
    /// plane `ax + by + cz + d >= 0` in the same coordinate space.
    ///
    /// Non-finite or inverted bounds and non-finite planes never reject. The
    /// calculation uses f64 and an error margin, keeping boundary cases. The
    /// caller must ensure the supplied plane itself conservatively represents
    /// the intended frustum; rounding during plane extraction or a transform is
    /// not accounted for here.
    pub fn outside_plane(&self, plane: [f32; 4]) -> bool {
        if plane.iter().any(|value| !value.is_finite())
            || (0..3).any(|axis| {
                !self.min[axis].is_finite()
                    || !self.max[axis].is_finite()
                    || self.min[axis] > self.max[axis]
            })
        {
            return false;
        }

        let mut distance = f64::from(plane[3]);
        let mut magnitude = distance.abs();
        for (axis, &normal) in plane[..3].iter().enumerate() {
            let coordinate = if normal >= 0.0 {
                self.max[axis]
            } else {
                self.min[axis]
            };
            let term = f64::from(normal) * f64::from(coordinate);
            distance += term;
            magnitude += term.abs();
        }
        distance < -magnitude * (8.0 * f64::EPSILON)
    }

    /// Returns true if any inward-facing plane strictly excludes the box.
    /// Empty plane sets and invalid planes cannot reject a box by themselves.
    pub fn outside_frustum(&self, planes: &[[f32; 4]]) -> bool {
        planes.iter().any(|&plane| self.outside_plane(plane))
    }
}

/// A contiguous subset of an indexed triangle list.
#[derive(Clone, Debug, PartialEq)]
pub struct Meshlet {
    /// Index of the first triangle in the input index slice.
    pub first_triangle: u32,
    /// Number of consecutive source triangles.
    pub triangle_count: u32,
    /// Bounds of all referenced vertices, including degenerate triangles.
    pub bounds: Bounds,
    /// Source vertex indices, in order of first appearance.
    pub vertices: Vec<u32>,
    /// Three local indices per triangle, in original triangle and winding order.
    pub local_indices: Vec<u8>,
}

/// Invalid indexed geometry cannot safely produce meshlet bounds.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MeshletError {
    /// The index buffer is not a triangle list.
    IncompleteTriangle { index_count: usize },
    /// The triangle count cannot be represented by the output range type.
    TooManyTriangles,
    /// A source index does not address a position.
    IndexOutOfBounds { index_offset: usize, vertex: u32 },
    /// A referenced position has a NaN or infinite component.
    NonFinitePosition { vertex: u32 },
}

impl fmt::Display for MeshletError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::IncompleteTriangle { index_count } => {
                write!(f, "index count {index_count} is not divisible by three")
            }
            Self::TooManyTriangles => write!(f, "triangle count exceeds u32"),
            Self::IndexOutOfBounds {
                index_offset,
                vertex,
            } => {
                write!(
                    f,
                    "vertex index {vertex} at offset {index_offset} is out of bounds"
                )
            }
            Self::NonFinitePosition { vertex } => {
                write!(f, "vertex {vertex} has a non-finite position")
            }
        }
    }
}

impl std::error::Error for MeshletError {}

/// Partition indexed triangles without reordering them or modifying the source.
///
/// The supplied positions and indices share the same vertex-index origin.
/// Validate and build one material primitive at a time to preserve material
/// boundaries. Unreferenced positions are ignored. On invalid input an error is
/// returned; callers must not use partial output to cull or draw geometry.
/// Empty input yields no meshlets.
pub fn build_meshlets(
    positions: &[[f32; 3]],
    indices: &[u32],
) -> Result<Vec<Meshlet>, MeshletError> {
    if !indices.len().is_multiple_of(3) {
        return Err(MeshletError::IncompleteTriangle {
            index_count: indices.len(),
        });
    }
    if indices.len() / 3 > u32::MAX as usize {
        return Err(MeshletError::TooManyTriangles);
    }

    let mut meshlets = Vec::new();
    let mut current: Option<Meshlet> = None;
    for (triangle, corners) in indices.as_chunks::<3>().0.iter().enumerate() {
        for (corner, &vertex) in corners.iter().enumerate() {
            let position =
                positions
                    .get(vertex as usize)
                    .ok_or(MeshletError::IndexOutOfBounds {
                        index_offset: triangle * 3 + corner,
                        vertex,
                    })?;
            if position.iter().any(|component| !component.is_finite()) {
                return Err(MeshletError::NonFinitePosition { vertex });
            }
        }

        if let Some(meshlet) = &current {
            let additional = corners
                .iter()
                .enumerate()
                .filter(|(i, vertex)| {
                    !meshlet.vertices.contains(vertex) && !corners[..*i].contains(vertex)
                })
                .count();
            if meshlet.triangle_count as usize == MAX_TRIANGLES
                || meshlet.vertices.len() + additional > MAX_VERTICES
            {
                meshlets.push(current.take().unwrap());
            }
        }

        let meshlet = current.get_or_insert_with(|| Meshlet {
            first_triangle: triangle as u32,
            triangle_count: 0,
            bounds: Bounds {
                min: [f32::INFINITY; 3],
                max: [f32::NEG_INFINITY; 3],
            },
            vertices: Vec::new(),
            local_indices: Vec::new(),
        });
        for &vertex in corners {
            let local = match meshlet.vertices.iter().position(|&v| v == vertex) {
                Some(local) => local,
                None => {
                    let local = meshlet.vertices.len();
                    meshlet.vertices.push(vertex);
                    let position = positions[vertex as usize];
                    for (axis, &coordinate) in position.iter().enumerate() {
                        meshlet.bounds.min[axis] = meshlet.bounds.min[axis].min(coordinate);
                        meshlet.bounds.max[axis] = meshlet.bounds.max[axis].max(coordinate);
                    }
                    local
                }
            };
            meshlet.local_indices.push(local as u8);
        }
        meshlet.triangle_count += 1;
    }
    if let Some(meshlet) = current {
        meshlets.push(meshlet);
    }
    Ok(meshlets)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preserves_triangles_winding_duplicates_and_degenerates() {
        let positions = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let indices = [2, 1, 0, 2, 2, 1, 2, 1, 0];
        let meshlets = build_meshlets(&positions, &indices).unwrap();
        assert_eq!(meshlets.len(), 1);
        let meshlet = &meshlets[0];
        assert_eq!(meshlet.first_triangle, 0);
        assert_eq!(meshlet.triangle_count, 3);
        assert_eq!(
            meshlet.bounds,
            Bounds {
                min: [0.0; 3],
                max: [1.0, 1.0, 0.0]
            }
        );
        let restored: Vec<_> = meshlet
            .local_indices
            .iter()
            .map(|&i| meshlet.vertices[i as usize])
            .collect();
        assert_eq!(restored, indices);
    }

    #[test]
    fn partitions_at_both_limits_and_preserves_ranges() {
        let positions: Vec<_> = (0..300).map(|i| [i as f32, 0.0, 0.0]).collect();
        let indices: Vec<u32> = (0..300).collect();
        let meshlets = build_meshlets(&positions, &indices).unwrap();
        let mut first = 0;
        for meshlet in &meshlets {
            assert_eq!(meshlet.first_triangle, first);
            assert!(meshlet.vertices.len() <= MAX_VERTICES);
            assert!(meshlet.triangle_count as usize <= MAX_TRIANGLES);
            let original =
                &indices[first as usize * 3..(first + meshlet.triangle_count) as usize * 3];
            let restored: Vec<_> = meshlet
                .local_indices
                .iter()
                .map(|&i| meshlet.vertices[i as usize])
                .collect();
            assert_eq!(restored, original);
            first += meshlet.triangle_count;
        }
        assert_eq!(first, 100);
        let repeated = vec![0; (MAX_TRIANGLES + 1) * 3];
        let split = build_meshlets(&positions, &repeated).unwrap();
        assert_eq!(split.len(), 2);
        assert_eq!(split[0].triangle_count as usize, MAX_TRIANGLES);
        assert_eq!(split[1].first_triangle as usize, MAX_TRIANGLES);
        assert_eq!(split[1].triangle_count, 1);
    }

    #[test]
    fn duplicate_vertices_at_capacity_do_not_force_a_split() {
        let positions = vec![[0.0; 3]; 64];
        let mut indices: Vec<u32> = (0..63).collect();
        indices.extend([63, 63, 63]);
        let meshlets = build_meshlets(&positions, &indices).unwrap();
        assert_eq!(meshlets.len(), 1);
        assert_eq!(meshlets[0].vertices.len(), 64);
    }

    #[test]
    fn rejects_invalid_input_and_ignores_unused_positions() {
        assert!(build_meshlets(&[], &[]).unwrap().is_empty());
        assert!(matches!(
            build_meshlets(&[], &[0]),
            Err(MeshletError::IncompleteTriangle { .. })
        ));
        assert!(matches!(
            build_meshlets(&[], &[0, 0, 0]),
            Err(MeshletError::IndexOutOfBounds {
                index_offset: 0,
                vertex: 0
            })
        ));
        let positions = [[0.0; 3], [f32::NAN, 0.0, 0.0], [f32::INFINITY, 0.0, 0.0]];
        assert!(build_meshlets(&positions, &[0, 0, 0]).is_ok());
        assert!(matches!(
            build_meshlets(&positions, &[0, 1, 0]),
            Err(MeshletError::NonFinitePosition { vertex: 1 })
        ));
        assert!(matches!(
            build_meshlets(&positions, &[2, 0, 0]),
            Err(MeshletError::NonFinitePosition { vertex: 2 })
        ));
    }

    #[test]
    fn conservative_plane_culling_keeps_boundaries_and_invalid_input() {
        let bounds = Bounds {
            min: [-1.0; 3],
            max: [1.0; 3],
        };
        assert!(!bounds.outside_plane([1.0, 0.0, 0.0, 1.0]));
        assert!(bounds.outside_plane([1.0, 0.0, 0.0, -2.0]));
        assert!(bounds.outside_plane([-1.0, 0.0, 0.0, -2.0]));
        assert!(!bounds.outside_plane([f32::NAN, 0.0, 0.0, -2.0]));
        assert!(!bounds.outside_frustum(&[]));
        assert!(bounds.outside_frustum(&[[0.0, 1.0, 0.0, 2.0], [1.0, 0.0, 0.0, -2.0]]));
        assert!(
            !Bounds {
                min: [1.0; 3],
                max: [-1.0; 3]
            }
            .outside_plane([1.0, 0.0, 0.0, -2.0])
        );
        assert!(
            !Bounds {
                min: [f32::NAN; 3],
                max: [1.0; 3]
            }
            .outside_plane([1.0, 0.0, 0.0, -2.0])
        );
    }

    #[test]
    fn extreme_coordinates_and_cancellation_are_conservative() {
        let extreme = Bounds {
            min: [f32::MAX; 3],
            max: [f32::MAX; 3],
        };
        assert!(!extreme.outside_plane([f32::MAX, -f32::MAX, 0.0, 0.0]));
        assert!(extreme.outside_plane([-f32::MAX, 0.0, 0.0, 0.0]));
    }
}
