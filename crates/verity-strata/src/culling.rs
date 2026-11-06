//! CPU reference reverse-Z depth pyramid and projected-bounds occlusion test.
//!
//! Reverse Z maps the far/background depth to 0 and the near plane to 1.
//! Pyramid texels contain the **minimum** depth of their source texels: a
//! single background sample therefore prevents rejection over its footprint.
//! These primitives do not acquire depth data or integrate with a renderer.

use std::fmt;

/// Invalid input to [`DepthPyramid::new`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DepthPyramidError {
    /// Both image dimensions must be nonzero.
    EmptyImage,
    /// The dimensions cannot be represented by the platform's `usize`.
    ImageTooLarge,
    /// The depth slice does not contain exactly one value per pixel.
    IncorrectDepthCount { expected: usize, actual: usize },
    /// A depth value is non-finite or outside the inclusive range [0, 1].
    InvalidDepth { index: usize },
}

impl fmt::Display for DepthPyramidError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyImage => write!(f, "depth image dimensions must be nonzero"),
            Self::ImageTooLarge => write!(f, "depth image dimensions exceed usize"),
            Self::IncorrectDepthCount { expected, actual } => {
                write!(f, "expected {expected} depth values, got {actual}")
            }
            Self::InvalidDepth { index } => write!(f, "invalid depth at index {index}"),
        }
    }
}

impl std::error::Error for DepthPyramidError {}

/// A row-major image in a [`DepthPyramid`].
#[derive(Clone, Debug, PartialEq)]
pub struct DepthLevel {
    width: u32,
    height: u32,
    depths: Vec<f32>,
}

impl DepthLevel {
    /// Width in texels.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Height in texels.
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Minimum reverse-Z depths, in row-major order.
    pub fn depths(&self) -> &[f32] {
        &self.depths
    }
}

/// A minimum-reduction depth pyramid, including the original image at level 0.
///
/// Each parent covers up to 2 × 2 texels of the previous level; odd dimensions
/// are rounded up, so edge texels are never dropped.
#[derive(Clone, Debug, PartialEq)]
pub struct DepthPyramid {
    levels: Vec<DepthLevel>,
}

/// Conservative screen-space bounds for an occlusion query.
///
/// `min` and `max` are pixel-edge coordinates in the same viewport and image
/// orientation as the depth image, with `min` inclusive and `max` exclusive.
/// They must enclose every potentially covered sample, including numerical and
/// rasterization margins. `nearest_depth` must be an upper bound on the
/// object's reverse-Z depth at every covered sample, including projection and
/// depth quantization error. The caller is responsible for these guarantees.
/// Set `intersects_near_plane` when the object touches or crosses the near
/// plane, or whenever safe projection cannot be established; such queries are
/// never rejected. Invalid or empty rectangles also remain visible.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ProjectedBounds {
    /// Minimum pixel-edge coordinate, in image space.
    pub min: [f64; 2],
    /// Maximum pixel-edge coordinate, in image space.
    pub max: [f64; 2],
    /// Upper bound on the object's closest reverse-Z depth, in [0, 1].
    pub nearest_depth: f32,
    /// Whether near-plane intersection or uncertain projection prevents culling.
    pub intersects_near_plane: bool,
}

impl DepthPyramid {
    /// Builds a pyramid from an actual row-major reverse-Z depth image.
    ///
    /// The image must describe conservative occluders for the exact view and
    /// viewport being queried. Clear/background pixels must be 0. For MSAA,
    /// resolve each pixel to the minimum of *all* samples, including uncovered
    /// samples. Stale depth, non-conservative occluders, incompatible depth
    /// conventions, or depth containing the tested object can invalidate the
    /// result; managing those conditions is the caller's responsibility.
    pub fn new(width: u32, height: u32, depth: &[f32]) -> Result<Self, DepthPyramidError> {
        if width == 0 || height == 0 {
            return Err(DepthPyramidError::EmptyImage);
        }
        let expected = (width as usize)
            .checked_mul(height as usize)
            .ok_or(DepthPyramidError::ImageTooLarge)?;
        if depth.len() != expected {
            return Err(DepthPyramidError::IncorrectDepthCount {
                expected,
                actual: depth.len(),
            });
        }
        if let Some(index) = depth
            .iter()
            .position(|&value| !value.is_finite() || !(0.0..=1.0).contains(&value))
        {
            return Err(DepthPyramidError::InvalidDepth { index });
        }

        let mut levels = vec![DepthLevel {
            width,
            height,
            depths: depth.to_vec(),
        }];
        while levels.last().unwrap().width > 1 || levels.last().unwrap().height > 1 {
            let previous = levels.last().unwrap();
            let width = previous.width / 2 + previous.width % 2;
            let height = previous.height / 2 + previous.height % 2;
            let mut depths = Vec::with_capacity(width as usize * height as usize);
            for y in 0..height {
                for x in 0..width {
                    let mut minimum = 1.0_f32;
                    for dy in 0..2 {
                        for dx in 0..2 {
                            let source_x = x * 2 + dx;
                            let source_y = y * 2 + dy;
                            if source_x < previous.width && source_y < previous.height {
                                let index =
                                    source_y as usize * previous.width as usize + source_x as usize;
                                minimum = minimum.min(previous.depths[index]);
                            }
                        }
                    }
                    depths.push(minimum);
                }
            }
            levels.push(DepthLevel {
                width,
                height,
                depths,
            });
        }
        Ok(Self { levels })
    }

    /// Levels from the original image through the final 1 × 1 reduction.
    pub fn levels(&self) -> &[DepthLevel] {
        &self.levels
    }

    /// Returns true only when the entire projected rectangle is occluded.
    ///
    /// The rectangle is clipped to the image, then expanded to include every
    /// touched pixel. A suitable pyramid level covers those pixels, possibly
    /// including extra pixels. Rejection requires the object's closest depth
    /// to be *strictly* behind every covered depth; equality stays visible.
    /// A rectangle entirely outside the viewport returns false rather than
    /// treating this method as a frustum test.
    pub fn is_occluded(&self, bounds: ProjectedBounds) -> bool {
        if bounds.intersects_near_plane
            || !bounds.nearest_depth.is_finite()
            || !(0.0..=1.0).contains(&bounds.nearest_depth)
            || (0..2).any(|axis| {
                !bounds.min[axis].is_finite()
                    || !bounds.max[axis].is_finite()
                    || bounds.min[axis] >= bounds.max[axis]
            })
        {
            return false;
        }

        let base = &self.levels[0];
        let dimensions = [f64::from(base.width), f64::from(base.height)];
        if (0..2).any(|axis| bounds.max[axis] <= 0.0 || bounds.min[axis] >= dimensions[axis]) {
            return false;
        }
        let mut min = [0_u32; 2];
        let mut max = [0_u32; 2];
        for axis in 0..2 {
            min[axis] = bounds.min[axis].clamp(0.0, dimensions[axis]).floor() as u32;
            // The viewport intersection is nonempty, so the exclusive end is
            // at least one and subtracting one yields an inclusive texel index.
            max[axis] = bounds.max[axis].clamp(0.0, dimensions[axis]).ceil() as u32 - 1;
        }

        let mut level_index = 0;
        while (max[0] - min[0] > 1 || max[1] - min[1] > 1) && level_index + 1 < self.levels.len() {
            for axis in 0..2 {
                min[axis] /= 2;
                max[axis] /= 2;
            }
            level_index += 1;
        }
        let level = &self.levels[level_index];
        for y in min[1]..=max[1] {
            for x in min[0]..=max[0] {
                let index = y as usize * level.width as usize + x as usize;
                if bounds.nearest_depth >= level.depths[index] {
                    return false;
                }
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bounds(min: [f64; 2], max: [f64; 2], nearest_depth: f32) -> ProjectedBounds {
        ProjectedBounds {
            min,
            max,
            nearest_depth,
            intersects_near_plane: false,
        }
    }

    #[test]
    fn reduces_odd_dimensions_without_losing_edges() {
        let pyramid =
            DepthPyramid::new(3, 3, &[0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]).unwrap();
        assert_eq!(pyramid.levels().len(), 3);
        assert_eq!(
            (pyramid.levels()[1].width(), pyramid.levels()[1].height()),
            (2, 2)
        );
        assert_eq!(pyramid.levels()[1].depths(), &[0.5, 0.4, 0.2, 0.1]);
        assert_eq!(pyramid.levels()[2].depths(), &[0.1]);
    }

    #[test]
    fn background_and_equal_depth_stay_visible() {
        let pyramid = DepthPyramid::new(2, 2, &[0.8, 0.8, 0.8, 0.0]).unwrap();
        assert!(!pyramid.is_occluded(bounds([0.0, 0.0], [2.0, 2.0], 0.2)));
        assert!(!pyramid.is_occluded(bounds([0.0, 0.0], [1.0, 1.0], 0.8)));
        assert!(pyramid.is_occluded(bounds([0.0, 0.0], [1.0, 1.0], 0.2)));
        assert!(!pyramid.is_occluded(bounds([1.0, 1.0], [2.0, 2.0], 0.0)));
    }

    #[test]
    fn partial_pixels_and_viewport_clipping_are_conservative() {
        let pyramid = DepthPyramid::new(3, 1, &[0.9, 0.0, 0.9]).unwrap();
        assert!(!pyramid.is_occluded(bounds([0.5, 0.0], [1.01, 1.0], 0.1)));
        assert!(pyramid.is_occluded(bounds([-10.0, 0.0], [0.5, 1.0], 0.1)));
        assert!(!pyramid.is_occluded(bounds([3.0, 0.0], [4.0, 1.0], 0.1)));
    }

    #[test]
    fn near_plane_and_invalid_inputs_remain_visible() {
        let pyramid = DepthPyramid::new(1, 1, &[1.0]).unwrap();
        let mut query = bounds([0.0, 0.0], [1.0, 1.0], 0.5);
        query.intersects_near_plane = true;
        assert!(!pyramid.is_occluded(query));
        query.intersects_near_plane = false;
        query.nearest_depth = f32::NAN;
        assert!(!pyramid.is_occluded(query));
        query.nearest_depth = -0.1;
        assert!(!pyramid.is_occluded(query));
        query.nearest_depth = 0.5;
        query.min[0] = f64::NAN;
        assert!(!pyramid.is_occluded(query));
        query.min[0] = 1.0;
        assert!(!pyramid.is_occluded(query));
    }

    #[test]
    fn construction_rejects_invalid_depth_images() {
        assert_eq!(
            DepthPyramid::new(0, 1, &[]),
            Err(DepthPyramidError::EmptyImage)
        );
        assert_eq!(
            DepthPyramid::new(2, 1, &[0.5]),
            Err(DepthPyramidError::IncorrectDepthCount {
                expected: 2,
                actual: 1
            })
        );
        for value in [f32::NAN, f32::INFINITY, -0.1, 1.1] {
            assert_eq!(
                DepthPyramid::new(1, 1, &[value]),
                Err(DepthPyramidError::InvalidDepth { index: 0 })
            );
        }
    }

    #[test]
    fn coarse_levels_can_only_add_occluder_footprint() {
        let pyramid = DepthPyramid::new(5, 1, &[0.0, 0.9, 0.9, 0.9, 0.9]).unwrap();
        // The selected coarse texel includes the background pixel at x=0.
        assert!(!pyramid.is_occluded(bounds([1.0, 0.0], [4.0, 1.0], 0.5)));
        assert!(pyramid.is_occluded(bounds([2.0, 0.0], [5.0, 1.0], 0.5)));
    }
}
