//! Whole-primitive LODs backed by bounded, demand-loaded CPU index pages.
//!
//! Pages contain complete triangles in little-endian u32 form. The coarsest
//! level is pinned, so a failed demand load always leaves drawable geometry.
//! This is synchronous CPU residency, not GPU streaming: the caller must upload
//! the selected indices and synchronize GPU buffers with every frame in flight.
//! Simplification is performed independently for each material primitive; it is
//! not continuous cluster LOD. See [`crate::lod`] for its seam and error limits.

use std::convert::Infallible;
use std::fmt;

use crate::Bounds;
use crate::lod::{LodError, LodOptions, build_lods};
use crate::residency::{PageCache, PageError, PageId, PageSource};

#[derive(Debug)]
struct IndexPages(Vec<Box<[u8]>>);

impl PageSource for IndexPages {
    type Error = Infallible;

    fn page_count(&self) -> u32 {
        self.0.len() as u32
    }

    fn page_size(&mut self, id: PageId) -> Result<usize, Self::Error> {
        Ok(self.0[id.0 as usize].len())
    }

    fn load_page(&mut self, id: PageId) -> Result<Box<[u8]>, Self::Error> {
        Ok(self.0[id.0 as usize].clone())
    }
}

#[derive(Debug)]
struct Level {
    first_page: u32,
    page_count: u32,
    index_count: usize,
    error: f64,
}

/// An invalid cook, or a cache budget too small to guarantee the coarse level.
#[derive(Debug)]
pub enum VirtualGeometryError {
    Lod(LodError),
    /// Pages must hold at least one triangle and have a size divisible by 12.
    InvalidPageSize,
    /// The page count cannot be represented by the page ID type.
    TooManyPages,
    /// The resident budget cannot hold the pinned coarsest level.
    FallbackExceedsBudget {
        bytes: usize,
        budget: usize,
    },
    /// A cache invariant or source operation failed.
    Cache(PageError<Infallible>),
}

impl fmt::Display for VirtualGeometryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Lod(error) => write!(f, "LOD cooking failed: {error}"),
            Self::InvalidPageSize => write!(f, "page size must be a positive multiple of 12"),
            Self::TooManyPages => write!(f, "geometry has too many pages"),
            Self::FallbackExceedsBudget { bytes, budget } => {
                write!(f, "fallback requires {bytes} bytes but budget is {budget}")
            }
            Self::Cache(error) => write!(f, "geometry page cache: {error}"),
        }
    }
}

impl std::error::Error for VirtualGeometryError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Lod(error) => Some(error),
            Self::Cache(error) => Some(error),
            _ => None,
        }
    }
}

/// Cooked index hierarchy and its bounded CPU resident page cache.
///
/// The backing page store, metadata, output vector, and transient source loads
/// are outside `resident_budget`. Cache residency is per object, not per frame;
/// renderers must retain/copy uploaded indices until their GPU work completes.
pub struct VirtualGeometry {
    levels: Vec<Level>,
    bounds: Option<Bounds>,
    cache: PageCache<IndexPages>,
}

impl VirtualGeometry {
    /// Simplify one static material primitive and pin its coarse fallback.
    /// All generated levels refer to the input vertex/attribute buffers.
    pub fn cook(
        positions: &[[f32; 3]],
        indices: &[u32],
        options: LodOptions,
        page_bytes: usize,
        resident_budget: usize,
    ) -> Result<Self, VirtualGeometryError> {
        if page_bytes == 0 || !page_bytes.is_multiple_of(12) {
            return Err(VirtualGeometryError::InvalidPageSize);
        }
        let lods = build_lods(positions, indices, options).map_err(VirtualGeometryError::Lod)?;
        let mut bounds = None::<Bounds>;
        for &index in indices {
            let position = positions[index as usize];
            match &mut bounds {
                Some(bounds) => {
                    for (axis, &value) in position.iter().enumerate() {
                        bounds.min[axis] = bounds.min[axis].min(value);
                        bounds.max[axis] = bounds.max[axis].max(value);
                    }
                }
                None => {
                    bounds = Some(Bounds {
                        min: position,
                        max: position,
                    })
                }
            }
        }
        let mut pages = Vec::new();
        let mut levels = Vec::with_capacity(lods.len());
        for lod in lods {
            let first_page =
                u32::try_from(pages.len()).map_err(|_| VirtualGeometryError::TooManyPages)?;
            for chunk in lod.indices.chunks(page_bytes / 4) {
                if pages.len() == u32::MAX as usize {
                    return Err(VirtualGeometryError::TooManyPages);
                }
                let mut bytes = Vec::with_capacity(chunk.len() * 4);
                for index in chunk {
                    bytes.extend_from_slice(&index.to_le_bytes());
                }
                pages.push(bytes.into_boxed_slice());
            }
            levels.push(Level {
                first_page,
                page_count: pages.len() as u32 - first_page,
                index_count: lod.indices.len(),
                error: lod.error,
            });
        }
        let fallback = levels.last().expect("LOD cooking always produces a level");
        let fallback_bytes = fallback.index_count * 4;
        if fallback_bytes > resident_budget {
            return Err(VirtualGeometryError::FallbackExceedsBudget {
                bytes: fallback_bytes,
                budget: resident_budget,
            });
        }
        let mut cache = PageCache::new(IndexPages(pages), resident_budget);
        for page in fallback.first_page..fallback.first_page + fallback.page_count {
            cache
                .pin(PageId(page))
                .map_err(VirtualGeometryError::Cache)?;
        }
        Ok(Self {
            levels,
            bounds,
            cache,
        })
    }

    /// Bounds of the original referenced positions; `None` for empty geometry.
    /// All simplified indices reference those same positions.
    pub fn bounds(&self) -> Option<Bounds> {
        self.bounds
    }

    /// Number of levels, including the original and coarse fallback.
    pub fn level_count(&self) -> usize {
        self.levels.len()
    }

    /// Approximate accumulated object-space simplification error.
    pub fn level_error(&self, level: usize) -> Option<f64> {
        self.levels.get(level).map(|level| level.error)
    }

    /// Bytes currently resident in the bounded CPU cache, including the fallback.
    pub fn resident_bytes(&self) -> usize {
        self.cache.resident_bytes()
    }

    /// Select a level, synchronously load its pages and copy its triangle indices.
    ///
    /// Selection uses the same approximate metric and conservative invalid-input
    /// behavior as [`crate::lod::select_lod`]. If the desired level cannot fit
    /// alongside the pinned fallback, the fallback is returned instead. Its
    /// screen error may exceed the requested budget. `out_indices` can be reused
    /// across calls; its contents are replaced on success. Page loads are CPU
    /// operations and do not synchronize or upload GPU resources.
    pub fn resolve(
        &mut self,
        pixels_per_object_unit: f64,
        max_error_pixels: f64,
        out_indices: &mut Vec<u32>,
    ) -> Result<usize, VirtualGeometryError> {
        let mut desired = 0;
        if pixels_per_object_unit.is_finite()
            && pixels_per_object_unit >= 0.0
            && max_error_pixels.is_finite()
            && max_error_pixels >= 0.0
        {
            for (index, level) in self.levels.iter().enumerate().skip(1) {
                if pixels_per_object_unit == 0.0
                    || level.error <= max_error_pixels / pixels_per_object_unit
                {
                    desired = index;
                }
            }
        }
        let fallback = self.levels.len() - 1;
        let selected = if desired == fallback
            || self.levels[desired].index_count * 4
                > self.cache.budget() - self.levels[fallback].index_count * 4
        {
            fallback
        } else {
            let level = &self.levels[desired];
            let start = level.first_page;
            let end = start + level.page_count;
            let mut pinned_end = start;
            let mut fits = true;
            for page in start..end {
                match self.cache.pin(PageId(page)) {
                    Ok(()) => pinned_end = page + 1,
                    Err(PageError::PageTooLarge { .. } | PageError::InsufficientSpace { .. }) => {
                        fits = false;
                        break;
                    }
                    Err(error) => {
                        for pinned in start..pinned_end {
                            self.cache
                                .unpin(PageId(pinned))
                                .map_err(VirtualGeometryError::Cache)?;
                        }
                        return Err(VirtualGeometryError::Cache(error));
                    }
                }
            }
            for pinned in start..pinned_end {
                self.cache
                    .unpin(PageId(pinned))
                    .map_err(VirtualGeometryError::Cache)?;
            }
            if fits { desired } else { fallback }
        };
        let level = &self.levels[selected];
        out_indices.clear();
        out_indices.reserve(level.index_count);
        for page in level.first_page..level.first_page + level.page_count {
            let bytes = self
                .cache
                .get(PageId(page))
                .map_err(VirtualGeometryError::Cache)?
                .ok_or(VirtualGeometryError::Cache(PageError::NotResident(PageId(
                    page,
                ))))?;
            for &index in bytes.as_chunks::<4>().0 {
                out_indices.push(u32::from_le_bytes(index));
            }
        }
        Ok(selected)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn options() -> LodOptions {
        LodOptions {
            max_levels: 4,
            triangle_ratio: 0.5,
            max_step_error: 100.0,
        }
    }

    fn grid() -> (Vec<[f32; 3]>, Vec<u32>) {
        let mut positions = Vec::new();
        for y in 0..9 {
            for x in 0..9 {
                positions.push([x as f32, y as f32, 0.0]);
            }
        }
        let mut indices = Vec::new();
        for y in 0..8 {
            for x in 0..8 {
                let a = y * 9 + x;
                indices.extend_from_slice(&[a, a + 1, a + 9, a + 1, a + 10, a + 9]);
            }
        }
        (positions, indices)
    }

    #[test]
    fn cooks_and_resolves_real_levels() {
        let (positions, indices) = grid();
        let lods = build_lods(&positions, &indices, options()).unwrap();
        assert!(lods.len() > 1);
        let mut geometry =
            VirtualGeometry::cook(&positions, &indices, options(), 48, 8192).unwrap();
        let mut output = Vec::new();
        assert_eq!(
            geometry.resolve(f64::INFINITY, 1.0, &mut output).unwrap(),
            0
        );
        assert_eq!(output, indices);
        let selected = geometry.resolve(0.0, 1.0, &mut output).unwrap();
        assert_eq!(selected, lods.len() - 1);
        assert_eq!(output, lods[selected].indices);
        assert!(output.len() < indices.len());
        assert_eq!(
            geometry.bounds(),
            Some(Bounds {
                min: [0.0; 3],
                max: [8.0, 8.0, 0.0]
            })
        );
    }

    #[test]
    fn bounded_cache_falls_back_without_partial_triangles() {
        let (positions, indices) = grid();
        let lods = build_lods(&positions, &indices, options()).unwrap();
        let coarse = lods.last().unwrap();
        let budget = coarse.indices.len() * 4;
        let mut geometry =
            VirtualGeometry::cook(&positions, &indices, options(), 12, budget).unwrap();
        let mut output = Vec::new();
        assert_eq!(
            geometry.resolve(f64::INFINITY, 1.0, &mut output).unwrap(),
            lods.len() - 1
        );
        assert_eq!(output, coarse.indices);
        assert!(geometry.resident_bytes() <= budget);
        assert!(matches!(
            VirtualGeometry::cook(&positions, &indices, options(), 12, budget - 1),
            Err(VirtualGeometryError::FallbackExceedsBudget { .. })
        ));
    }

    #[test]
    fn evicted_pages_reload_and_fallback_stays_pinned() {
        let (positions, indices) = grid();
        let mut geometry =
            VirtualGeometry::cook(&positions, &indices, options(), 48, 8192).unwrap();
        let mut output = Vec::new();
        assert_eq!(
            geometry.resolve(f64::INFINITY, 0.0, &mut output).unwrap(),
            0
        );
        let first = PageId(geometry.levels[0].first_page);
        assert!(geometry.cache.unload(first).unwrap());
        let fallback = geometry.levels.last().unwrap().first_page;
        assert!(matches!(
            geometry.cache.unload(PageId(fallback)),
            Err(PageError::Pinned(_))
        ));
        assert_eq!(
            geometry.resolve(f64::INFINITY, 0.0, &mut output).unwrap(),
            0
        );
        assert_eq!(output, indices);
        assert!(geometry.resident_bytes() <= 8192);
    }

    #[test]
    fn empty_and_invalid_input() {
        let mut geometry = VirtualGeometry::cook(&[], &[], options(), 12, 0).unwrap();
        let mut output = vec![42];
        assert_eq!(geometry.resolve(1.0, 1.0, &mut output).unwrap(), 0);
        assert!(output.is_empty());
        assert_eq!(geometry.bounds(), None);
        assert!(matches!(
            VirtualGeometry::cook(&[], &[], options(), 4, 0),
            Err(VirtualGeometryError::InvalidPageSize)
        ));
    }
}
