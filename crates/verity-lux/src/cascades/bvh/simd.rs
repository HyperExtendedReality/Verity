//! AVX2 SAH for finite f32 centroid mappings.

#[cfg(target_arch = "x86")]
use std::arch::x86 as arch;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64 as arch;

use glam::Vec3A;

use super::{BINS, Bin, Bounds, Partition, Sah, partition_by};

/// Proof that the build can use AVX2 on this host, including on Rayon workers.
#[derive(Clone, Copy)]
pub(super) struct Avx2(());

impl Avx2 {
    pub(super) fn detect() -> Option<Self> {
        std::is_x86_feature_detected!("avx2").then_some(Self(()))
    }

    pub(super) fn partition<T>(
        self,
        elements: &mut [T],
        bounds: &impl Fn(&T) -> Bounds,
        sah: Sah,
        origin: Vec3A,
        scale: Vec3A,
    ) -> Option<Partition> {
        // SAFETY: The only constructor checks CPU and OS support for AVX2.
        unsafe { partition(elements, bounds, sah, origin, scale) }
    }
}

/// # Safety
/// The current CPU and OS must support AVX2.
#[target_feature(enable = "avx2")]
unsafe fn partition<T>(
    elements: &mut [T],
    bounds: &impl Fn(&T) -> Bounds,
    sah: Sah,
    origin: Vec3A,
    scale: Vec3A,
) -> Option<Partition> {
    let lanes = |v: Vec3A| arch::_mm_setr_ps(v.x, v.y, v.z, 0.0);
    let xyz = |v| {
        Vec3A::new(
            arch::_mm_cvtss_f32(v),
            f32::from_bits(arch::_mm_extract_ps::<1>(v) as u32),
            f32::from_bits(arch::_mm_extract_ps::<2>(v) as u32),
        )
    };
    let origin = lanes(origin);
    let scale = lanes(scale);
    let origin8 = arch::_mm256_set_m128(origin, origin);
    let scale8 = arch::_mm256_set_m128(scale, scale);
    let last_bin = arch::_mm256_set1_epi32(sah.bins as i32 - 1);
    let map_centers = |lo, hi| {
        let centers = arch::_mm256_set_m128(lanes(hi), lanes(lo));
        let coords = arch::_mm256_mul_ps(arch::_mm256_sub_ps(centers, origin8), scale8);
        // Truncation matches the portable unsigned cast after clamping negatives and NaNs.
        // Signed overflow produces INT_MIN, which the unsigned min clamps to the last bin.
        arch::_mm256_min_epu32(
            arch::_mm256_cvttps_epi32(arch::_mm256_max_ps(coords, arch::_mm256_setzero_ps())),
            last_bin,
        )
    };

    // Like TinyBVH, store [-min | max] so one vector max extends a bin's bounds.
    // Keep counts separate to avoid a cache-line-sized stride for each packed box.
    let mut boxes = [[arch::_mm256_set1_ps(f32::NEG_INFINITY); BINS]; 3];
    let mut counts = [[0; BINS]; 3];
    let mut add = |coords, item: Bounds| {
        let packed = arch::_mm256_set_m128(lanes(item.max), lanes(-item.min));
        let [x, y, z] = [
            arch::_mm_cvtsi128_si32(coords) as usize,
            arch::_mm_extract_epi32::<1>(coords) as usize,
            arch::_mm_extract_epi32::<2>(coords) as usize,
        ];
        if sah.active & 1 != 0 {
            boxes[0][x] = arch::_mm256_max_ps(boxes[0][x], packed);
            counts[0][x] += 1;
        }
        if sah.active & 2 != 0 {
            boxes[1][y] = arch::_mm256_max_ps(boxes[1][y], packed);
            counts[1][y] += 1;
        }
        if sah.active & 4 != 0 {
            boxes[2][z] = arch::_mm256_max_ps(boxes[2][z], packed);
            counts[2][z] += 1;
        }
    };
    let mut pairs = elements.chunks_exact(2);
    for pair in &mut pairs {
        let lo = bounds(&pair[0]);
        let hi = bounds(&pair[1]);
        let coords = map_centers(lo.center(), hi.center());
        add(arch::_mm256_castsi256_si128(coords), lo);
        add(arch::_mm256_extracti128_si256::<1>(coords), hi);
    }
    if let [element] = pairs.remainder() {
        let item = bounds(element);
        let coords = map_centers(item.center(), Vec3A::ZERO);
        add(arch::_mm256_castsi256_si128(coords), item);
    }

    let (axis, plane) = sah.best_split(elements.len(), |axis, i| {
        let count = counts[axis][i];
        if count == 0 {
            return Bin::EMPTY;
        }
        let packed = boxes[axis][i];
        Bin {
            bounds: Bounds {
                min: -xyz(arch::_mm256_castps256_ps128(packed)),
                max: xyz(arch::_mm256_extractf128_ps::<1>(packed)),
            },
            count,
        }
    })?;
    let plane = arch::_mm_set1_ps(plane as f32);
    let axis = 1 << axis;
    Some(partition_by(elements, bounds, |center| {
        let coords = arch::_mm_mul_ps(arch::_mm_sub_ps(lanes(center), origin), scale);
        // The chosen plane is an integer in 1..bins, so comparing before the cast and
        // clamp gives the same side, including centroids rounded onto the plane.
        arch::_mm_movemask_ps(arch::_mm_cmp_ps::<{ arch::_CMP_GE_OQ }>(coords, plane)) & axis == 0
    }))
}
