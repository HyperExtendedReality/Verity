//! Surface islands for the outgoing radiance cache (`surfaceUpdate` in
//! `shaders/cascades.slang`).
//!
//! [`chart`] cuts one BLAS into islands: triangles joined across shared edges
//! while they face within [`ISLAND_COS`] of the island's first triangle, each
//! island flattened onto its own plane in local metres. Every instance then
//! gets its own copy of its mesh's islands in one atlas ([`pack`]), at a
//! density that keeps texels about the same size in the world, and
//! [`rasterize`] writes the texel records the cache pass shades from: a
//! triangle and barycentrics per texel, so no texel starts inside geometry.

use glam::{UVec4, Vec2, Vec3};

/// cos 35°: how far a triangle may turn from its island's first triangle.
const ISLAND_COS: f32 = 0.819;
const ISLAND_TRIANGLES: usize = 4096;
/// Corners closer than this weld, since glTF splits vertices at UV seams.
const WELD: f32 = 1.0e-4;
/// Atlas width in texels.
pub const ATLAS_WIDTH: u32 = 2048;

/// One BLAS's islands. Triangles are in BLAS order.
#[derive(Clone, Default)]
pub struct Charts {
    /// Each island's extent on its plane, in local metres.
    pub extents: Vec<Vec2>,
    /// Each triangle's island.
    pub island: Vec<u32>,
    /// Each triangle's corners on its island's plane, in local metres from the
    /// island's lower corner.
    pub uv: Vec<[Vec2; 3]>,
}

/// An island's place in the atlas, in texels.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rect {
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
}

pub fn chart(triangles: &[[Vec3; 3]]) -> Charts {
    let n = triangles.len();
    let normals: Vec<Vec3> = triangles
        .iter()
        .map(|[a, b, c]| (*b - *a).cross(*c - *a).normalize_or_zero())
        .collect();
    // Neighbours across welded edges: sort the edges so equal ones meet.
    let weld = |p: Vec3| (p / WELD).round().as_ivec3().to_array();
    let mut edges: Vec<([[i32; 3]; 2], u32)> = Vec::with_capacity(n * 3);
    for (t, corners) in triangles.iter().enumerate() {
        for i in 0..3 {
            let (a, b) = (weld(corners[i]), weld(corners[(i + 1) % 3]));
            edges.push((if a < b { [a, b] } else { [b, a] }, t as u32));
        }
    }
    edges.sort_unstable();
    let mut neighbours: Vec<Vec<u32>> = vec![Vec::new(); n];
    for run in edges.chunk_by(|x, y| x.0 == y.0) {
        for (i, &(_, a)) in run.iter().enumerate() {
            for &(_, b) in &run[i + 1..] {
                if a != b {
                    neighbours[a as usize].push(b);
                    neighbours[b as usize].push(a);
                }
            }
        }
    }

    let mut charts = Charts {
        extents: Vec::new(),
        island: vec![u32::MAX; n],
        uv: vec![[Vec2::ZERO; 3]; n],
    };
    let mut members = Vec::new();
    for seed in 0..n {
        if charts.island[seed] != u32::MAX {
            continue;
        }
        let id = charts.extents.len() as u32;
        let facing = normals[seed];
        members.clear();
        members.push(seed as u32);
        charts.island[seed] = id;
        let mut next = 0;
        while next < members.len() {
            let t = members[next] as usize;
            next += 1;
            for &other in &neighbours[t] {
                let o = other as usize;
                if charts.island[o] == u32::MAX
                    && members.len() < ISLAND_TRIANGLES
                    && normals[o].dot(facing) >= ISLAND_COS
                {
                    charts.island[o] = id;
                    members.push(other);
                }
            }
        }
        // Project onto the area-weighted plane, which every member faces.
        let plane = members
            .iter()
            .map(|&t| {
                let [a, b, c] = triangles[t as usize];
                (b - a).cross(c - a)
            })
            .sum::<Vec3>()
            .try_normalize()
            .unwrap_or(facing);
        let (u, v) = plane.any_orthonormal_pair();
        let (mut low, mut high) = (Vec2::MAX, Vec2::MIN);
        for &t in &members {
            let projected = triangles[t as usize].map(|p| Vec2::new(p.dot(u), p.dot(v)));
            for p in projected {
                low = low.min(p);
                high = high.max(p);
            }
            charts.uv[t as usize] = projected;
        }
        for &t in &members {
            charts.uv[t as usize] = charts.uv[t as usize].map(|p| p - low);
        }
        charts.extents.push(high - low);
    }
    charts
}

/// An island's size in texels at `density` texels per local metre, at least
/// one. No gutter: lookups clamp to their own island and never filter across
/// texels. None when wider than the atlas.
pub fn island_size(extent: Vec2, density: f32) -> Option<[u32; 2]> {
    let side = |e: f32| (e * density).ceil().max(1.0) as u32;
    let size = [side(extent.x), side(extent.y)];
    (size[0] <= ATLAS_WIDTH && size[1] <= ATLAS_WIDTH).then_some(size)
}

/// Shelf-packs `sizes` into rows [`ATLAS_WIDTH`] wide, tallest first. Returns
/// each one's place and the height used.
pub fn pack(sizes: &[[u32; 2]]) -> (Vec<Rect>, u32) {
    let mut order: Vec<usize> = (0..sizes.len()).collect();
    order.sort_unstable_by_key(|&i| std::cmp::Reverse(sizes[i][1]));
    let mut rects = vec![
        Rect {
            x: 0,
            y: 0,
            w: 0,
            h: 0
        };
        sizes.len()
    ];
    let (mut x, mut y, mut shelf) = (0, 0, 0);
    for i in order {
        let [w, h] = sizes[i];
        if x + w > ATLAS_WIDTH {
            (x, y, shelf) = (0, y + shelf, 0);
        }
        rects[i] = Rect { x, y, w, h };
        x += w;
        shelf = shelf.max(h);
    }
    (rects, y + shelf)
}

/// `x` as half-float bits, rounded to nearest. Chart coordinates are
/// non-negative and far below the half-float maximum; tiny values flush to zero.
pub fn half(x: f32) -> u32 {
    let bits = x.to_bits();
    let sign = (bits >> 16) & 0x8000;
    let exponent = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    if exponent <= 0 {
        return sign;
    }
    if exponent >= 31 {
        return sign | 0x7bff;
    }
    let mantissa = bits & 0x7f_ffff;
    // A carry out of the mantissa correctly bumps the exponent.
    (sign | ((exponent as u32) << 10) | (mantissa >> 13)) + ((mantissa >> 12) & 1)
}

fn barycentric(p: Vec2, [a, b, c]: [Vec2; 3]) -> Option<Vec3> {
    let d = (b - a).perp_dot(c - a);
    if d.abs() < 1.0e-12 {
        return None;
    }
    let w1 = (p - a).perp_dot(c - a) / d;
    let w2 = (b - a).perp_dot(p - a) / d;
    Some(Vec3::new(1.0 - w1 - w2, w1, w2))
}

/// A texel record: `x` = global triangle, `y` = instance, `z` = barycentrics
/// of the second and third corners as unorm16, `w` = 1.
fn record(triangle: u32, instance: u32, weights: Vec3) -> UVec4 {
    let w = weights.max(Vec3::ZERO);
    let w = w / w.element_sum().max(1.0e-12);
    let unorm = |x: f32| (x.clamp(0.0, 1.0) * 65535.0).round() as u32;
    UVec4::new(triangle, instance, unorm(w.y) | (unorm(w.z) << 16), 1)
}

/// One instance's texel records, one block per placed island, row-major in
/// its rect. `rects` holds each island's place, or None when it has none.
/// Triangle `j` of `charts` is global triangle `first + j`.
pub fn rasterize(
    charts: &Charts,
    first: u32,
    instance: u32,
    density: f32,
    rects: &[Option<Rect>],
) -> Vec<(Rect, Vec<UVec4>)> {
    let mut blocks: Vec<Option<(Rect, Vec<UVec4>)>> = rects
        .iter()
        .map(|r| r.map(|r| (r, vec![UVec4::ZERO; (r.w * r.h) as usize])))
        .collect();
    // Corners in rect-local texels.
    let corners = |j: usize| charts.uv[j].map(|p| p * density);
    for (j, &island) in charts.island.iter().enumerate() {
        let Some((rect, texels)) = &mut blocks[island as usize] else {
            continue;
        };
        let q = corners(j);
        let low = q[0].min(q[1]).min(q[2]).floor().max(Vec2::ZERO);
        let high = q[0].max(q[1]).max(q[2]).ceil();
        let (x1, y1) = ((high.x as u32).min(rect.w), (high.y as u32).min(rect.h));
        let mut covered = false;
        for y in low.y as u32..y1 {
            for x in low.x as u32..x1 {
                let centre = Vec2::new(x as f32 + 0.5, y as f32 + 0.5);
                if let Some(w) = barycentric(centre, q).filter(|w| w.min_element() >= -1.0e-6) {
                    texels[(y * rect.w + x) as usize] = record(first + j as u32, instance, w);
                    covered = true;
                }
            }
        }
        // A triangle smaller than a texel still claims the texel at its centre.
        if !covered {
            let centre = ((q[0] + q[1] + q[2]) / 3.0).as_uvec2();
            let at = (centre.y.min(rect.h - 1) * rect.w + centre.x.min(rect.w - 1)) as usize;
            if texels[at].w == 0 {
                texels[at] = record(first + j as u32, instance, Vec3::splat(1.0 / 3.0));
            }
        }
    }
    // Gutter and gaps take a neighbour's triangle, clamped onto it, so bilinear
    // footprints and texel-boundary lookups still land on a surface.
    for (rect, texels) in blocks.iter_mut().flatten() {
        // Gutter and rounding leave at most a few texels to reach, corners included.
        for _ in 0..8 {
            let before = texels.clone();
            for y in 0..rect.h {
                for x in 0..rect.w {
                    let at = (y * rect.w + x) as usize;
                    if before[at].w != 0 {
                        continue;
                    }
                    let neighbour = [(-1i32, 0i32), (1, 0), (0, -1), (0, 1)]
                        .into_iter()
                        .map(|(dx, dy)| (x as i32 + dx, y as i32 + dy))
                        .filter(|&(nx, ny)| {
                            nx >= 0 && ny >= 0 && (nx as u32) < rect.w && (ny as u32) < rect.h
                        })
                        .map(|(nx, ny)| before[(ny as u32 * rect.w + nx as u32) as usize])
                        .find(|n| n.w != 0);
                    if let Some(n) = neighbour {
                        let j = (n.x - first) as usize;
                        let centre = Vec2::new(x as f32 + 0.5, y as f32 + 0.5);
                        let w = barycentric(centre, corners(j)).unwrap_or(Vec3::splat(1.0 / 3.0));
                        texels[at] = record(n.x, instance, w);
                    }
                }
            }
            if *texels == before {
                break;
            }
        }
    }
    blocks.into_iter().flatten().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quad(a: Vec3, b: Vec3, c: Vec3, d: Vec3) -> [[Vec3; 3]; 2] {
        [[a, b, c], [a, c, d]]
    }

    /// A unit cube, each face with its own vertices, as glTF stores it.
    fn cube() -> Vec<[Vec3; 3]> {
        let p = |x, y, z| Vec3::new(x, y, z);
        [
            quad(p(0., 0., 0.), p(0., 1., 0.), p(1., 1., 0.), p(1., 0., 0.)),
            quad(p(0., 0., 1.), p(1., 0., 1.), p(1., 1., 1.), p(0., 1., 1.)),
            quad(p(0., 0., 0.), p(1., 0., 0.), p(1., 0., 1.), p(0., 0., 1.)),
            quad(p(0., 1., 0.), p(0., 1., 1.), p(1., 1., 1.), p(1., 1., 0.)),
            quad(p(0., 0., 0.), p(0., 0., 1.), p(0., 1., 1.), p(0., 1., 0.)),
            quad(p(1., 0., 0.), p(1., 1., 0.), p(1., 1., 1.), p(1., 0., 1.)),
        ]
        .concat()
    }

    /// An n×n grid of unit quads in z = 0, folded up along x = fold if given.
    fn sheet(n: usize, fold: Option<usize>) -> Vec<[Vec3; 3]> {
        let at = |x: usize, y: usize| match fold {
            Some(f) if x > f => Vec3::new(f as f32, y as f32, (x - f) as f32),
            _ => Vec3::new(x as f32, y as f32, 0.0),
        };
        let mut triangles = Vec::new();
        for y in 0..n {
            for x in 0..n {
                triangles.extend(quad(at(x, y), at(x + 1, y), at(x + 1, y + 1), at(x, y + 1)));
            }
        }
        triangles
    }

    #[test]
    fn faces_split_and_flat_sheets_stay_whole() {
        let charts = chart(&cube());
        assert_eq!(charts.extents.len(), 6);
        for island in 0..6 {
            assert_eq!(charts.island.iter().filter(|&&i| i == island).count(), 2);
            let e = charts.extents[island as usize];
            assert!((e - Vec2::ONE).abs().max_element() < 1.0e-5, "{e}");
        }
        assert_eq!(chart(&sheet(8, None)).extents.len(), 1);
        assert_eq!(chart(&sheet(8, Some(4))).extents.len(), 2);
    }

    #[test]
    fn islands_stay_within_their_cone_and_keep_lengths() {
        let triangles = sheet(6, Some(3));
        let charts = chart(&triangles);
        for (t, corners) in triangles.iter().enumerate() {
            let uv = charts.uv[t];
            // Each island here is flat, so projection keeps every edge length.
            for i in 0..3 {
                let world = corners[i].distance(corners[(i + 1) % 3]);
                assert!((uv[i].distance(uv[(i + 1) % 3]) - world).abs() < 1.0e-4);
            }
            let extent = charts.extents[charts.island[t] as usize];
            assert!(
                uv.iter()
                    .all(|p| p.cmpge(-Vec2::splat(1e-5)).all() && p.cmple(extent + 1e-5).all())
            );
        }
    }

    #[test]
    fn halves_round_to_nearest() {
        assert_eq!(half(0.0), 0);
        assert_eq!(half(1.0), 0x3c00);
        assert_eq!(half(0.5), 0x3800);
        assert_eq!(half(2.5), 0x4100);
        assert_eq!(half(1.0 + 1.0 / 2048.0 + 1.0e-6), 0x3c01);
        assert_eq!(half(1.0e9), 0x7bff);
    }

    #[test]
    fn packed_rects_never_overlap() {
        let sizes: Vec<[u32; 2]> = (0..500u32)
            .map(|i| [2 + (i * 7) % 64, 2 + (i * 13) % 40])
            .collect();
        let (rects, height) = pack(&sizes);
        for (i, a) in rects.iter().enumerate() {
            assert_eq!([a.w, a.h], sizes[i]);
            assert!(a.x + a.w <= ATLAS_WIDTH && a.y + a.h <= height);
            for b in &rects[i + 1..] {
                let apart =
                    a.x + a.w <= b.x || b.x + b.w <= a.x || a.y + a.h <= b.y || b.y + b.h <= a.y;
                assert!(apart, "{a:?} {b:?}");
            }
        }
    }

    #[test]
    fn texels_map_back_onto_their_triangles() {
        let triangles = cube();
        let charts = chart(&triangles);
        let density = 7.0;
        let sizes: Vec<[u32; 2]> = charts
            .extents
            .iter()
            .map(|&e| island_size(e, density).unwrap())
            .collect();
        let (rects, _) = pack(&sizes);
        let rects: Vec<Option<Rect>> = rects.into_iter().map(Some).collect();
        let first = 100;
        let blocks = rasterize(&charts, first, 3, density, &rects);
        let mut seen = vec![false; triangles.len()];
        for (rect, texels) in &blocks {
            for (at, texel) in texels.iter().enumerate() {
                // Dilation reaches every texel the triangles leave.
                assert_eq!(texel.w, 1, "uncovered texel in {rect:?}");
                assert_eq!(texel.y, 3);
                let j = (texel.x - first) as usize;
                seen[j] = true;
                let b = (texel.z & 0xffff) as f32 / 65535.0;
                let c = (texel.z >> 16) as f32 / 65535.0;
                assert!(b + c <= 1.0 + 1.0e-4);
                let [p0, p1, p2] = triangles[j];
                let world = p0 * (1.0 - b - c) + p1 * b + p2 * c;
                // The point lies on the cube's surface.
                let distance = world.min(Vec3::ONE - world).min_element().abs();
                assert!(distance < 1.0e-4, "{world}");
                // Interior texels sit at their own centre.
                let (x, y) = (
                    (at as u32 % rect.w) as f32 + 0.5,
                    (at as u32 / rect.w) as f32 + 0.5,
                );
                let uv = charts.uv[j];
                let texel_uv = (uv[0] * (1.0 - b - c) + uv[1] * b + uv[2] * c) * density;
                let inside = x < rect.w as f32 - 1.0 && y < rect.h as f32 - 1.0;
                if inside {
                    assert!((texel_uv - Vec2::new(x, y)).abs().max_element() < 1.0e-3);
                }
            }
        }
        assert!(seen.iter().all(|&s| s), "every triangle owns a texel");
    }
}
