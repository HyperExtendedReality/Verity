//! Curves through control points, and the arc-length polylines everything
//! else (lanes, sidewalks, frontages) is measured along.

use glam::{Vec2, Vec3};

/// Points along a uniform Catmull-Rom curve through `controls`, about
/// `spacing` apart. The ends are mirrored, so the curve starts and ends on them.
pub fn catmull_rom(controls: &[Vec3], spacing: f32) -> Vec<Vec3> {
    let Some(&first) = controls.first() else {
        return Vec::new();
    };
    let mut points = vec![first];
    for i in 0..controls.len().saturating_sub(1) {
        let (p1, p2) = (controls[i], controls[i + 1]);
        let p0 = if i > 0 { controls[i - 1] } else { 2.0 * p1 - p2 };
        let p3 = controls.get(i + 2).copied().unwrap_or(2.0 * p2 - p1);
        let steps = (p1.distance(p2) / spacing.max(0.01)).ceil().clamp(1.0, 4096.0) as usize;
        for step in 1..=steps {
            let t = step as f32 / steps as f32;
            let (t2, t3) = (t * t, t * t * t);
            points.push(
                0.5 * (2.0 * p1
                    + (p2 - p0) * t
                    + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
                    + (3.0 * p1 - p0 - 3.0 * p2 + p3) * t3),
            );
        }
    }
    points
}

/// A cubic Bézier from `a` heading `da` to `b` arriving along `db`, as points.
pub fn bend(a: Vec3, da: Vec3, b: Vec3, db: Vec3, steps: usize) -> Vec<Vec3> {
    let reach = a.distance(b) * 0.4;
    let (c1, c2) = (a + da * reach, b - db * reach);
    (0..=steps)
        .map(|i| {
            let t = i as f32 / steps as f32;
            let u = 1.0 - t;
            a * u * u * u + c1 * 3.0 * u * u * t + c2 * 3.0 * u * t * t + b * t * t * t
        })
        .collect()
}

/// The left of a heading, in the ground plane (Z is up).
pub fn left(direction: Vec3) -> Vec3 {
    Vec3::new(-direction.y, direction.x, 0.0).normalize_or_zero()
}

/// A polyline measured by arc length.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Path {
    pub points: Vec<Vec3>,
    /// Distance along the path to each point; starts at 0.
    pub lengths: Vec<f32>,
}

impl Path {
    /// Drops repeated points, which have no direction.
    pub fn new(points: impl IntoIterator<Item = Vec3>) -> Self {
        let mut path = Self::default();
        for p in points {
            match path.points.last() {
                Some(&last) if last.distance_squared(p) < 1e-6 => {}
                Some(&last) => {
                    path.lengths.push(path.lengths.last().unwrap() + last.distance(p));
                    path.points.push(p);
                }
                None => {
                    path.lengths.push(0.0);
                    path.points.push(p);
                }
            }
        }
        path
    }

    pub fn length(&self) -> f32 {
        self.lengths.last().copied().unwrap_or(0.0)
    }

    /// The point `s` along, clamped to the ends, and the unit heading there.
    pub fn at(&self, s: f32) -> (Vec3, Vec3) {
        match self.points.len() {
            0 => (Vec3::ZERO, Vec3::X),
            1 => (self.points[0], Vec3::X),
            n => {
                let i = self.lengths.partition_point(|&l| l <= s).clamp(1, n - 1);
                let (a, b) = (self.points[i - 1], self.points[i]);
                let span = self.lengths[i] - self.lengths[i - 1];
                let t = ((s - self.lengths[i - 1]) / span).clamp(0.0, 1.0);
                (a.lerp(b, t), (b - a) / span)
            }
        }
    }

    /// The heading leaving the start, and arriving at the end.
    pub fn ends(&self) -> (Vec3, Vec3) {
        (self.at(0.0).1, self.at(self.length()).1)
    }

    /// The distance along the path nearest `p` in the ground plane, and how far
    /// `p` is from it there.
    pub fn nearest(&self, p: Vec3) -> (f32, f32) {
        let p = p.truncate();
        let mut best = (0.0, f32::INFINITY);
        for i in 1..self.points.len() {
            let (a, b) = (self.points[i - 1].truncate(), self.points[i].truncate());
            let ab = b - a;
            let t = ((p - a).dot(ab) / ab.length_squared().max(1e-9)).clamp(0.0, 1.0);
            let d = p.distance(a + ab * t);
            if d < best.1 {
                best = (self.lengths[i - 1] + t * (self.lengths[i] - self.lengths[i - 1]), d);
            }
        }
        best
    }

    /// The path moved `distance` to its left (negative: right) in the ground
    /// plane, each point along the mean of its neighbouring segments' normals.
    pub fn offset(&self, distance: f32) -> Path {
        let n = self.points.len();
        Path::new((0..n).map(|i| {
            let before = self.points[i.saturating_sub(1)];
            let after = self.points[(i + 1).min(n - 1)];
            let normal = left(after - before);
            // Keep the offset's width through a bend, within reason.
            let incoming = left(self.points[i] - before);
            let scale = if i == 0 || i == n - 1 {
                1.0
            } else {
                1.0 / normal.dot(incoming).max(0.5)
            };
            self.points[i] + normal * distance * scale
        }))
    }

    /// The part of the path from `start` along it to `end` short of its far end.
    pub fn trimmed(&self, start: f32, end: f32) -> Path {
        let (from, to) = (start, self.length() - end);
        if to - from < 0.05 {
            let (mid, _) = self.at((from + to) * 0.5);
            return Path::new([mid]);
        }
        let inside = (self.points.iter().zip(&self.lengths))
            .filter(|&(_, &l)| l > from && l < to)
            .map(|(&p, _)| p);
        Path::new(
            std::iter::once(self.at(from).0)
                .chain(inside)
                .chain([self.at(to).0]),
        )
    }

    pub fn reversed(&self) -> Path {
        Path::new(self.points.iter().rev().copied())
    }

    /// Centre and radius of a circle in the ground plane holding the path.
    pub fn circle(&self) -> (Vec2, f32) {
        let (min, max) = (self.points.iter()).fold((Vec2::MAX, Vec2::MIN), |(lo, hi), p| {
            (lo.min(p.truncate()), hi.max(p.truncate()))
        });
        ((min + max) * 0.5, (max - min).length() * 0.5)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn straight_controls_give_a_straight_curve_of_their_length() {
        let controls = [Vec3::ZERO, Vec3::new(10.0, 0.0, 0.0), Vec3::new(30.0, 0.0, 0.0)];
        let path = Path::new(catmull_rom(&controls, 1.0));
        assert!((path.length() - 30.0).abs() < 1e-3);
        assert!(path.points.iter().all(|p| p.y.abs() < 1e-5));
        let (p, d) = path.at(20.0);
        assert!(p.distance(Vec3::new(20.0, 0.0, 0.0)) < 0.2);
        assert!(d.distance(Vec3::X) < 1e-4);
    }

    #[test]
    fn a_circle_through_its_controls_measures_its_circumference() {
        let r = 50.0;
        let controls: Vec<_> = (0..=16)
            .map(|i| {
                let a = i as f32 / 16.0 * std::f32::consts::TAU;
                Vec3::new(r * a.cos(), r * a.sin(), 0.0)
            })
            .collect();
        let length = Path::new(catmull_rom(&controls, 0.5)).length();
        let expected = std::f32::consts::TAU * r;
        assert!((length - expected).abs() / expected < 0.005, "{length} vs {expected}");
    }

    #[test]
    fn offsets_trims_and_nearest_points() {
        let path = Path::new([Vec3::ZERO, Vec3::new(10.0, 0.0, 0.0)]);
        let left = path.offset(2.0);
        assert!(left.points.iter().all(|p| (p.y - 2.0).abs() < 1e-5));
        let trimmed = path.trimmed(3.0, 2.0);
        assert!((trimmed.length() - 5.0).abs() < 1e-5);
        let (s, d) = path.nearest(Vec3::new(4.0, -3.0, 9.0));
        assert!((s - 4.0).abs() < 1e-5 && (d - 3.0).abs() < 1e-5);
        assert_eq!(path.reversed().at(0.0).0, Vec3::new(10.0, 0.0, 0.0));
    }
}
