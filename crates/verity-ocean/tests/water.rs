use glam::{Vec2, Vec3};
use std::f32::consts::{PI, TAU};
use verity_core::{Entity, Transform, World};
use verity_ocean::*;
use verity_physics::*;

struct Flat(f32);

impl Water for Flat {
    fn sample(&self, _: Vec2) -> WaterSample {
        WaterSample {
            height: self.0,
            velocity: Vec3::ZERO,
        }
    }
}

#[test]
fn dispersion_has_deep_and_shallow_limits() {
    let k = 0.5;
    assert!((angular_frequency(k, f32::INFINITY) - (GRAVITY * k).sqrt()).abs() < 1e-5);
    // Shallow: ω ≈ k √(g h).
    assert!((angular_frequency(0.01, 1.0) / (0.01 * GRAVITY.sqrt()) - 1.0).abs() < 1e-3);
    assert!(
        (group_velocity(k, f32::INFINITY) - 0.5 * angular_frequency(k, f32::INFINITY) / k).abs()
            < 1e-5
    );
    for depth in [0.5, 3.0, 40.0] {
        let omega = 1.3;
        assert!((angular_frequency(wavenumber(omega, depth), depth) - omega).abs() < 1e-4);
    }
}

#[test]
fn ocean_height_variance_matches_its_spectrum() {
    let settings = OceanSettings {
        resolution: 256,
        cascades: vec![1000.0, 100.0, 10.0],
        spectra: vec![Spectrum::default()],
        ..OceanSettings::default()
    };
    let spectrum = &settings.spectra[0];
    // ∫S(ω) dω by midpoint rule.
    let dw = 0.001;
    let m0: f32 = (1..20_000)
        .map(|i| spectrum.frequency_density(i as f32 * dw, f32::INFINITY) * dw)
        .sum();
    let ocean = Ocean::new(settings);
    let variance: f32 = ocean
        .cascades()
        .iter()
        .map(|c| {
            c.displacement.iter().map(|d| d.z * d.z).sum::<f32>() / c.displacement.len() as f32
        })
        .sum();
    let (measured, expected) = (4.0 * variance.sqrt(), 4.0 * m0.sqrt());
    assert!(
        (measured / expected - 1.0).abs() < 0.25,
        "Hs {measured} vs {expected}"
    );
    // Hasselmann fetch law for 10 m/s over 100 km: about 1.6 m.
    assert!((1.0..2.5).contains(&expected), "Hs {expected}");
}

#[test]
fn ocean_moves_and_samples_consistently() {
    let mut ocean = Ocean::new(OceanSettings::default());
    ocean.update(1.0 / 30.0);
    let p = Vec2::new(12.3, -4.5);
    let s = ocean.sample(p);
    assert!(s.height.is_finite() && s.velocity.is_finite() && s.velocity != Vec3::ZERO);
    assert!(s.height.abs() < 5.0);
    let foam = ocean.foam(p);
    assert!((0.0..=1.0).contains(&foam));
}

/// A beach sloping up along +X: 20 m deep at x = 0, dry past x = 1000, and
/// 1 km alongshore so oblique rays reach the middle from the offshore edge.
fn beach() -> Seabed {
    let mut seabed = Seabed::flat(240, 200, 5.0, 0.0);
    for y in 0..seabed.height {
        for x in 0..seabed.width {
            seabed.elevation[x + y * seabed.width] = -20.0 + seabed.cell_center(x, y).x * 0.02;
        }
    }
    seabed
}

#[test]
fn shore_waves_refract_by_snells_law() {
    let settings = ShoreSettings {
        direction: 30f32.to_radians(),
        ..ShoreSettings::default()
    };
    let omega = TAU / settings.period;
    let shore = ShoreWaves::new(&beach(), 0.0, settings);
    let at_depth = |d: f32| Vec2::new((20.0 - d) / 0.02, 800.0);
    let (k20, k5) = (wavenumber(omega, 20.0), wavenumber(omega, 5.0));
    let predicted = (30f32.to_radians().sin() * k20 / k5).asin();
    let got = shore.local_wave(at_depth(5.0)).direction;
    assert!(
        (got.y.asin() - predicted).abs() < 0.03,
        "angle {} vs {predicted}",
        got.y.asin()
    );
}

#[test]
fn shore_waves_shoal_then_break() {
    let shore = ShoreWaves::new(&beach(), 0.0, ShoreSettings::default());
    let at = |d: f32| shore.local_wave(Vec2::new((20.0 - d) / 0.02, 100.0));
    assert!(
        at(3.0).height > at(15.0).height * 1.05,
        "shoaling raises waves"
    );
    assert_eq!(at(15.0).breaking, 0.0);
    let surf = at(1.0);
    assert!(
        (surf.height - 0.78).abs() < 0.03 && surf.breaking == 1.0,
        "{surf:?}"
    );
    let mut shore = shore;
    shore.set_time(2.0);
    let s = shore.sample(Vec2::new(900.0, 100.0));
    assert!(s.height.is_finite() && s.height.abs() < 1.0);
}

#[test]
fn breaker_profile_is_periodic_and_has_phase_derivatives() {
    let profile = BreakerProfile::default();
    let phase = 0.73;
    let b = 0.8;
    let shape = profile.evaluate(phase, b);
    let periodic = profile.evaluate(phase + TAU, b);
    assert!((shape.along - periodic.along).abs() < 1e-5);
    assert!((shape.up - periodic.up).abs() < 1e-5);
    let eps = 0.001;
    let before = profile.evaluate(phase - eps, b);
    let after = profile.evaluate(phase + eps, b);
    assert!(((after.along - before.along) / (2.0 * eps) - shape.along_derivative).abs() < 1e-3);
    assert!(((after.up - before.up) / (2.0 * eps) - shape.up_derivative).abs() < 1e-3);
    let unbroken = profile.evaluate(phase, 0.0);
    assert!((unbroken.along + phase.sin()).abs() < 1e-6);
    assert!((unbroken.up - phase.cos()).abs() < 1e-6);
}

#[test]
fn changing_breaker_profile_updates_the_cpu_surface() {
    let seabed = Seabed::flat(32, 8, 1.0, -1.0);
    let mut shore = ShoreWaves::new(&seabed, 0.0, ShoreSettings::default());
    let point = seabed.cell_center(12, 3);
    let original = shore.displacement_at_rest(point).0;
    let profile = BreakerProfile {
        horizontal_second_harmonic: 0.0,
        vertical_second_harmonic: 0.0,
    };
    shore.set_profile(profile);
    assert_eq!(shore.profile(), profile);
    assert!((shore.displacement_at_rest(point).0 - original).length() > 0.01);
}

#[test]
fn breaker_grid_reconstructs_the_cpu_surface_and_velocity() {
    let seabed = Seabed::flat(32, 8, 1.0, -1.0);
    let mut shore = ShoreWaves::new(&seabed, 0.0, ShoreSettings::default());
    shore.set_time(1.3);
    let p = seabed.cell_center(12, 3);
    let grid = shore.breaker_grid();
    let i = 12 + 3 * grid.width;
    assert_eq!(grid.height, seabed.height);
    assert_eq!(grid.origin, seabed.origin);
    assert_eq!(grid.cell, seabed.cell);
    let b = grid.breaking[i];
    assert_eq!(b, 1.0);
    let shape = shore
        .profile()
        .evaluate(grid.phase[i] - grid.omega * grid.time, b);
    let vertical = grid.wave_height[i] * 0.5;
    let horizontal =
        (vertical / (grid.wavenumber[i] * grid.depth[i]).tanh()).min(0.9 / grid.wavenumber[i]);
    let (s, c) = (grid.lean * b).sin_cos();
    let along = horizontal * shape.along * c + vertical * shape.up * s;
    let up = vertical * shape.up * c - horizontal * shape.along * s;
    let expected = (grid.direction[i] * along).extend(up);
    let (displacement, velocity) = shore.displacement_at_rest(p);
    assert!((expected - displacement).length() < 1e-5);

    let eps = 0.001;
    shore.set_time(1.3 - eps);
    let before = shore.displacement_at_rest(p).0;
    shore.set_time(1.3 + eps);
    let after = shore.displacement_at_rest(p).0;
    assert!(((after - before) / (2.0 * eps) - velocity).length() < 1e-3);
    shore.set_time(1.3 + shore.settings().period);
    assert!((shore.displacement_at_rest(p).0 - displacement).length() < 1e-5);
}

#[test]
fn breaker_can_overturn_while_gameplay_returns_one_finite_sample() {
    let seabed = Seabed::flat(128, 4, 0.2, -1.0);
    let mut shore = ShoreWaves::new(&seabed, 0.0, ShoreSettings::default());
    let p = seabed.cell_center(64, 2);
    let grid = shore.breaker_grid();
    let i = 64 + 2 * grid.width;
    let time_at_crest = grid.phase[i] / grid.omega;
    shore.set_time(time_at_crest);
    let delta = Vec2::new(0.001, 0.0);
    let left = p - delta + shore.displacement_at_rest(p - delta).0.truncate();
    let right = p + delta + shore.displacement_at_rest(p + delta).0.truncate();
    assert!(right.x < left.x, "parametric crest must fold forward");
    let sample = shore.sample(p);
    assert!(sample.height.is_finite() && sample.velocity.is_finite());
}

#[test]
fn breaker_grid_preserves_signed_depth_and_dry_surface_is_still() {
    let mut seabed = Seabed::flat(4, 2, 1.0, -0.01);
    seabed.elevation[3] = 0.5;
    let shore = ShoreWaves::new(&seabed, 0.0, ShoreSettings::default());
    let grid = shore.breaker_grid();
    assert!((grid.depth[0] - 0.01).abs() < 1e-6);
    assert_eq!(grid.depth[3], -0.5);
    let (dry_displacement, dry_velocity) = shore.displacement_at_rest(seabed.cell_center(3, 0));
    assert_eq!(dry_displacement, Vec3::ZERO);
    assert_eq!(dry_velocity, Vec3::ZERO);
    let (shallow_displacement, shallow_velocity) =
        shore.displacement_at_rest(seabed.cell_center(0, 0));
    assert!(shallow_displacement.is_finite() && shallow_velocity.is_finite());
}

fn total(water: &ShallowWater) -> f32 {
    water.volume()
}

#[test]
fn lake_at_rest_stays_at_rest_over_bumps_and_islands() {
    let mut seabed = Seabed::flat(32, 32, 1.0, 0.0);
    for y in 0..32 {
        for x in 0..32 {
            seabed.elevation[x + y * 32] =
                -2.0 + 2.5 * ((x as f32 * 0.4).sin() * (y as f32 * 0.3).cos()).max(0.0);
        }
    }
    let mut water = ShallowWater::new(&seabed, 0.0);
    let before = water.depth.clone();
    for _ in 0..50 {
        water.step(0.05, &ShallowWaterSettings::default());
    }
    let flow = water
        .u
        .iter()
        .chain(&water.v)
        .fold(0.0f32, |m, v| m.max(v.abs()));
    let drift = water
        .depth
        .iter()
        .zip(&before)
        .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
    assert!(flow < 1e-4 && drift < 1e-4, "flow {flow}, drift {drift}");
}

#[test]
fn dam_break_conserves_water_over_dry_ground() {
    let mut seabed = Seabed::flat(64, 64, 0.5, 0.0);
    for y in 0..64 {
        for x in 0..64 {
            seabed.elevation[x + y * 64] = if x < 20 { -2.0 } else { -0.2 + 0.01 * x as f32 };
        }
    }
    let mut water = ShallowWater::new(&seabed, 0.0);
    let before = total(&water);
    for _ in 0..100 {
        water.step(0.05, &ShallowWaterSettings::default());
    }
    assert!(
        (total(&water) / before - 1.0).abs() < 1e-3,
        "{} vs {before}",
        total(&water)
    );
    assert!(water.depth.iter().all(|h| h.is_finite() && *h >= 0.0));
}

/// Time for a standing wave 8 m long in 20 m of water to first cross zero:
/// a quarter period.
fn quarter_period(dispersion: bool) -> f32 {
    let n = 64;
    let mut water = ShallowWater::new(&Seabed::flat(n, n, 1.0, -20.0), 0.0);
    let k = TAU / 8.0;
    for c in 0..n * n {
        water.depth[c] += 0.05 * (k * ((c % n) as f32 + 0.5)).cos();
    }
    let settings = ShallowWaterSettings {
        dispersion,
        ..ShallowWaterSettings::default()
    };
    let dt = 0.005;
    for step in 1..400 {
        water.step(dt, &settings);
        if water.depth[0] < 20.0 {
            return step as f32 * dt;
        }
    }
    f32::INFINITY
}

#[test]
fn surface_waves_travel_at_airy_speed() {
    let k = TAU / 8.0;
    let airy = TAU / angular_frequency(k, 20.0) / 4.0;
    let shallow = TAU / (k * (GRAVITY * 20.0).sqrt()) / 4.0;
    let dispersive = quarter_period(true);
    assert!(
        (dispersive / airy - 1.0).abs() < 0.1,
        "{dispersive} vs Airy {airy}"
    );
    let plain = quarter_period(false);
    assert!(
        (plain / shallow - 1.0).abs() < 0.2,
        "{plain} vs shallow water {shallow}"
    );
}

#[test]
fn modifiers_add_water_and_edges_follow_their_target() {
    let mut water = ShallowWater::new(&Seabed::flat(32, 32, 1.0, -1.0), 0.0);
    let before = total(&water);
    water.add_water(Vec2::splat(16.0), 4.0, 10.0);
    assert!((total(&water) - before - 10.0).abs() < 1e-3);
    water.push(Vec2::splat(16.0), 4.0, Vec2::new(1.0, 0.0), 1.0);
    water.step(0.1, &ShallowWaterSettings::default());
    assert!(water.sample(Vec2::splat(17.0)).velocity.x > 0.1);
    for _ in 0..20 {
        water.relax_edges(&Flat(0.5), 4.0);
        water.step(0.05, &ShallowWaterSettings::default());
    }
    assert!((water.sample(Vec2::new(0.5, 16.0)).height - 0.5).abs() < 0.05);
}

#[test]
fn buoyancy_lifts_by_displaced_weight() {
    let gravity = Vec3::new(0.0, 0.0, -9.81);
    let fluid = Fluid::default();
    let cube = PhysicsBody::new(
        RigidBody::Dynamic,
        Collider::Cuboid {
            half_extents: Vec3::splat(0.5),
        },
    );
    let half_under = buoyancy(&Flat(0.0), &Transform::default(), &cube, &fluid, gravity);
    assert!((half_under.force.z / (fluid.density * 9.81 * 0.5) - 1.0).abs() < 1e-3);
    assert!(half_under.torque.length() < 1e-3);
    let sphere = PhysicsBody::new(RigidBody::Dynamic, Collider::Sphere { radius: 1.0 });
    let f = buoyancy(&Flat(0.0), &Transform::default(), &sphere, &fluid, gravity);
    assert!((f.force.z / (fluid.density * 9.81 * 2.0 / 3.0 * PI) - 1.0).abs() < 1e-3);
    assert_eq!(
        buoyancy(&Flat(-5.0), &Transform::default(), &sphere, &fluid, gravity).force,
        Vec3::ZERO
    );
}

#[test]
fn light_box_floats_at_its_waterline() {
    let mut world = World::new();
    let mut physics = PhysicsWorld::default();
    let settings = PhysicsSettings::default();
    let entity = world.spawn(Entity {
        name: "crate".into(),
        transform: Transform {
            translation: [0.0, 0.0, 2.0],
            ..Default::default()
        },
        ..Entity::default()
    });
    let mut body = PhysicsBody::new(
        RigidBody::Dynamic,
        Collider::Cuboid {
            half_extents: Vec3::splat(0.5),
        },
    );
    body.material.density = 500.0;
    physics.insert(&world, entity, body).unwrap();
    let water = Flat(0.0);
    for _ in 0..600 {
        let transform = world.get(entity).unwrap().transform.clone();
        let body = physics.get_mut(entity).unwrap();
        body.force = buoyancy(
            &water,
            &transform,
            body,
            &Fluid::default(),
            settings.gravity,
        );
        physics.step(&mut world, &settings, 1.0 / 60.0);
    }
    // Floats with 500/1025 of its height under water.
    let z = world.get(entity).unwrap().transform.translation[2];
    let expected = 0.5 - 500.0 / 1025.0;
    assert!((z - expected).abs() < 0.03, "z {z} vs {expected}");
}

#[test]
fn ripples_spread_and_layer_over_water() {
    let mut ripples = Ripples::new(RipplesSettings::default());
    ripples.follow(Vec2::new(10.0, 10.0));
    ripples.disturb(Vec2::new(10.0, 10.0), 0.3, 0.05);
    ripples.step(0.5);
    let ring = ripples.sample(Vec2::new(10.2, 10.0)).height;
    assert!(ring.abs() > 1e-4);
    ripples.follow(Vec2::new(10.5, 10.0));
    let scrolled = ripples.sample(Vec2::new(10.2, 10.0)).height;
    assert!(
        (scrolled - ring).abs() < 1e-6,
        "scrolling keeps ripples in place"
    );
    let layered = (Flat(1.0), &ripples).sample(Vec2::new(10.2, 10.0));
    assert_eq!(layered.height, 1.0 + ring);
}
