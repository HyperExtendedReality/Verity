//! Speed and accuracy of each water model. `cargo run -p verity-ocean --release --example water_bench`

use glam::Vec2;
use std::hint::black_box;
use std::time::Instant;
use verity_ocean::*;

/// Median seconds per call, after a warm-up.
fn time(mut f: impl FnMut()) -> f64 {
    f();
    let mut runs: Vec<f64> = (0..15)
        .map(|_| {
            let start = Instant::now();
            f();
            start.elapsed().as_secs_f64()
        })
        .collect();
    runs.sort_by(f64::total_cmp);
    runs[runs.len() / 2]
}

fn report(name: &str, seconds: f64) {
    println!("{name:<34}{:>12.3} ms", seconds * 1e3);
}

fn ocean_speed() {
    for resolution in [128, 256, 512] {
        let settings = OceanSettings {
            resolution,
            ..OceanSettings::default()
        };
        let start = Instant::now();
        let mut ocean = Ocean::new(settings);
        report(
            &format!("Ocean::new {resolution}"),
            start.elapsed().as_secs_f64(),
        );
        let mut t = 0.0;
        report(
            &format!("Ocean::update {resolution}"),
            time(|| {
                t += 1.0 / 60.0;
                ocean.update(t);
            }),
        );
    }
    let mut ocean = Ocean::new(OceanSettings::default());
    ocean.update(1.0 / 60.0);
    report(
        "Ocean::sample x1000",
        time(|| {
            for i in 0..1000 {
                black_box(ocean.sample(Vec2::new(i as f32 * 0.37, i as f32 * 0.11)));
            }
        }),
    );
}

fn slope(n: usize) -> Seabed {
    let mut seabed = Seabed::flat(n, n, 1.0, 0.0);
    for y in 0..n {
        for x in 0..n {
            seabed.elevation[x + y * n] = -10.0 + 12.0 * x as f32 / n as f32;
        }
    }
    seabed
}

fn shallow_speed() {
    for n in [64, 128, 256] {
        for dispersion in [false, true] {
            let mut water = ShallowWater::new(&slope(n), 0.0);
            water.add_water(Vec2::splat(n as f32 * 0.3), 6.0, 40.0);
            let settings = ShallowWaterSettings {
                dispersion,
                ..ShallowWaterSettings::default()
            };
            report(
                &format!("ShallowWater::step {n} disp={dispersion}"),
                time(|| water.step(1.0 / 60.0, &settings)),
            );
        }
    }
    // Blending the open sea in over the edges, as a coast does every frame.
    let mut coast = ShallowWater::new(&slope(256), 0.0);
    let ocean = Ocean::new(OceanSettings::default());
    report(
        "ShallowWater::relax_edges 256",
        time(|| coast.relax_edges(&ocean, 16.0)),
    );
    // Nothing moving: the dispersive layer should cost no FFTs.
    let mut lake = ShallowWater::new(&slope(256), 0.0);
    let settings = ShallowWaterSettings::default();
    report(
        "ShallowWater::step 256 still lake",
        time(|| lake.step(1.0 / 60.0, &settings)),
    );
    // Quarter-metre cells need several bulk substeps per frame.
    let mut seabed = slope(128);
    seabed.cell = 0.25;
    let mut fine = ShallowWater::new(&seabed, 0.0);
    fine.add_water(Vec2::splat(10.0), 2.0, 2.0);
    report(
        "ShallowWater::step 128 cell 0.25",
        time(|| fine.step(1.0 / 60.0, &settings)),
    );
}

fn shore_speed() {
    let n = 512;
    let mut seabed = Seabed::flat(n, n, 2.0, 0.0);
    for y in 0..n {
        for x in 0..n {
            seabed.elevation[x + y * n] = -20.0 + 0.04 * x as f32 + (y as f32 * 0.05).sin();
        }
    }
    let settings = ShoreSettings {
        direction: 0.4,
        ..ShoreSettings::default()
    };
    report(
        "ShoreWaves::new 512",
        time(|| {
            black_box(ShoreWaves::new(&seabed, 0.0, settings.clone()));
        }),
    );
    let shore = ShoreWaves::new(&seabed, 0.0, settings);
    report(
        "ShoreWaves::sample x1000",
        time(|| {
            for i in 0..1000 {
                black_box(shore.sample(Vec2::new(300.0 + i as f32 * 0.3, 400.0)));
            }
        }),
    );
}

fn ripples_speed() {
    let mut ripples = Ripples::new(RipplesSettings::default());
    ripples.disturb(Vec2::ZERO, 0.5, 0.1);
    report("Ripples::step", time(|| ripples.step(1.0 / 60.0)));
}

/// How far a surface query lands from the surface point that truly sits there.
fn ocean_lookup_error() {
    for choppiness in [0.9, 1.3] {
        let mut ocean = Ocean::new(OceanSettings {
            choppiness,
            ..OceanSettings::default()
        });
        ocean.update(40.0);
        let (mut sum, mut worst, mut count) = (0.0f64, 0.0f32, 0);
        for y in 0..200 {
            for x in 0..200 {
                let rest = Vec2::new(x as f32, y as f32) * 0.61;
                let d = ocean.displacement(rest);
                let error = (ocean.sample(rest + d.truncate()).height - d.z).abs();
                sum += (error * error) as f64;
                worst = worst.max(error);
                count += 1;
            }
        }
        println!(
            "lookup error, choppiness {choppiness}: rms {:.2} mm, worst {:.1} mm",
            (sum / count as f64).sqrt() * 1e3,
            worst * 1e3
        );
    }
}

fn whitecap_coverage() {
    for wind_speed in [5.0, 10.0, 20.0] {
        let mut ocean = Ocean::new(OceanSettings {
            spectra: vec![Spectrum {
                wind_speed,
                ..Spectrum::default()
            }],
            ..OceanSettings::default()
        });
        for frame in 0..120 {
            ocean.update(frame as f32 / 30.0);
        }
        let coverage: Vec<String> = ocean
            .cascades()
            .iter()
            .map(|c| {
                let foamy = c.foam.iter().filter(|f| **f > 0.1).count();
                format!("{:.2}%", 100.0 * foamy as f32 / c.foam.len() as f32)
            })
            .collect();
        let points = (0..10_000).map(|i| Vec2::new((i % 100) as f32, (i / 100) as f32) * 2.3);
        let surface = points.filter(|p| ocean.foam(*p) > 0.1).count() as f32 / 100.0;
        // Monahan & Muircheartaigh 1980.
        let observed = 3.84e-4 * wind_speed.powf(3.41);
        println!(
            "whitecaps, wind {wind_speed} m/s: cascades {coverage:?}, surface {surface:.2}% (observed {observed:.2}%)"
        );
    }
}

/// A ring reaches the window edge; whatever is left inside is reflection.
fn ripple_reflection() {
    let settings = RipplesSettings {
        damping: 0.0,
        ..RipplesSettings::default()
    };
    let mut ripples = Ripples::new(settings.clone());
    ripples.disturb(Vec2::ZERO, 0.4, 0.1);
    let energy = |r: &Ripples| {
        let mut sum = 0.0f32;
        for y in -30..30 {
            for x in -30..30 {
                let h = r
                    .sample(Vec2::new(x as f32, y as f32) * settings.cell)
                    .height;
                sum += h * h;
            }
        }
        sum
    };
    let start = energy(&ripples);
    for _ in 0..20 * 60 {
        ripples.step(1.0 / 60.0);
    }
    println!(
        "ripple energy left after 20 s: {:.3}%",
        100.0 * energy(&ripples) / start
    );
}

/// Tows a pressure point at `speed`: the wake half-angle should be 19.47°
/// whatever the speed, which no non-dispersive solver can produce.
fn kelvin_wake() {
    for speed in [1.0f32, 1.5] {
        let settings = RipplesSettings {
            resolution: 512,
            cell: 0.05,
            depth: f32::INFINITY,
            damping: 0.0,
        };
        let cell = settings.cell;
        let mut ripples = Ripples::new(settings);
        let dt = 1.0 / 60.0;
        let start = Vec2::new(-9.0, 0.0);
        let frames = (15.0 / speed / dt) as usize;
        for frame in 0..frames {
            let position = start + Vec2::X * speed * frame as f32 * dt;
            ripples.disturb(position, 0.2, 0.002);
            ripples.step(dt);
        }
        let boat = start + Vec2::X * speed * frames as f32 * dt;
        // Outermost point of the wake at each distance behind the boat.
        let mut angles = vec![];
        for back in [4.0f32, 6.0, 8.0] {
            let at = |y: f32| {
                ripples
                    .sample(boat - Vec2::X * back + Vec2::Y * y)
                    .height
                    .abs()
            };
            let peak = (0..200).map(|i| at(i as f32 * cell)).fold(0.0, f32::max);
            let edge = (0..200).rev().find(|i| at(*i as f32 * cell) > 0.15 * peak);
            angles.push((edge.unwrap_or(0) as f32 * cell / back).atan().to_degrees());
        }
        println!("Kelvin half-angle at {speed} m/s: {angles:.1?} (theory 19.5)");
    }
}

fn main() {
    ocean_speed();
    shallow_speed();
    shore_speed();
    ripples_speed();
    ocean_lookup_error();
    whitecap_coverage();
    ripple_reflection();
    kelvin_wake();
}
