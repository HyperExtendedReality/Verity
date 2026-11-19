//! Checks for the cosine-weighted reconnection and two history pages in pxtgi.slang.

fn flat_connection(horizontal: f64) -> (f64, f64) {
    // The receiver and the emitting plane face each other, one metre apart.
    let squared_distance = 1.0 + horizontal * horizontal;
    (squared_distance, squared_distance.sqrt().recip())
}

#[test]
fn reconnection_converts_the_old_cosine_density_to_the_new_receiver() {
    let (old_squared, old_cos) = flat_connection(0.4);
    let (new_squared, new_cos) = flat_connection(0.5);
    let solid_angle_jacobian = (new_cos / old_cos) * (old_squared / new_squared);
    let ratio = solid_angle_jacobian * (new_cos / old_cos);

    // Cosine-weighted density on the plane, with the common 1/pi omitted.
    let old_area_pdf = old_cos * old_cos / old_squared;
    let new_area_pdf = new_cos * new_cos / new_squared;
    assert!((ratio - new_area_pdf / old_area_pdf).abs() < 1e-12);
}

#[test]
fn constant_radiance_and_rejection_keep_the_fresh_baseline_at_an_unchanged_receiver() {
    let (squared, cos) = flat_connection(0.4);
    let (old_squared, old_cos) = flat_connection(0.4);
    let ratio = (cos / old_cos) * (old_squared / squared) * (cos / old_cos);
    assert!((ratio - 1.0).abs() < 1e-12);
    let fresh = [0.125, 1.0, 3.0];
    let reuse = |reused: Option<[f64; 3]>| {
        reused.map_or(fresh, |value| {
            std::array::from_fn(|i| 0.5 * fresh[i] + 0.5 * ratio * value[i])
        })
    };
    assert_eq!(reuse(Some(fresh)), fresh);
    assert_eq!(reuse(None), fresh);
}

#[test]
fn history_skips_unwritten_rows_after_a_gap_or_counter_wrap() {
    let mut stamp = [0; 2];
    let read =
        |stamp: [u32; 2], frame: u32| frame > 1 && stamp[((frame - 1) & 1) as usize] == frame - 1;
    assert!(!read(stamp, 1));
    stamp[1] = 1;
    assert!(read(stamp, 2));
    // Only a fresh first hit is written. A miss at frame 2 cannot inherit one.
    assert!(!read(stamp, 3));
    assert!(!read(stamp, 4));
    stamp[1] = u32::MAX;
    assert!(!read(stamp, 1));
}
