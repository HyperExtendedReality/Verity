//! What applications rely on of the method list.
use verity_core::LightingMethod;
use verity_lux::{Lux, METHODS, Method, method_info, pxtgi::Pxtgi};

#[test]
fn every_gi_lighting_method_has_a_method() {
    for lighting in [
        LightingMethod::SplitRadianceCascades,
        LightingMethod::Pxtgi,
        LightingMethod::ScreenVoxel,
        LightingMethod::Residual,
    ] {
        let info = method_info(lighting).expect("a method");
        assert_eq!(info.lighting, lighting);
        assert!(Lux::for_lighting(lighting).is_some());
    }
    for lighting in [
        LightingMethod::DirectOnly,
        LightingMethod::DirectAndHemisphere,
    ] {
        assert!(method_info(lighting).is_none());
        assert!(Lux::for_lighting(lighting).is_none());
    }
}

#[test]
fn methods_are_told_apart() {
    for (i, a) in METHODS.iter().enumerate() {
        assert!(!a.name.is_empty() && !a.about.is_empty());
        for b in &METHODS[i + 1..] {
            assert_ne!(a.lighting, b.lighting);
            assert_ne!(a.name, b.name);
        }
    }
}

#[test]
fn lux_runs_the_chosen_method() {
    let mut pxtgi = Lux::for_lighting(LightingMethod::Pxtgi).expect("a method");
    assert!(pxtgi.follows_physics());
    assert!(pxtgi.method_mut::<Pxtgi>().is_some());
    let mut cascades = Lux::for_lighting(LightingMethod::SplitRadianceCascades).expect("a method");
    assert!(!cascades.follows_physics());
    assert!(cascades.method_mut::<Pxtgi>().is_none());
    let chosen = (METHODS[1].method)();
    assert!(matches!(chosen, Method::Pxtgi(_)));
    let residual = method_info(LightingMethod::Residual).expect("a method");
    assert!(matches!((residual.method)(), Method::ScreenVoxel(s) if s.residual));
}
