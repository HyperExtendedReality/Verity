//! What applications rely on of the method list.
use verity_core::LightingMethod;
use verity_lux::{
    EXPERIMENTAL_METHODS, Lux, METHODS, Method,
    cascades::settings::{FinalGather, Settings},
    method_info,
    pxtgi::{Pxtgi, PxtgiSettings},
    voxel::ScreenVoxelSettings,
};

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
fn hardware_gi_is_offered_and_waits_for_the_gpu() {
    let info = method_info(LightingMethod::HardwareGi).expect("a method");
    assert!(METHODS.iter().any(|m| m.lighting == LightingMethod::HardwareGi));
    assert!(!info.name.is_empty());
    let lux = Lux::for_lighting(LightingMethod::HardwareGi).expect("a method");
    // The profile, and so the method, is chosen when the renderer lends the GPU.
    assert!(lux.hardware_profile().is_none());
}

#[test]
fn methods_are_told_apart() {
    let methods: Vec<_> = METHODS.iter().chain(EXPERIMENTAL_METHODS).collect();
    for (i, a) in methods.iter().enumerate() {
        assert!(!a.name.is_empty() && !a.about.is_empty());
        for b in &methods[i + 1..] {
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

#[test]
fn experimental_presets_install_with_only_the_selected_option_enabled() {
    let split = Settings::budgeted();
    let screen_voxel = ScreenVoxelSettings::default();
    let residual = ScreenVoxelSettings::residual();
    assert_eq!(split.final_gather, FinalGather::Probes);
    assert!(!split.surface_cold_priority && !split.emitter_low_discrepancy);
    assert!(!screen_voxel.coarse_occupancy && !residual.coarse_occupancy);

    let presets = [
        (
            LightingMethod::SplitColdPriority,
            Method::SplitRadianceCascades(Settings {
                surface_cold_priority: true,
                ..split
            }),
        ),
        (
            LightingMethod::SplitLowDiscrepancyEmitters,
            Method::SplitRadianceCascades(Settings {
                emitter_low_discrepancy: true,
                ..split
            }),
        ),
        (
            LightingMethod::SplitCosineGather,
            Method::SplitRadianceCascades(Settings {
                final_gather: FinalGather::Cosine,
                ..split
            }),
        ),
        (
            LightingMethod::CascadeGuided,
            Method::SplitRadianceCascades(Settings {
                final_gather: FinalGather::Guided,
                ..split
            }),
        ),
        (
            LightingMethod::ScreenVoxelBricks,
            Method::ScreenVoxel(ScreenVoxelSettings {
                coarse_occupancy: true,
                ..screen_voxel
            }),
        ),
        (
            LightingMethod::ResidualBricks,
            Method::ScreenVoxel(ScreenVoxelSettings {
                coarse_occupancy: true,
                ..residual
            }),
        ),
        (
            LightingMethod::PxtgiTemporalDiffuse,
            Method::Pxtgi(PxtgiSettings {
                temporal_diffuse_reuse: true,
                ..PxtgiSettings::default()
            }),
        ),
        (
            LightingMethod::SplitSurfaceBounceGuide,
            Method::SplitRadianceCascades(Settings {
                surface_bounce_guide: true,
                ..split
            }),
        ),
        (
            LightingMethod::ScreenVoxelSourceHistory,
            Method::ScreenVoxel(ScreenVoxelSettings {
                source_aware_history: true,
                ..screen_voxel
            }),
        ),
        (
            LightingMethod::ResidualSourceHistory,
            Method::ScreenVoxel(ScreenVoxelSettings {
                source_aware_history: true,
                ..residual
            }),
        ),
        (
            LightingMethod::UltraPbrMsgi,
            Method::SplitRadianceCascades(Settings {
                final_gather: FinalGather::MaterialSpace,
                ..split
            }),
        ),
    ];
    assert_eq!(EXPERIMENTAL_METHODS.len(), presets.len());
    for (lighting, expected) in presets {
        let info = method_info(lighting).expect("an experimental preset");
        assert!(METHODS.iter().all(|info| info.lighting != lighting));
        assert!(Lux::for_lighting(lighting).is_some());
        match ((info.method)(), expected) {
            (Method::SplitRadianceCascades(actual), Method::SplitRadianceCascades(expected)) => {
                assert_eq!(actual, expected);
            }
            (Method::ScreenVoxel(actual), Method::ScreenVoxel(expected)) => {
                assert_eq!(actual, expected);
            }
            (Method::Pxtgi(actual), Method::Pxtgi(expected)) => {
                assert!(!PxtgiSettings::default().temporal_diffuse_reuse);
                assert_eq!(
                    actual.temporal_diffuse_reuse,
                    expected.temporal_diffuse_reuse
                );
                assert_eq!(actual.max_distance, 0.0);
            }
            _ => panic!("wrong backend for {lighting:?}"),
        }
    }
}
