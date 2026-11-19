//! Global illumination for the Verity renderer: several methods, one extension.
//!
//! [`Lux`] is what the renderer installs; it runs one [`Method`]:
//!
//! - [`cascades`], Split Radiance Cascades, which traces a GI scene of its own.
//! - [`pxtgi`], which traces the physics scene.
//! - [`voxel`], which traces the screen, then a voxel copy of the scene, and
//!   as residual GI takes the voxels' light and corrects it by the screen's.
//!
//! The editor offers [`METHODS`] and a separate [`EXPERIMENTAL_METHODS`] group.
//! To add a method, implement [`GiMethod`], add a [`Method`] variant and a registry
//! entry, and give it a `LightingMethod` so scenes can name it. GI follows terrain through
//! `RenderExtension::retained_changed`.
//!
//! `legacy/` keeps the original Bevy source of the cascades as a porting
//! reference; it is not compiled.

pub mod cage;
pub mod cascades;
pub mod holographic;
pub mod pxtgi;
mod pxtgi_accel;
mod rt;
mod shared;
pub mod voxel;

use cascades::{
    SplitRadianceCascades,
    settings::{FinalGather, Settings},
};
use pxtgi::{Capture, Pxtgi, PxtgiSettings};
use std::{
    any::Any,
    time::{Duration, Instant},
};
use verity_core::{LightingMethod, World};
use verity_physics::PhysicsWorld;
use verity_render::{
    FrameContext, Gpu, RenderExtension, RetainedChange, Targets, asset::SceneData,
    extension::Result,
};
use voxel::{ScreenVoxel, ScreenVoxelSettings};

/// How far a method has come towards lighting the scene.
#[derive(Clone, Debug, PartialEq)]
pub enum GiStatus {
    /// The driver is compiling the transport pipelines, in the background.
    Compiling {
        since: Instant,
    },
    Ready {
        compile_time: Duration,
    },
    Failed(String),
}

/// A way of computing global illumination. Everything the renderer needs of one
/// is [`RenderExtension`]; this adds what the application asks of it.
pub trait GiMethod: RenderExtension {
    /// Follow the physics world. Call after stepping it, before drawing.
    fn sync(&mut self, _physics: &PhysicsWorld, _world: &World) {}

    /// Whether [`GiMethod::sync`] does anything, so callers can skip preparing it.
    fn follows_physics(&self) -> bool {
        false
    }

    fn status(&self) -> GiStatus;

    /// How long the GPU took over the method's passes in a recent frame, if it
    /// measures that.
    fn gpu_time(&self) -> Option<Duration> {
        None
    }

    /// Milliseconds per pass in a recent frame, in frame order, if it measures that.
    fn pass_times(&self) -> &[(&'static str, f32)] {
        &[]
    }

    /// Trace with hardware ray queries where the GPU has them, or walk the
    /// method's own tree. Methods that trace nothing ignore it.
    fn set_hardware_rays(&mut self, _on: bool) {}

    /// Read back the next frame's HDR image, if the method can.
    fn request_capture(&mut self) {}

    /// The capture asked for, once the GPU has finished it.
    fn take_capture(&mut self) -> Option<Capture> {
        None
    }
}

/// A [`GiMethod`] to run, with its settings.
pub enum Method {
    SplitRadianceCascades(Settings),
    Pxtgi(PxtgiSettings),
    ScreenVoxel(ScreenVoxelSettings),
}

/// What an application shows of a method.
#[derive(Clone, Copy, Debug)]
pub struct MethodInfo {
    /// How scenes name it.
    pub lighting: LightingMethod,
    pub name: &'static str,
    pub about: &'static str,
    /// The method with its usual settings.
    pub method: fn() -> Method,
}

/// Standard GI methods for applications to offer.
pub const METHODS: &[MethodInfo] = &[
    MethodInfo {
        lighting: LightingMethod::SplitRadianceCascades,
        name: "Split radiance cascades GI",
        about: "Probes in a hierarchy of cascades, traced against a GI scene of its own.",
        method: || Method::SplitRadianceCascades(Settings::budgeted()),
    },
    MethodInfo {
        lighting: LightingMethod::Pxtgi,
        name: "PXTGI (physics traced GI)",
        about: "Rays through the physics scene, which follows bodies as they move.",
        method: || Method::Pxtgi(PxtgiSettings::default()),
    },
    MethodInfo {
        lighting: LightingMethod::ScreenVoxel,
        name: "Screen and voxel GI",
        about: "Rays along the depth buffer, then through a voxel copy of the scene.",
        method: || Method::ScreenVoxel(ScreenVoxelSettings::default()),
    },
    MethodInfo {
        lighting: LightingMethod::Residual,
        name: "Residual GI",
        about: "The voxel copy's settled light, corrected where rays along the depth buffer see it differ.",
        // shortcut: shares ScreenVoxel's passes, flagged by settings; a variant of its own when it needs its own.
        method: || Method::ScreenVoxel(ScreenVoxelSettings::residual()),
    },
];

/// Opt-in presets that reuse the standard methods with experimental settings.
pub const EXPERIMENTAL_METHODS: &[MethodInfo] = &[
    MethodInfo {
        lighting: LightingMethod::SplitColdPriority,
        name: "Split GI + cold cache priority",
        about: "Give new or invalidated surface-cache slots more bounce rays at the same budget.",
        method: || {
            Method::SplitRadianceCascades(Settings {
                surface_cold_priority: true,
                ..Settings::budgeted()
            })
        },
    },
    MethodInfo {
        lighting: LightingMethod::SplitLowDiscrepancyEmitters,
        name: "Split GI + low-discrepancy emitters",
        about: "Use 3D low-discrepancy sampling for emissive light candidates.",
        method: || {
            Method::SplitRadianceCascades(Settings {
                emitter_low_discrepancy: true,
                ..Settings::budgeted()
            })
        },
    },
    MethodInfo {
        lighting: LightingMethod::ScreenVoxelBricks,
        name: "Screen and voxel GI + empty-brick skipping",
        about: "Skip known-empty bricks while tracing through the voxel copy of the scene.",
        method: || {
            Method::ScreenVoxel(ScreenVoxelSettings {
                coarse_occupancy: true,
                ..ScreenVoxelSettings::default()
            })
        },
    },
    MethodInfo {
        lighting: LightingMethod::ResidualBricks,
        name: "Residual GI + empty-brick skipping",
        about: "Use the same empty-brick skipping with Residual GI.",
        method: || {
            Method::ScreenVoxel(ScreenVoxelSettings {
                coarse_occupancy: true,
                ..ScreenVoxelSettings::residual()
            })
        },
    },
    MethodInfo {
        lighting: LightingMethod::SplitCosineGather,
        name: "Split GI + cosine gather",
        about: "Trace a cosine-sampled diffuse gather instead of interpolating probes; a control for guided GI.",
        method: || {
            Method::SplitRadianceCascades(Settings {
                final_gather: FinalGather::Cosine,
                ..Settings::budgeted()
            })
        },
    },
    MethodInfo {
        lighting: LightingMethod::CascadeGuided,
        name: "Cascade-guided GI",
        about: "Trace a diffuse gather guided by the previous frame's cascades; not a default quality upgrade.",
        method: || {
            Method::SplitRadianceCascades(Settings {
                final_gather: FinalGather::Guided,
                ..Settings::budgeted()
            })
        },
    },
    MethodInfo {
        lighting: LightingMethod::PxtgiTemporalDiffuse,
        name: "PXTGI + temporal diffuse hit",
        about: "Try one previous physics-traced hit after validating and re-shading it.",
        method: || {
            Method::Pxtgi(PxtgiSettings {
                temporal_diffuse_reuse: true,
                ..PxtgiSettings::default()
            })
        },
    },
    MethodInfo {
        lighting: LightingMethod::SplitSurfaceBounceGuide,
        name: "Split GI + surface bounce guide",
        about: "Guide the existing surface-cache rays without adding a traced receiver gather.",
        method: || {
            Method::SplitRadianceCascades(Settings {
                surface_bounce_guide: true,
                ..Settings::budgeted()
            })
        },
    },
    MethodInfo {
        lighting: LightingMethod::UltraPbrMsgi,
        name: "Ultra PBR / MSGI",
        about: "Trace diffuse and glossy indirect light in the visible material's shading frame, using Split's surface cache.",
        method: || {
            Method::SplitRadianceCascades(Settings {
                final_gather: FinalGather::MaterialSpace,
                ..Settings::budgeted()
            })
        },
    },
    MethodInfo {
        lighting: LightingMethod::ScreenVoxelSourceHistory,
        name: "Screen and voxel GI + source-aware history",
        about: "Shorten history when the balance of screen and voxel hits changes.",
        method: || {
            Method::ScreenVoxel(ScreenVoxelSettings {
                source_aware_history: true,
                ..ScreenVoxelSettings::default()
            })
        },
    },
    MethodInfo {
        lighting: LightingMethod::ResidualSourceHistory,
        name: "Residual GI + source-aware history",
        about: "Reject stale history when the voxel baseline or screen correction changes source.",
        method: || {
            Method::ScreenVoxel(ScreenVoxelSettings {
                source_aware_history: true,
                ..ScreenVoxelSettings::residual()
            })
        },
    },
];

/// The [`MethodInfo`] of a scene's lighting, if it uses GI.
pub fn method_info(lighting: LightingMethod) -> Option<&'static MethodInfo> {
    METHODS
        .iter()
        .chain(EXPERIMENTAL_METHODS)
        .find(|info| info.lighting == lighting)
}

/// The GI extension: install one with `Renderer::set_extension` before loading
/// the geometry it should light.
pub struct Lux {
    method: Box<dyn GiMethod>,
}

impl Lux {
    pub fn new(method: Method) -> Self {
        let method: Box<dyn GiMethod> = match method {
            Method::SplitRadianceCascades(settings) => {
                Box::new(SplitRadianceCascades::new(settings))
            }
            Method::Pxtgi(settings) => Box::new(Pxtgi::new(settings)),
            Method::ScreenVoxel(settings) => Box::new(ScreenVoxel::new(settings)),
        };
        Self { method }
    }

    /// Lux for a scene's lighting, if it uses GI.
    pub fn for_lighting(lighting: LightingMethod) -> Option<Self> {
        method_info(lighting).map(|info| Self::new((info.method)()))
    }

    /// Follow the physics world. Call after stepping it, before drawing.
    pub fn sync(&mut self, physics: &PhysicsWorld, world: &World) {
        self.method.sync(physics, world);
    }

    /// Whether [`Lux::sync`] does anything for this method.
    pub fn follows_physics(&self) -> bool {
        self.method.follows_physics()
    }

    pub fn status(&self) -> GiStatus {
        self.method.status()
    }

    pub fn gpu_time(&self) -> Option<Duration> {
        self.method.gpu_time()
    }

    /// Milliseconds per pass in a recent frame, in frame order.
    pub fn pass_times(&self) -> &[(&'static str, f32)] {
        self.method.pass_times()
    }

    /// Trace with hardware ray queries where the GPU has them (the default),
    /// or in software. Split Radiance Cascades recompile its passes to switch.
    pub fn set_hardware_rays(&mut self, on: bool) {
        self.method.set_hardware_rays(on);
    }

    /// Read back the next frame's HDR image, if the method can.
    pub fn request_capture(&mut self) {
        self.method.request_capture();
    }

    /// The capture asked for, once the GPU has finished it.
    pub fn take_capture(&mut self) -> Option<Capture> {
        self.method.take_capture()
    }

    /// The running method, if it is a `T`, for what only that method offers.
    pub fn method_mut<T: 'static>(&mut self) -> Option<&mut T> {
        self.method.as_any_mut().downcast_mut()
    }

    /// The method's status in a line, for showing to a person.
    pub fn status_line(&self) -> String {
        match self.status() {
            GiStatus::Compiling { since } => format!(
                "GI compiling pipelines… {}s (hemisphere until ready)",
                since.elapsed().as_secs()
            ),
            GiStatus::Failed(error) => format!("GI failed: {error}"),
            GiStatus::Ready { compile_time } => match self.gpu_time() {
                Some(time) => format!("GI active ({:.2} ms GPU)", time.as_secs_f32() * 1000.0),
                None => format!("GI active (compiled in {:.1}s)", compile_time.as_secs_f32()),
            },
        }
    }
}

impl RenderExtension for Lux {
    fn scene_loaded(
        &mut self,
        gpu: &mut Gpu,
        scene: &SceneData,
        material_base: u32,
        draw_base: u32,
    ) -> Result<()> {
        (self.method).scene_loaded(gpu, scene, material_base, draw_base)
    }

    fn retained_changed(
        &mut self,
        gpu: &mut Gpu,
        changes: &[RetainedChange],
        material: u32,
    ) -> Result<()> {
        self.method.retained_changed(gpu, changes, material)
    }

    fn targets_changed(&mut self, gpu: &mut Gpu, targets: &Targets) -> Result<()> {
        self.method.targets_changed(gpu, targets)
    }

    fn prepare(&mut self, gpu: &mut Gpu, viewport: [u32; 4]) -> Result<()> {
        self.method.prepare(gpu, viewport)
    }

    fn record(&mut self, gpu: &mut Gpu, frame: &FrameContext) -> Result<()> {
        self.method.record(gpu, frame)
    }

    fn destroy(&mut self, gpu: &mut Gpu) {
        self.method.destroy(gpu);
    }

    fn active(&self) -> bool {
        self.method.active()
    }

    fn sky_reflections(&self) -> bool {
        self.method.sky_reflections()
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}
