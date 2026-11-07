//! Global illumination for the Verity renderer: several methods, one extension.
//!
//! [`Lux`] is what the renderer installs; it runs one [`Method`]:
//!
//! - [`cascades`], Split Radiance Cascades, which traces a GI scene of its own.
//! - [`pxtgi`], which traces the physics scene.
//! - [`voxel`], which traces the screen, then a voxel copy of the scene.
//!
//! The editor offers whatever [`METHODS`] lists. To add a method, implement
//! [`GiMethod`] for it, add a [`Method`] variant and a row in [`METHODS`], and
//! give it a `LightingMethod` so scenes can name it. It follows terrain through
//! `RenderExtension::retained_changed`.
//!
//! `legacy/` keeps the original Bevy source of the cascades as a porting
//! reference; it is not compiled.

pub mod cascades;
pub mod pxtgi;
mod shared;
pub mod voxel;

use cascades::SplitRadianceCascades;
use pxtgi::{Capture, Pxtgi, PxtgiSettings};
use std::{
    any::Any,
    time::{Duration, Instant},
};
use voxel::{ScreenVoxel, ScreenVoxelSettings};
use verity_core::{LightingMethod, World};
use verity_physics::PhysicsWorld;
use verity_render::{
    FrameContext, Gpu, RenderExtension, RetainedChange, Targets, asset::SceneData,
    extension::Result,
};

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

    /// Read back the next frame's HDR image, if the method can.
    fn request_capture(&mut self) {}

    /// The capture asked for, once the GPU has finished it.
    fn take_capture(&mut self) -> Option<Capture> {
        None
    }
}

/// A [`GiMethod`] to run, with its settings.
pub enum Method {
    SplitRadianceCascades(cascades::settings::Settings),
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

/// Every method, for applications to offer.
pub const METHODS: &[MethodInfo] = &[
    MethodInfo {
        lighting: LightingMethod::SplitRadianceCascades,
        name: "Split radiance cascades GI",
        about: "Probes in a hierarchy of cascades, traced against a GI scene of its own.",
        method: || Method::SplitRadianceCascades(cascades::settings::Settings::budgeted()),
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
];

/// The [`MethodInfo`] of a scene's lighting, if it uses GI.
pub fn method_info(lighting: LightingMethod) -> Option<&'static MethodInfo> {
    METHODS.iter().find(|info| info.lighting == lighting)
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
