//! Verifies the renderer's extension hook and thin G-buffer: Sponza with a
//! render extension that decodes the G-buffer using lux's own shader decoders
//! into four bands (shading normal, geometric normal, base color, motion).
//!
//! `cargo run --release -p hyperon-lux --example surface_view`
use ash::vk;
use glam::Vec3;
use hyperon_core::World;
use hyperon_render::{
    FrameContext, Gpu, RenderExtension, Renderer, Sun, Targets, asset::SceneData,
    extension::Result, gpu,
};
use std::{path::Path, sync::Arc, time::Instant};
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop},
    window::{Window, WindowId},
};

const SPIRV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/debug_surface.debugSurface.spv"));

#[derive(Default)]
struct SurfaceView {
    set_layout: vk::DescriptorSetLayout,
    layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    pool: vk::DescriptorPool,
    set: vk::DescriptorSet,
}

impl RenderExtension for SurfaceView {
    fn scene_loaded(&mut self, _: &mut Gpu, _: &SceneData, _: u32, _: u32) -> Result<()> {
        Ok(())
    }

    fn targets_changed(&mut self, gpu: &mut Gpu, targets: &Targets) -> Result<()> {
        let device = gpu.device;
        unsafe {
            if self.pipeline == vk::Pipeline::null() {
                use vk::DescriptorType as D;
                let bindings: Vec<_> = [
                    D::SAMPLED_IMAGE,
                    D::SAMPLED_IMAGE,
                    D::SAMPLED_IMAGE,
                    D::STORAGE_IMAGE,
                ]
                .into_iter()
                .enumerate()
                .map(|(i, kind)| {
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(i as u32)
                        .descriptor_type(kind)
                        .descriptor_count(1)
                        .stage_flags(vk::ShaderStageFlags::COMPUTE)
                })
                .collect();
                self.set_layout = device.create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                    None,
                )?;
                let push = [vk::PushConstantRange::default()
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
                    .size(16)];
                self.layout = device.create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::default()
                        .set_layouts(&[self.set_layout])
                        .push_constant_ranges(&push),
                    None,
                )?;
                self.pipeline = gpu::compute_pipeline(device, vk::PipelineCache::null(), self.layout, SPIRV, c"debugSurface")?;
                let sizes = [
                    vk::DescriptorPoolSize::default()
                        .ty(D::SAMPLED_IMAGE)
                        .descriptor_count(3),
                    vk::DescriptorPoolSize::default()
                        .ty(D::STORAGE_IMAGE)
                        .descriptor_count(1),
                ];
                self.pool = device.create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::default()
                        .max_sets(1)
                        .pool_sizes(&sizes),
                    None,
                )?;
                self.set = device.allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(self.pool)
                        .set_layouts(&[self.set_layout]),
                )?[0];
            }
            let views = [
                (vk::DescriptorType::SAMPLED_IMAGE, targets.gbuffer),
                (vk::DescriptorType::SAMPLED_IMAGE, targets.receiver),
                (vk::DescriptorType::SAMPLED_IMAGE, targets.motion),
                (vk::DescriptorType::STORAGE_IMAGE, targets.hdr),
            ];
            let infos = views.map(|(_, view)| {
                [vk::DescriptorImageInfo::default()
                    .image_view(view)
                    .image_layout(vk::ImageLayout::GENERAL)]
            });
            let writes: Vec<_> = views
                .iter()
                .zip(&infos)
                .enumerate()
                .map(|(binding, ((kind, _), info))| {
                    vk::WriteDescriptorSet::default()
                        .dst_set(self.set)
                        .dst_binding(binding as u32)
                        .descriptor_type(*kind)
                        .image_info(info)
                })
                .collect();
            device.update_descriptor_sets(&writes, &[]);
        }
        Ok(())
    }

    fn record(&mut self, gpu: &mut Gpu, frame: &FrameContext) -> Result<()> {
        let [_, _, width, height] = frame.viewport;
        let device = gpu.device;
        let viewport: Vec<u8> = frame
            .viewport
            .iter()
            .flat_map(|v| v.to_ne_bytes())
            .collect();
        unsafe {
            device.cmd_bind_pipeline(frame.cmd, vk::PipelineBindPoint::COMPUTE, self.pipeline);
            device.cmd_bind_descriptor_sets(
                frame.cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.layout,
                0,
                &[self.set],
                &[],
            );
            device.cmd_push_constants(
                frame.cmd,
                self.layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                &viewport,
            );
            device.cmd_dispatch(frame.cmd, width.div_ceil(8), height.div_ceil(8), 1);
        }
        Ok(())
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn destroy(&mut self, gpu: &mut Gpu) {
        unsafe {
            gpu.device.destroy_pipeline(self.pipeline, None);
            gpu.device.destroy_pipeline_layout(self.layout, None);
            gpu.device.destroy_descriptor_pool(self.pool, None);
            gpu.device
                .destroy_descriptor_set_layout(self.set_layout, None);
        }
    }
}

struct Viewer {
    world: World,
    window: Option<Arc<Window>>,
    renderer: Option<Renderer>,
    start: Instant,
}

impl ApplicationHandler for Viewer {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let attributes = Window::default_attributes()
            .with_title("Hyperon G-buffer view")
            .with_inner_size(winit::dpi::LogicalSize::new(1280, 720));
        let window = Arc::new(event_loop.create_window(attributes).expect("window"));
        let result = Renderer::new(window.clone()).and_then(|mut renderer| {
            renderer.set_extension(Box::new(SurfaceView::default()))?;
            let packed = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../assets/packed");
            for scene in ["NewSponza_Main.glb", "NewSponza_Curtains.glb"] {
                renderer.load_gltf(packed.join(scene), 512)?;
            }
            renderer.set_sun(Some(Sun {
                direction: Vec3::new(-0.3, -1.0, 0.2).normalize(),
                color: Vec3::splat(3.0),
            }));
            Ok(renderer)
        });
        match result {
            Ok(renderer) => {
                self.renderer = Some(renderer);
                self.window = Some(window);
                self.start = Instant::now();
            }
            Err(error) => {
                eprintln!("{error}");
                event_loop.exit();
            }
        }
    }

    fn about_to_wait(&mut self, _: &ActiveEventLoop) {
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, id: WindowId, event: WindowEvent) {
        if self.window.as_ref().map(|w| w.id()) != Some(id) {
            return;
        }
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                if let Some(renderer) = &mut self.renderer {
                    renderer.resize(size.width, size.height);
                }
            }
            WindowEvent::RedrawRequested => {
                let Some(renderer) = &mut self.renderer else {
                    return;
                };
                // A moving camera gives the motion band something to show.
                let t = self.start.elapsed().as_secs_f32() * 0.15;
                let center = Vec3::new(2.0, 3.0, 4.0);
                let eye = center + Vec3::new(6.0 * t.cos(), 0.0, 1.2 * t.sin());
                let target = center + Vec3::new(-6.0 * t.cos(), 2.0, -1.2 * t.sin());
                renderer.set_camera(eye, target, 60f32.to_radians());
                if let Err(error) = renderer.draw_world(&self.world) {
                    eprintln!("render: {error}");
                    event_loop.exit();
                }
            }
            _ => {}
        }
    }
}

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    EventLoop::new()?.run_app(&mut Viewer {
        world: World::new(),
        window: None,
        renderer: None,
        start: Instant::now(),
    })?;
    Ok(())
}
