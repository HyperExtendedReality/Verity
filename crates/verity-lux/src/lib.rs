//! Split Radiance Cascades global illumination for the Verity renderer.
//!
//! This is a rebuild of the Bevy `bevy_lux` crate on Verity's Vulkan
//! renderer. The transport shaders in `shaders/` carry over; their inputs
//! change from Bevy's deferred G-buffer, receiver pass and material atlas to
//! Verity's visibility buffer, bindless materials and glTF scenes, and their
//! BSDF becomes the renderer's multiple-scattering GGX with Hammon diffuse.
//! The original Bevy source stays in `legacy/` as the porting reference; it
//! is not compiled.
//!
//! Ported so far: [`scene`], the CPU builder of the two-level BVH and the
//! geometry/scene word buffers the traversal shader reads.

pub mod gi;
pub mod scene;
pub mod settings;
