//! Hardware tiers: which acceleration structures and GI method suit the GPU.
//!
//! Lux has several ways to answer a ray: Vulkan acceleration structures with
//! vendor extensions (NVIDIA cluster BLAS and partitioned TLAS, AMD dense
//! geometry), plain Vulkan ray queries, its own compressed four-wide BVH with
//! optional oriented child bounds and spatial splits, and voxel tracing over
//! an SDF or sparse occupancy. [`HardwareProfile::classify`] picks among them
//! for one device, so a scene gets hardware ray tracing on RTX and RDNA GPUs and
//! a fallback on integrated, older and mobile GPUs without anyone setting flags.
//!
//! Every choice can still be forced with its environment switch: `1` turns it
//! on where the device supports it, `0` turns it off, and leaving it unset
//! keeps the profile's choice. The switches are listed on
//! [`HardwareProfile::with_env`].

use ash::vk;
use verity_render::{Gpu, extension::RayFeatures};

use crate::{
    Method,
    cascades::settings::Settings,
    voxel::{ScreenVoxelSettings, VoxelStorage, VoxelVisibility},
};

/// PCI vendor IDs as Vulkan reports them.
mod vendor_id {
    pub const AMD: u32 = 0x1002;
    pub const IMAGINATION: u32 = 0x1010;
    pub const APPLE: u32 = 0x106B;
    pub const NVIDIA: u32 = 0x10DE;
    pub const ARM: u32 = 0x13B5;
    pub const SAMSUNG: u32 = 0x144D;
    pub const BROADCOM: u32 = 0x14E4;
    pub const QUALCOMM: u32 = 0x5143;
    pub const INTEL: u32 = 0x8086;
}

/// Who made the GPU.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Vendor {
    Nvidia,
    Amd,
    Intel,
    Apple,
    /// Mobile designs: Arm Mali, Qualcomm Adreno, Imagination PowerVR,
    /// Samsung Xclipse and Broadcom VideoCore.
    Mobile,
    Other(u32),
}

impl Vendor {
    pub fn from_id(id: u32) -> Self {
        match id {
            vendor_id::NVIDIA => Self::Nvidia,
            vendor_id::AMD => Self::Amd,
            vendor_id::INTEL => Self::Intel,
            vendor_id::APPLE => Self::Apple,
            vendor_id::ARM
            | vendor_id::QUALCOMM
            | vendor_id::IMAGINATION
            | vendor_id::SAMSUNG
            | vendor_id::BROADCOM => Self::Mobile,
            other => Self::Other(other),
        }
    }
}

/// What kind of device it is, which decides how much memory and bandwidth a
/// fallback may spend.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeviceClass {
    /// A discrete desktop or laptop GPU with memory of its own.
    Discrete,
    /// A desktop or laptop GPU sharing system memory.
    Integrated,
    /// A phone, tablet or handheld GPU, a software rasterizer, or anything
    /// with too little device memory for the larger structures.
    Low,
}

/// The device's capability tier, best first.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Tier {
    /// Ray queries plus a vendor's acceleration-structure extension: NVIDIA
    /// cluster or partitioned structures (RTX 40 and later drivers), AMD dense
    /// geometry (RDNA 3 and later), or opacity micromaps.
    HardwareExtended,
    /// Ray queries with the core Vulkan acceleration structures only.
    Hardware,
    /// No ray queries, but a desktop-class GPU: Lux walks its own BVH.
    Software,
    /// No ray queries on a [`DeviceClass::Low`] device: voxel GI.
    Low,
}

/// What [`HardwareProfile::classify`] reads of a device.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeviceSummary {
    pub vendor_id: u32,
    pub device_type: vk::PhysicalDeviceType,
    /// The largest device-local heap, in bytes.
    pub device_local_bytes: u64,
    /// Whether the renderer enabled ray queries.
    pub ray_query: bool,
    pub ray_features: RayFeatures,
}

/// Device-local memory below which a GPU counts as [`DeviceClass::Low`],
/// whatever it reports itself as: spatial splits and the surface cache would
/// crowd out the scene.
const LOW_MEMORY: u64 = 2 << 30;

impl DeviceSummary {
    /// Read the device the renderer lent an extension.
    pub fn of(gpu: &Gpu) -> Self {
        // SAFETY: `gpu.physical` was enumerated from `gpu.instance`, which
        // outlives the borrow.
        let (properties, memory) = unsafe {
            (
                gpu.instance.get_physical_device_properties(gpu.physical),
                gpu.instance
                    .get_physical_device_memory_properties(gpu.physical),
            )
        };
        let heaps = &memory.memory_heaps[..memory.memory_heap_count as usize];
        let device_local_bytes = heaps
            .iter()
            .filter(|heap| heap.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL))
            .map(|heap| heap.size)
            .max()
            .unwrap_or(0);
        Self {
            vendor_id: properties.vendor_id,
            device_type: properties.device_type,
            device_local_bytes,
            ray_query: gpu.ray_query,
            ray_features: gpu.ray_features,
        }
    }

    pub fn vendor(&self) -> Vendor {
        Vendor::from_id(self.vendor_id)
    }

    pub fn class(&self) -> DeviceClass {
        let vendor = self.vendor();
        let mobile = vendor == Vendor::Mobile
            || matches!(
                self.device_type,
                vk::PhysicalDeviceType::CPU | vk::PhysicalDeviceType::OTHER
            );
        if mobile || self.device_local_bytes < LOW_MEMORY {
            DeviceClass::Low
        } else if self.device_type == vk::PhysicalDeviceType::DISCRETE_GPU {
            DeviceClass::Discrete
        } else {
            // Integrated and virtual GPUs share memory with the CPU; Apple's
            // report integrated too.
            DeviceClass::Integrated
        }
    }
}

/// How the software BVH of static meshes is laid out. Only used where rays are
/// traced in software.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SoftwareLayout {
    /// Oriented child bounds where they cut a child's area enough to pay for
    /// the extra test (`VERITY_LUX_ORIENTED_BLAS`).
    pub oriented: bool,
    /// Spatial splits: triangles straddling a split are referenced from both
    /// sides, which tightens bounds at the cost of memory
    /// (`VERITY_LUX_SPATIAL_BLAS`).
    pub spatial: bool,
}

/// Which hardware acceleration features to build with, where supported.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct HardwareBuilds {
    /// NVIDIA cluster BLAS for static geometry (`VERITY_LUX_CLUSTER_BLAS`).
    pub cluster_blas: bool,
    /// NVIDIA partitioned TLAS, updating only changed instance slots
    /// (`VERITY_LUX_PARTITIONED_TLAS`).
    pub partitioned_tlas: bool,
    /// AMD dense geometry format BLAS (`VERITY_LUX_DGF_BLAS`).
    pub dgf_blas: bool,
    /// Opacity micromaps for provable alpha coverage
    /// (`VERITY_LUX_OPACITY_MICROMAPS`).
    pub opacity_micromaps: bool,
    /// Commit opaque two-sided hits in hardware (`VERITY_LUX_OPAQUE_RAYS`).
    pub opaque_rays: bool,
}

/// The acceleration choices for one device.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HardwareProfile {
    pub tier: Tier,
    pub vendor: Vendor,
    pub class: DeviceClass,
    /// Trace with hardware ray queries.
    pub hardware_rays: bool,
    pub builds: HardwareBuilds,
    pub software: SoftwareLayout,
    /// The device's ray features, so overrides cannot enable what it lacks.
    features: RayFeatures,
    ray_query: bool,
}

impl HardwareProfile {
    /// The profile for the device the renderer lent an extension, with
    /// environment overrides applied.
    pub fn detect(gpu: &Gpu) -> Self {
        Self::classify(&DeviceSummary::of(gpu)).with_env()
    }

    /// The profile for a device, without environment overrides.
    pub fn classify(device: &DeviceSummary) -> Self {
        let vendor = device.vendor();
        let class = device.class();
        let features = if device.ray_query {
            device.ray_features
        } else {
            RayFeatures::default()
        };
        let extended = features.cluster_acceleration
            || features.partitioned_acceleration
            || features.dense_geometry
            || features.opacity_micromap;
        let tier = match (device.ray_query, extended, class) {
            (true, true, _) => Tier::HardwareExtended,
            (true, false, _) => Tier::Hardware,
            (false, _, DeviceClass::Low) => Tier::Low,
            (false, _, _) => Tier::Software,
        };
        // Vendor builds stay opt-in. On Sponza, cluster BLAS with a partitioned
        // TLAS and opaque commits made GI 10% slower on an RTX 4070, and DGF
        // encoded no BLAS on an RDNA 3 iGPU (docs/validation.md).
        let builds = HardwareBuilds::default();
        // Every tier keeps the plain four-wide software BVH. On Sponza, spatial
        // splits gained 2-9% software GI on an RTX 4070 and 3% on an RDNA 3
        // iGPU for 2-10x the build time (seconds per model); oriented bounds
        // cost 12% on the RTX and gained nothing on the iGPU. Both remain
        // opt-ins through `with_env`.
        let software = SoftwareLayout::default();
        Self {
            tier,
            vendor,
            class,
            hardware_rays: device.ray_query,
            builds,
            software,
            features,
            ray_query: device.ray_query,
        }
    }

    /// Apply the environment switches. Each takes `1` (on, where supported)
    /// or `0` (off); unset keeps the profile's choice.
    ///
    /// | Switch | Field |
    /// | --- | --- |
    /// | `VERITY_LUX_CLUSTER_BLAS` | [`HardwareBuilds::cluster_blas`] |
    /// | `VERITY_LUX_PARTITIONED_TLAS` | [`HardwareBuilds::partitioned_tlas`] |
    /// | `VERITY_LUX_DGF_BLAS` | [`HardwareBuilds::dgf_blas`] |
    /// | `VERITY_LUX_OPACITY_MICROMAPS` | [`HardwareBuilds::opacity_micromaps`] |
    /// | `VERITY_LUX_OPAQUE_RAYS` | [`HardwareBuilds::opaque_rays`] |
    /// | `VERITY_LUX_ORIENTED_BLAS` | [`SoftwareLayout::oriented`] |
    /// | `VERITY_LUX_SPATIAL_BLAS` | [`SoftwareLayout::spatial`] |
    pub fn with_env(self) -> Self {
        self.with_overrides(|key| std::env::var_os(key)?.into_string().ok())
    }

    fn with_overrides(mut self, read: impl Fn(&str) -> Option<String>) -> Self {
        let switch = |key: &str, value: &mut bool| match read(key).as_deref() {
            Some("1") => *value = true,
            Some("0") => *value = false,
            _ => {}
        };
        let builds = &mut self.builds;
        switch("VERITY_LUX_CLUSTER_BLAS", &mut builds.cluster_blas);
        switch("VERITY_LUX_PARTITIONED_TLAS", &mut builds.partitioned_tlas);
        switch("VERITY_LUX_DGF_BLAS", &mut builds.dgf_blas);
        switch(
            "VERITY_LUX_OPACITY_MICROMAPS",
            &mut builds.opacity_micromaps,
        );
        switch("VERITY_LUX_OPAQUE_RAYS", &mut builds.opaque_rays);
        switch("VERITY_LUX_ORIENTED_BLAS", &mut self.software.oriented);
        switch("VERITY_LUX_SPATIAL_BLAS", &mut self.software.spatial);
        self.supported()
    }

    /// Drop the builds the device cannot do.
    fn supported(mut self) -> Self {
        let f = self.features;
        let b = &mut self.builds;
        b.cluster_blas &= f.cluster_acceleration;
        b.partitioned_tlas &= f.partitioned_acceleration;
        b.dgf_blas &= f.dense_geometry;
        b.opacity_micromaps &= f.opacity_micromap;
        b.opaque_rays &= self.ray_query;
        self.hardware_rays &= self.ray_query;
        self
    }

    /// The tier and how rays are traced, in a few words for a person.
    pub fn describe(&self) -> &'static str {
        match (self.tier, self.hardware_rays) {
            (Tier::Low, _) => "low-end GPU: screen and voxel GI",
            (_, true) => "hardware rays",
            (_, false) => "software BVH",
        }
    }

    /// The GI method that fits this tier, with its usual settings for it.
    ///
    /// Ray-traced tiers and desktop software tracing get Split Radiance
    /// Cascades, budgeted on integrated GPUs. [`Tier::Low`] gets screen and
    /// voxel GI with SDF visibility over hashed bricks: no triangle BVH, a
    /// fixed memory footprint, and fewer rays per texel.
    pub fn recommended_method(&self) -> Method {
        match self.tier {
            Tier::Low => Method::ScreenVoxel(ScreenVoxelSettings {
                rays_per_texel: 2,
                levels: 4,
                visibility: VoxelVisibility::Sdf,
                storage: VoxelStorage::Hash,
                ..ScreenVoxelSettings::default()
            }),
            _ => {
                let mut settings = match self.class {
                    DeviceClass::Discrete => Settings::default(),
                    DeviceClass::Integrated | DeviceClass::Low => Settings::budgeted(),
                };
                settings.hardware_rays = self.hardware_rays;
                Method::SplitRadianceCascades(settings)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1 << 30;

    fn device(vendor_id: u32, device_type: vk::PhysicalDeviceType, gib: u64) -> DeviceSummary {
        DeviceSummary {
            vendor_id,
            device_type,
            device_local_bytes: gib * GIB,
            ray_query: false,
            ray_features: RayFeatures::default(),
        }
    }

    fn ray_traced(mut device: DeviceSummary, features: RayFeatures) -> DeviceSummary {
        device.ray_query = true;
        device.ray_features = features;
        device
    }

    fn features(f: impl FnOnce(&mut RayFeatures)) -> RayFeatures {
        let mut features = RayFeatures::default();
        f(&mut features);
        features
    }

    #[test]
    fn rtx_with_cluster_structures_is_extended() {
        let rtx = ray_traced(
            device(vendor_id::NVIDIA, vk::PhysicalDeviceType::DISCRETE_GPU, 8),
            features(|f| {
                f.cluster_acceleration = true;
                f.partitioned_acceleration = true;
            }),
        );
        let profile = HardwareProfile::classify(&rtx);
        assert_eq!(profile.tier, Tier::HardwareExtended);
        assert_eq!(profile.vendor, Vendor::Nvidia);
        assert!(profile.hardware_rays);
        assert_eq!(profile.software, SoftwareLayout::default());
    }

    #[test]
    fn ray_queries_alone_are_hardware() {
        let rdna2 = ray_traced(
            device(vendor_id::AMD, vk::PhysicalDeviceType::DISCRETE_GPU, 8),
            RayFeatures::default(),
        );
        assert_eq!(HardwareProfile::classify(&rdna2).tier, Tier::Hardware);
    }

    #[test]
    fn desktop_without_ray_queries_traces_in_software() {
        let pascal = device(vendor_id::NVIDIA, vk::PhysicalDeviceType::DISCRETE_GPU, 8);
        let profile = HardwareProfile::classify(&pascal);
        assert_eq!(profile.tier, Tier::Software);
        assert!(!profile.hardware_rays);
        assert_eq!(profile.software, SoftwareLayout::default());
        let igpu = device(vendor_id::INTEL, vk::PhysicalDeviceType::INTEGRATED_GPU, 4);
        let profile = HardwareProfile::classify(&igpu);
        assert_eq!(profile.class, DeviceClass::Integrated);
        assert!(matches!(
            profile.recommended_method(),
            Method::SplitRadianceCascades(Settings {
                hardware_rays: false,
                ..
            })
        ));
    }

    #[test]
    fn mobile_and_small_gpus_fall_back_to_voxels() {
        for low in [
            device(vendor_id::ARM, vk::PhysicalDeviceType::INTEGRATED_GPU, 8),
            device(
                vendor_id::QUALCOMM,
                vk::PhysicalDeviceType::INTEGRATED_GPU,
                6,
            ),
            device(vendor_id::NVIDIA, vk::PhysicalDeviceType::DISCRETE_GPU, 1),
            device(0x10005, vk::PhysicalDeviceType::CPU, 16),
        ] {
            let profile = HardwareProfile::classify(&low);
            assert_eq!(profile.tier, Tier::Low, "{low:?}");
            assert!(matches!(
                profile.recommended_method(),
                Method::ScreenVoxel(ScreenVoxelSettings {
                    visibility: VoxelVisibility::Sdf,
                    storage: VoxelStorage::Hash,
                    ..
                })
            ));
        }
    }

    #[test]
    fn mobile_ray_tracing_still_uses_hardware_rays() {
        let immortalis = ray_traced(
            device(vendor_id::ARM, vk::PhysicalDeviceType::INTEGRATED_GPU, 8),
            RayFeatures::default(),
        );
        let profile = HardwareProfile::classify(&immortalis);
        assert_eq!(profile.tier, Tier::Hardware);
        assert_eq!(profile.class, DeviceClass::Low);
        assert!(matches!(
            profile.recommended_method(),
            Method::SplitRadianceCascades(Settings {
                hardware_rays: true,
                ..
            })
        ));
    }

    #[test]
    fn overrides_cannot_enable_missing_features() {
        let rdna3 = ray_traced(
            device(vendor_id::AMD, vk::PhysicalDeviceType::DISCRETE_GPU, 16),
            features(|f| f.dense_geometry = true),
        );
        let on = |_: &str| Some("1".to_owned());
        let profile = HardwareProfile::classify(&rdna3).with_overrides(on);
        assert!(profile.builds.dgf_blas && profile.builds.opaque_rays);
        assert!(!profile.builds.cluster_blas && !profile.builds.partitioned_tlas);
        assert!(profile.software.oriented && profile.software.spatial);

        let off = |_: &str| Some("0".to_owned());
        let pascal = device(vendor_id::NVIDIA, vk::PhysicalDeviceType::DISCRETE_GPU, 8);
        let profile = HardwareProfile::classify(&pascal).with_overrides(off);
        assert_eq!(profile.software, SoftwareLayout::default());
        assert!(!profile.builds.opaque_rays);
    }

    #[test]
    fn features_without_ray_queries_are_ignored() {
        let mut odd = device(vendor_id::NVIDIA, vk::PhysicalDeviceType::DISCRETE_GPU, 8);
        odd.ray_features.cluster_acceleration = true;
        let profile = HardwareProfile::classify(&odd).with_overrides(|_| Some("1".into()));
        assert_eq!(profile.tier, Tier::Software);
        assert!(!profile.builds.cluster_blas && !profile.hardware_rays);
    }
}
