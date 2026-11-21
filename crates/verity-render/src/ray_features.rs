use crate::extension::RayFeatures;
use ash::vk;
use std::ffi::{CStr, c_char, c_void};
use std::ptr;

const CLUSTER: &CStr = c"VK_NV_cluster_acceleration_structure";
const PARTITIONED: &CStr = c"VK_NV_partitioned_acceleration_structure";
const DENSE: &CStr = c"VK_AMDX_dense_geometry_format";

// The SDK's CLAS, PTLAS and DGF feature structs all contain a single
// VkBool32 after the standard mutable feature-chain header. Ash 0.38 lacks
// these declarations. The tags below come from vulkan_core.h (DGF's struct
// declaration is in vulkan_beta.h).
#[repr(C)]
struct Feature<const TAG: i32> {
    s_type: vk::StructureType,
    p_next: *mut c_void,
    supported: vk::Bool32,
}

type Cluster = Feature<1_000_569_000>;
type Partitioned = Feature<1_000_570_000>;
type Dense = Feature<1_000_478_000>;

// SAFETY: These private instances match their SDK structs and begin with the
// VkBaseOutStructure header, with the corresponding structure tag.
unsafe impl<const TAG: i32> vk::TaggedStructure for Feature<TAG> {
    const STRUCTURE_TYPE: vk::StructureType = vk::StructureType::from_raw(TAG);
}
// SAFETY: vk.xml declares all three feature structs as extenders of both roots.
unsafe impl<const TAG: i32> vk::ExtendsPhysicalDeviceFeatures2 for Feature<TAG> {}
// SAFETY: vk.xml declares all three feature structs as extenders of both roots.
unsafe impl<const TAG: i32> vk::ExtendsDeviceCreateInfo for Feature<TAG> {}

impl<const TAG: i32> Feature<TAG> {
    const fn new(supported: vk::Bool32) -> Self {
        Self {
            s_type: <Self as vk::TaggedStructure>::STRUCTURE_TYPE,
            p_next: ptr::null_mut(),
            supported,
        }
    }
}

/// Probe only after KHR acceleration structures and ray queries are available.
///
/// # Safety
/// `physical` must be a live physical device enumerated from `instance`.
pub(super) unsafe fn probe(
    instance: &ash::Instance,
    physical: vk::PhysicalDevice,
    has: impl Fn(&CStr) -> bool,
    indirect_build: bool,
) -> RayFeatures {
    let opacity = has(ash::ext::opacity_micromap::NAME);
    let cluster = has(CLUSTER);
    let partitioned = has(PARTITIONED);
    // DGF depends on KHR acceleration structures plus Vulkan 1.4 OR
    // extended_flags OR maintenance5. On our Vulkan 1.3 path, use maintenance5
    // for the buffer usage flags required by DGF, even on a 1.4 adapter.
    // The indirect command also avoids a crash on early DGF drivers when
    // recording the direct command with its now-required null range pointer.
    let dense = indirect_build && has(DENSE) && has(ash::khr::maintenance5::NAME);
    if !(opacity || cluster || partitioned || dense) {
        return RayFeatures::default();
    }
    let mut micromap = vk::PhysicalDeviceOpacityMicromapFeaturesEXT::default();
    let mut cluster_feature = Cluster::new(vk::FALSE);
    let mut partitioned_feature = Partitioned::new(vk::FALSE);
    let mut dense_feature = Dense::new(vk::FALSE);
    let mut maintenance5 = vk::PhysicalDeviceMaintenance5FeaturesKHR::default();
    let mut features = vk::PhysicalDeviceFeatures2::default();
    if opacity {
        features = features.push_next(&mut micromap);
    }
    if cluster {
        features = features.push_next(&mut cluster_feature);
    }
    if partitioned {
        features = features.push_next(&mut partitioned_feature);
    }
    if dense {
        features = features
            .push_next(&mut dense_feature)
            .push_next(&mut maintenance5);
    }
    // SAFETY: The device came from this instance. Each advertised extension's
    // output struct is initialized and remains live until the query returns.
    unsafe { instance.get_physical_device_features2(physical, &mut features) };
    RayFeatures {
        opacity_micromap: micromap.micromap != vk::FALSE,
        cluster_acceleration: cluster_feature.supported != vk::FALSE,
        partitioned_acceleration: partitioned_feature.supported != vk::FALSE,
        dense_geometry: dense_feature.supported != vk::FALSE
            && maintenance5.maintenance5 != vk::FALSE,
    }
}

impl RayFeatures {
    pub(super) fn add_extension_names(self, names: &mut Vec<*const c_char>) {
        if self.opacity_micromap {
            names.push(ash::ext::opacity_micromap::NAME.as_ptr());
        }
        if self.cluster_acceleration {
            names.push(CLUSTER.as_ptr());
        }
        if self.partitioned_acceleration {
            names.push(PARTITIONED.as_ptr());
        }
        if self.dense_geometry {
            names.extend([ash::khr::maintenance5::NAME.as_ptr(), DENSE.as_ptr()]);
        }
    }
}

/// Owns the optional feature structs until `create_device` returns.
pub(super) struct EnableChain {
    micromap: vk::PhysicalDeviceOpacityMicromapFeaturesEXT<'static>,
    cluster: Cluster,
    partitioned: Partitioned,
    dense: Dense,
    maintenance5: vk::PhysicalDeviceMaintenance5FeaturesKHR<'static>,
}

impl EnableChain {
    pub(super) fn new() -> Self {
        Self {
            micromap: vk::PhysicalDeviceOpacityMicromapFeaturesEXT::default().micromap(true),
            cluster: Cluster::new(vk::TRUE),
            partitioned: Partitioned::new(vk::TRUE),
            dense: Dense::new(vk::TRUE),
            maintenance5: vk::PhysicalDeviceMaintenance5FeaturesKHR::default().maintenance5(true),
        }
    }

    pub(super) fn push_next<'a>(
        &'a mut self,
        mut info: vk::DeviceCreateInfo<'a>,
        enabled: RayFeatures,
    ) -> vk::DeviceCreateInfo<'a> {
        if enabled.opacity_micromap {
            info = info.push_next(&mut self.micromap);
        }
        if enabled.cluster_acceleration {
            info = info.push_next(&mut self.cluster);
        }
        if enabled.partitioned_acceleration {
            info = info.push_next(&mut self.partitioned);
        }
        if enabled.dense_geometry {
            info = info
                .push_next(&mut self.dense)
                .push_next(&mut self.maintenance5);
        }
        info
    }
}
