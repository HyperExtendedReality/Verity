use ash::vk;

pub const GPU_STAGE_NAMES: [&str; 9] = [
    "Geometry uploads", "Scene setup and light bins", "Shadows", "Depth and occlusion",
    "Visibility", "Resolve", "GI", "Water", "Composite and UI",
];
const QUERIES: u32 = GPU_STAGE_NAMES.len() as u32 + 1;

pub(crate) struct GpuTimer {
    pool: vk::QueryPool,
    period: f32,
    bits: u32,
    pending: [bool; super::FRAMES],
    pub times: Option<[f32; GPU_STAGE_NAMES.len()]>,
}

impl GpuTimer {
    pub unsafe fn new(instance: &ash::Instance, device: &ash::Device, physical: vk::PhysicalDevice, queue_family: u32) -> Option<Self> {
        let bits = instance.get_physical_device_queue_family_properties(physical)[queue_family as usize].timestamp_valid_bits;
        if bits == 0 { return None; }
        let pool = device.create_query_pool(&vk::QueryPoolCreateInfo::default().query_type(vk::QueryType::TIMESTAMP).query_count(QUERIES * super::FRAMES as u32), None).ok()?;
        let period = instance.get_physical_device_properties(physical).limits.timestamp_period;
        Some(Self { pool, period, bits, pending: [false; super::FRAMES], times: None })
    }

    pub unsafe fn poll(&mut self, device: &ash::Device, slot: usize) {
        if !std::mem::take(&mut self.pending[slot]) { return; }
        let mut ticks = [0u64; GPU_STAGE_NAMES.len() + 1];
        if device.get_query_pool_results(self.pool, slot as u32 * QUERIES, &mut ticks, vk::QueryResultFlags::TYPE_64).is_ok() {
            let mask = u64::MAX >> (64 - self.bits.min(64));
            self.times = Some(std::array::from_fn(|i| (ticks[i + 1].wrapping_sub(ticks[i]) & mask) as f32 * self.period * 1e-6));
        }
    }

    pub unsafe fn begin(&self, device: &ash::Device, cmd: vk::CommandBuffer, slot: usize) {
        device.cmd_reset_query_pool(cmd, self.pool, slot as u32 * QUERIES, QUERIES);
        self.stamp(device, cmd, slot, 0);
    }

    pub unsafe fn stamp(&self, device: &ash::Device, cmd: vk::CommandBuffer, slot: usize, i: u32) {
        device.cmd_write_timestamp2(cmd, vk::PipelineStageFlags2::ALL_COMMANDS, self.pool, slot as u32 * QUERIES + i);
    }

    pub fn submitted(&mut self, slot: usize) { self.pending[slot] = true; }
    pub unsafe fn destroy(self, device: &ash::Device) { device.destroy_query_pool(self.pool, None); }
}
