To implement the clean-slate **"No Graphics API"** architecture in a custom **Rust** game engine, you cannot use high-level abstractions like `wgpu`. `wgpu` is modeled after the WebGPU specification, which perpetuates the exact legacy constraints Sebastian Aaltonen identifies (rigid bind groups, monolithic PSOs, abstract buffer types, and lack of native 64-bit pointer dereferencing in shaders).

Instead, your engine should build directly on **Vulkan 1.3/1.4 via `ash`**, targeting modern hardware features: **Buffer Device Address (BDA)**, **Synchronization2**, **Dynamic State 1–3**, and **Descriptor Buffers / Bindless Arrays**. Shaders should be written in **Slang** (or modern HLSL via DirectX Shader Compiler) to support native 64-bit pointers and shared struct layouts.

---

# System Architecture Overview

```
+----------------------------------------------------------------------------------------------------+
|                                    RUST "NO GRAPHICS API" ENGINE                                   |
+----------------------------------------------------------------------------------------------------+
|  Shared Data Layer       -> #[repr(C, align(16))] structs shared between Rust and Slang/HLSL      |
|  Memory Architecture     -> DeviceAddress<T> + Lockless ReBAR FrameBumpAllocator                  |
|  Root Argument Model     -> Exactly ONE 64-bit pointer pushed per Draw/Dispatch (vkCmdPushConstants)|
|  Bindless Textures       -> Global Unbounded Descriptor Array + 32-bit TextureId in Materials       |
|  Geometry Pipeline       -> Vertex Pulling via Raw Pointers (VkPipelineVertexInputState = Empty)   |
|  Lean Pipeline Cache     -> Stripped PSOs (Only Shader IR + RT Formats + MSAA; dynamic culling/DS) |
|  Synchronization         -> Resource-less Barriers (VkMemoryBarrier2 with Stage/Hazard bitmasks)   |
+----------------------------------------------------------------------------------------------------+
```

---

## Phase 1: Memory Architecture & Strongly Typed GPU Pointers

In this architecture, buffers are not abstract objects. Memory is simply a linear address space, and small dynamic data (draw arguments, uniforms, transforms) is written directly into host-visible, device-local memory (PCIe Resizable BAR / UMA).

### 1.1 Strongly-Typed Device Address Wrapper
Rust allows us to wrap raw 64-bit addresses into type-safe wrappers using `PhantomData`:

```rust
use std::marker::PhantomData;
use ash::vk;

#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceAddress<T> {
    pub addr: vk::DeviceAddress, // u64
    _marker: PhantomData<*const T>,
}

impl<T> DeviceAddress<T> {
    pub const NULL: Self = Self { addr: 0, _marker: PhantomData };

    pub fn new(addr: vk::DeviceAddress) -> Self {
        Self { addr, _marker: PhantomData }
    }

    /// Offset the pointer by `count` elements of `T`
    pub fn offset(&self, count: usize) -> Self {
        Self {
            addr: self.addr + (count * std::mem::size_of::<T>()) as vk::DeviceAddress,
            _marker: PhantomData,
        }
    }
}
```

### 1.2 The ReBAR Frame Bump Allocator (`gpuTempAlloc`)
Instead of managing descriptor pools, constant buffer views, or push constant size limits, create a lockless ring/bump allocator over persistently mapped ReBAR memory:

```rust
use std::ptr::NonNull;

pub struct GpuAllocation<T> {
    pub device_addr: DeviceAddress<T>,
    pub host_ref: &'static mut T,
}

pub struct FrameBumpAllocator {
    base_host: NonNull<u8>,
    base_device: vk::DeviceAddress,
    capacity: usize,
    offset: std::sync::atomic::AtomicUsize,
}

impl FrameBumpAllocator {
    pub fn new(base_host: NonNull<u8>, base_device: vk::DeviceAddress, capacity: usize) -> Self {
        Self {
            base_host,
            base_device,
            capacity,
            offset: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    /// Allocate temporary storage for draw arguments.
    /// Default alignment is 16 bytes (vec4) to enable 128-bit wide loads.
    pub fn alloc<T: bytemuck::NoUninit>(&self) -> Result<GpuAllocation<T>, OutOfMemoryError> {
        let size = std::mem::size_of::<T>();
        let align = std::mem::align_of::<T>().max(16);

        loop {
            let current = self.offset.load(std::sync::atomic::Ordering::Relaxed);
            let aligned = (current + align - 1) & !(align - 1);
            let next = aligned + size;

            if next > self.capacity {
                return Err(OutOfMemoryError);
            }

            if self.offset.compare_exchange_weak(
                current,
                next,
                std::sync::atomic::Ordering::SeqCst,
                std::sync::atomic::Ordering::Relaxed,
            ).is_ok() {
                unsafe {
                    let host_ptr = self.base_host.as_ptr().add(aligned) as *mut T;
                    return Ok(GpuAllocation {
                        device_addr: DeviceAddress::new(self.base_device + aligned as u64),
                        host_ref: &mut *host_ptr,
                    });
                }
            }
        }
    }

    pub fn reset(&mut self) {
        self.offset.store(0, std::sync::atomic::Ordering::Release);
    }
}
```

---

## Phase 2: CPU-GPU Shared Data Contracts

Rust and the shading language must share the exact same POD layout [43:40, 47:00]. Rust’s `#[repr(C, align(16))]` perfectly matches standard GPU SIMD alignments.

### 2.1 The Rust Side (`src/shaders/mesh_types.rs`)
```rust
use bytemuck::{Pod, Zeroable};
use crate::gpu::DeviceAddress;

#[repr(C, align(16))]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct MeshDrawData {
    pub world_view_proj: [f32; 16],
    pub positions: DeviceAddress<[f32; 3]>, // 64-bit GPU pointer to vertex array
    pub uvs: DeviceAddress<[f32; 2]>,       // 64-bit GPU pointer to UV array
    pub albedo_tex: u32,                    // 32-bit index into global texture array
    pub normal_tex: u32,
    pub pbr_tex: u32,
    pub _padding: u32,
}
```

### 2.2 The Shader Side (`mesh.slang` / HLSL 2021)
Because Slang supports raw pointers and typed buffers, it natively consumes this structure:

```hlsl
struct MeshDrawData {
    float4x4 world_view_proj;
    float3* positions;
    float2* uvs;
    uint albedo_tex;
    uint normal_tex;
    uint pbr_tex;
    uint _padding;
};

// Global push constant holding only the 64-bit GPU base pointer
[[vk::push_constant]]
MeshDrawData* g_DrawData;

struct VSOutput {
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
};

[shader("vertex")]
VSOutput vs_main(uint vertex_id : SV_VertexID) {
    VSOutput output;
    // Direct pointer dereferencing: raw memory load instruction emitted
    float3 pos = g_DrawData->positions[vertex_id];
    output.position = mul(g_DrawData->world_view_proj, float4(pos, 1.0));
    output.uv = g_DrawData->uvs[vertex_id];
    return output;
}
```

---

## Phase 3: Global Bindless Texturing

Eliminate `vkUpdateDescriptorSets`, `vkCmdBindDescriptorSets`, and per-material descriptor sets. Textures are managed in a single unbounded descriptor array.

### 3.1 The Rust Texture Registry
```rust
pub struct GlobalTextureHeap {
    descriptor_pool: vk::DescriptorPool,
    descriptor_set: vk::DescriptorSet,
    descriptor_set_layout: vk::DescriptorSetLayout,
    free_list: Vec<u32>,
    allocated_count: u32,
}

impl GlobalTextureHeap {
    pub const MAX_TEXTURES: u32 = 1_000_000;

    pub fn new(device: &ash::Device) -> Self {
        // Allocate 1 descriptor set with an unbounded binding (VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT)
        // binding = 0, descriptorType = COMBINED_IMAGE_SAMPLER, count = 1_000_000
        // ... (standard Vulkan bindless set initialization)
    }

    /// Register a texture once on load; return its permanent 32-bit index.
    pub fn register_texture(&mut self, image_view: vk::ImageView, sampler: vk::Sampler) -> u32 {
        let index = self.free_list.pop().unwrap_or_else(|| {
            let i = self.allocated_count;
            self.allocated_count += 1;
            i
        });

        // Write descriptor into binding 0 at array element `index`
        self.update_slot(index, image_view, sampler);
        index
    }
}
```

* Bind this single descriptor set **once** during frame setup. Never rebind during the pass [59:40, 61:33].

### 3.2 In-Shader Sampling
```hlsl
// Unbounded global texture array
Texture2D g_Textures[] : register(t0, space0);
SamplerState g_LinearSampler : register(s0, space0);

[shader("pixel")]
float4 ps_main(VSOutput input) : SV_Target {
    // Shaders sample textures purely via 32-bit indices from the root struct
    Texture2D albedo = g_Textures[NonUniformResourceIndex(g_DrawData->albedo_tex)];
    return albedo.Sample(g_LinearSampler, input.uv);
}
```

---

## Phase 4: Programmable Vertex Fetching (No Input Assembler)

Input assembly is a fixed-function relic that forces vertex buffers to declare formats, strides, and bindings.

### 4.1 Vulkan Vertex Input State in Rust
In your pipeline creation, define `VkPipelineVertexInputStateCreateInfo` as completely empty [81:30]:

```rust
let vertex_input_info = vk::PipelineVertexInputStateCreateInfo::default(); 
// vertex_binding_description_count = 0
// vertex_attribute_description_count = 0
```

### 4.2 Passing Geometry
Vertices are stored in raw `DeviceAddress` allocations. Meshes simply track their addresses:

```rust
pub struct GpuMesh {
    pub positions: DeviceAddress<[f32; 3]>,
    pub uvs: DeviceAddress<[f32; 2]>,
    pub indices: DeviceAddress<u32>,
    pub index_count: u32,
}
```
* Dynamic vertex strides do not require new pipeline permutations [64:41, 64:51].
* This lays the exact ground layer for **Mesh Shaders** (`VK_EXT_mesh_shader`), which consume raw memory meshlets directly without index deduplication hardware.

---

## Phase 5: Lean PSOs & Decoupled State

Aaltonen highlights that combining rasterizer state, input state, depth-stencil, and blending into one PSO causes the combinatorial explosion responsible for driver stalls and massive disk caches [24:40, 27:50, 80:50].

### 5.1 Enable Vulkan 1.3 Extended Dynamic States
When creating the Vulkan physical device in Rust, enable:
* `dynamic_rendering = true` (Eliminates `VkRenderPass` boilerplate)
* `synchronization2 = true`
* `buffer_device_address = true`
* Extended dynamic state flags (depth testing, stencil, cull mode, winding, blend equations) [85:41, 86:00, 86:40]

### 5.2 The Stripped-Down Pipeline Key
Your engine's pipeline cache key should contain **only** what the hardware requires to generate microcode [44:40, 79:40, 80:00]:

```rust
#[derive(Hash, PartialEq, Eq, Clone)]
pub struct LeanPipelineKey {
    pub vertex_shader: ShaderId,
    pub pixel_shader: ShaderId,
    pub color_formats: smallvec::SmallVec<[vk::Format; 4]>,
    pub depth_format: vk::Format,
    pub sample_count: vk::SampleCountFlags,
}
```

### 5.3 Pipeline Creation
Set dynamic states to strip out all non-essential rasterizer configurations [85:00, 86:40]:

```rust
let dynamic_states = [
    vk::DynamicState::VIEWPORT_WITH_COUNT,
    vk::DynamicState::SCISSOR_WITH_COUNT,
    vk::DynamicState::CULL_MODE,
    vk::DynamicState::FRONT_FACE,
    vk::DynamicState::PRIMITIVE_TOPOLOGY,
    vk::DynamicState::DEPTH_TEST_ENABLE,
    vk::DynamicState::DEPTH_WRITE_ENABLE,
    vk::DynamicState::DEPTH_COMPARE_OP,
];

let dynamic_state_info = vk::PipelineDynamicStateCreateInfo::default()
    .dynamic_states(&dynamic_states);
```

Depth-stencil changes (e.g., swapping from standard opaque testing to decal rendering) are now simple commands on the command buffer (`device.cmd_set_depth_test_enable(...)`), requiring **zero** PSO re-compilations [82:50, 83:10].

---

## Phase 6: Resource-less Synchronization & Barriers

Modern coherent caches mean that synchronization is about **execution order**, not individual image layout state machines [33:20, 68:50, 70:40]. Layout transitions should be eliminated by using `VK_IMAGE_LAYOUT_GENERAL` or keeping attachments in unified formats.

### 6.1 Rust Barrier Implementation
Eliminate CPU hash maps that track per-subresource states [69:11, 71:30]. Emit a single global execution barrier with a 32-bit hazard bitmask:

```rust
bitflags::bitflags! {
    pub struct GpuStages: u32 {
        const TOP               = 1 << 0;
        const DRAW_INDIRECT     = 1 << 1;
        const VERTEX_SHADER     = 1 << 2;
        const PIXEL_SHADER      = 1 << 3;
        const COMPUTE_SHADER    = 1 << 4;
        const COLOR_ATTACHMENT  = 1 << 5;
        const DEPTH_ATTACHMENT  = 1 << 6;
        const TRANSFER          = 1 << 7;
        const ALL               = 1 << 8;
    }
}

bitflags::bitflags! {
    pub struct HazardFlags: u32 {
        const NONE              = 0;
        const INDIRECT_ARGUMENT = 1 << 0;
        const DESCRIPTOR_HEAP   = 1 << 1;
    }
}

pub fn gpu_barrier(
    device: &ash::Device,
    cmd: vk::CommandBuffer,
    before: GpuStages,
    after: GpuStages,
    hazards: HazardFlags,
) {
    let mut access_src = vk::AccessFlags2::MEMORY_WRITE;
    let mut access_dst = vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE;

    if hazards.contains(HazardFlags::INDIRECT_ARGUMENT) {
        access_dst |= vk::AccessFlags2::INDIRECT_COMMAND_READ;
    }

    let memory_barrier = vk::MemoryBarrier2::default()
        .src_stage_mask(map_stages(before))
        .src_access_mask(access_src)
        .dst_stage_mask(map_stages(after))
        .dst_access_mask(access_dst);

    let dep_info = vk::DependencyInfo::default()
        .memory_barriers(std::slice::from_ref(&memory_barrier));

    unsafe {
        device.cmd_pipeline_barrier2(cmd, &dep_info);
    }
}
```

---

## Phase 7: The Complete Rust Draw Loop

Here is how rendering looks in engine space. A worker thread records thousands of draw calls without touching descriptor sets or pipeline hash maps [21:50, 27:10, 59:40]:

```rust
pub fn render_mesh_system(
    device: &ash::Device,
    cmd: vk::CommandBuffer,
    pipeline_layout: vk::PipelineLayout,
    allocator: &FrameBumpAllocator,
    mesh: &GpuMesh,
    material_albedo_idx: u32,
    wvp: glam::Mat4,
) {
    // 1. Allocate root struct directly in ReBAR mapped memory
    let mut alloc = allocator.alloc::<MeshDrawData>().expect("Frame memory full");

    // 2. CPU writes directly to mapped memory (lockless, zero-overhead)
    *alloc.host_ref = MeshDrawData {
        world_view_proj: wvp.to_cols_array(),
        positions: mesh.positions,
        uvs: mesh.uvs,
        albedo_tex: material_albedo_idx,
        normal_tex: material_albedo_idx + 1,
        pbr_tex: material_albedo_idx + 2,
        _padding: 0,
    };

    unsafe {
        // 3. Push EXACTLY ONE 64-bit GPU device address to the pipeline
        let root_gpu_ptr = alloc.device_addr.addr;
        device.cmd_push_constants(
            cmd,
            pipeline_layout,
            vk::ShaderStageFlags::ALL,
            0,
            std::slice::from_raw_parts(
                &root_gpu_ptr as *const u64 as *const u8,
                std::mem::size_of::<u64>(),
            ),
        );

        // 4. Issue the draw call
        device.cmd_draw(cmd, mesh.index_count, 1, 0, 0);
    }
}
```

---

## Roadmap & Migration Path for Your Engine

| Phase | Milestone | Expected Gain |
| :--- | :--- | :--- |
| **1. Memory** | Implement `DeviceAddress<T>` and `FrameBumpAllocator` over ReBAR. | Replaces dynamic uniform buffers and descriptor set pools. |
| **2. Shaders** | Integrate Slang compiler targeting SPIR-V with BDA support. | Enables native 64-bit pointers and shared Rust/Slang POD structs. |
| **3. Textures** | Allocate a global bindless descriptor array (1M slots); assign `TextureId`s. | Removes all descriptor set rebinding logic (`vkCmdBindDescriptorSets`). |
| **4. Geometry** | Set `VertexInputState` to empty; pull vertex data in shaders via `SV_VertexID` [81:30]. | Eliminates vertex layout descriptors and enables simple meshlet evolution. |
| **5. State** | Enable Vulkan 1.3 Extended Dynamic States; strip PSOs to render target formats [85:00, 86:40]. | Reduces pipeline permutations by >90%; completely solves shader compilation stutter [24:40, 27:50, 80:50]. |
| **6. Sync** | Replace resource/layout tracking with pure execution-stage `MemoryBarrier2`. | Removes RHI CPU overhead, enabling true multi-core linear command recording [21:50, 27:10, 69:11]. |