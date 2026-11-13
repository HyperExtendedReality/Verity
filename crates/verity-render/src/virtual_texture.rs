//! Virtual texturing for terrain layers.
//!
//! Each layer is a pair of square UASTC KTX2 files (`scripts/convert_terrain_layers.py`):
//! albedo with height in alpha, and normal, roughness and occlusion. Only the
//! 128-texel pages the camera sees stay on the GPU, in two atlases with a
//! four-texel border round each page so bilinear filtering never reads a
//! neighbour. Levels of 128 texels and below stay resident as ordinary mipped
//! textures, so every lookup has something to fall back to.
//!
//! The resolve pass marks the pages it wanted in a bitset, a sixteenth of the
//! pixels each frame. The CPU reads it back a few frames later, loads missing
//! pages on worker threads, coarsest first, and evicts the least recently
//! wanted. Workers read blocks straight from the mapped files and transcode
//! only the page they need.

use crate::{
    asset::{self, LoadOptions},
    failure, gpu, scene,
    upload::vk_format,
    Renderer, Result, FRAMES,
};
use ash::vk;
use basis_universal::{
    DecodeFlags, LowLevelUastcTranscoder, SliceParametersUastc, TranscoderBlockFormat,
};
use glam::UVec2;
use gpu_allocator::MemoryLocation;
use memmap2::Mmap;
use std::{
    cell::RefCell,
    collections::HashSet,
    fs::File,
    mem,
    ops::Range,
    path::{Path, PathBuf},
    sync::{
        mpsc::{channel, Receiver, Sender},
        Arc,
    },
};

/// Texels along a page's side, without its border.
const PAGE: u32 = 128;
const BORDER: u32 = 4;
/// Texels along a page's side in the atlas.
const SLOT: u32 = PAGE + 2 * BORDER;
/// 4x4 blocks along a slot's side.
const SLOT_BLOCKS: u32 = SLOT / 4;
/// Bytes of one map's page: 16 bytes a block in every supported format.
const PAGE_BYTES: usize = (SLOT_BLOCKS * SLOT_BLOCKS * 16) as usize;
/// Slots along the atlas side: 32 makes 1024 pages in two 4352² atlases, 38 MB as BC7.
const ATLAS_SLOTS: u32 = 32;
/// Pages uploaded per frame, which bounds each frame's copy to about 1.2 MB.
const UPLOAD_BUDGET: usize = 32;
/// Pages being transcoded at once.
const LOADING: usize = 64;
/// Mirrors `shaders/lib/terrain_layers.slang`.
const HEADER_WORDS: usize = 8;
const LAYER_WORDS: usize = 4;
const MAX_LAYERS: usize = 16;
const TABLE_START: usize = HEADER_WORDS + LAYER_WORDS * MAX_LAYERS;
const NO_PAGE: u32 = u32::MAX;

/// What each terrain layer is for. The resolve pass blends them from slope,
/// depth below the original ground, height, wetness and noise.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TerrainRole {
    /// Short grass on gentle, damp ground.
    Grass,
    /// Longer grass, patched in among `Grass`.
    Meadow,
    /// Bare earth between the grass.
    Soil,
    /// Earth patched in among `Soil`.
    Loam,
    /// Dug walls, a metre or two below the original surface.
    Subsoil,
    /// Cliffs and bedrock.
    Rock,
    /// Stony ground high up, below the snow.
    Scree,
    /// River beds: the wettest flat ground.
    Gravel,
    /// Beaches and sea floor.
    Sand,
}

pub const TERRAIN_ROLES: usize = 9;

impl TerrainRole {
    /// Every role, in the order [`Renderer::set_terrain_layers`] takes them.
    pub const ALL: [Self; TERRAIN_ROLES] = [
        Self::Grass,
        Self::Meadow,
        Self::Soil,
        Self::Loam,
        Self::Subsoil,
        Self::Rock,
        Self::Scree,
        Self::Gravel,
        Self::Sand,
    ];
}

#[derive(Clone, Debug)]
pub struct TerrainLayer {
    /// Base color with height in alpha.
    pub albedo: PathBuf,
    /// Normal in red and green, roughness in blue, ambient occlusion in alpha.
    pub surface: PathBuf,
    /// Metres the texture covers before it repeats.
    pub tile_size: f32,
}

impl TerrainLayer {
    /// A layer from `scripts/convert_terrain_layers.py`'s output: `<dir>/<name>_albedo.ktx2`
    /// and `<dir>/<name>_surface.ktx2`.
    pub fn converted(dir: impl AsRef<Path>, name: &str, tile_size: f32) -> Self {
        let dir = dir.as_ref();
        Self {
            albedo: dir.join(format!("{name}_albedo.ktx2")),
            surface: dir.join(format!("{name}_surface.ktx2")),
            tile_size,
        }
    }
}

/// How terrain layer streaming went in the last frame.
#[derive(Clone, Copy, Debug, Default)]
pub struct TerrainTextureStats {
    /// Pages on the GPU, and room for them.
    pub resident: usize,
    pub capacity: usize,
    /// Pages the last read-back asked for, with their coarser parents.
    pub wanted: usize,
    /// Pages being transcoded.
    pub loading: usize,
    /// Pages uploaded so far.
    pub uploaded: u64,
}

/// One map of one layer: a mapped file and where each level's blocks are.
struct Map {
    file: Mmap,
    levels: Vec<Range<usize>>,
    size: u32,
}

impl Map {
    fn open(path: &Path) -> Result<Self> {
        let named = |e: &dyn std::fmt::Display| failure(format!("{}: {e}", path.display()));
        let file = File::open(path).map_err(|e| named(&e))?;
        // shortcut: the file must not change while mapped; copy it to memory if
        // layers are ever edited while the renderer runs.
        let file = unsafe { Mmap::map(&file) }.map_err(|e| named(&e))?;
        let (levels, size) = {
            let reader = ktx2::Reader::new(&file[..]).map_err(|e| named(&format!("{e:?}")))?;
            let header = reader.header();
            let uastc = (reader.basic_dfd())
                .is_some_and(|d| d.color_model == Some(ktx2::ColorModel::UASTC));
            let size = header.pixel_width;
            if !uastc || header.format.is_some() || header.supercompression_scheme.is_some() {
                return Err(named(&"expected UASTC without supercompression (run scripts/convert_terrain_layers.py)"));
            }
            if header.pixel_height != size || !size.is_power_of_two() || size < 2 * PAGE {
                return Err(named(&format!(
                    "expected a square power-of-two size of at least {}",
                    2 * PAGE
                )));
            }
            if header.layer_count > 1 || header.face_count != 1 || header.pixel_depth > 1 {
                return Err(named(&"expected a single 2D texture"));
            }
            let base = file.as_ptr() as usize;
            let levels: Vec<_> = (reader.levels())
                .map(|level| {
                    let start = level.data.as_ptr() as usize - base;
                    start..start + level.data.len()
                })
                .collect();
            (levels, size)
        };
        let layout = Layout::new(size);
        for (index, level) in levels.iter().enumerate().take(layout.mips as usize + 1) {
            let blocks = (size >> index) as usize / 4;
            if level.len() != blocks * blocks * 16 {
                return Err(named(&format!("level {index} is not UASTC 4x4 blocks")));
            }
        }
        if levels.len() <= layout.mips as usize {
            return Err(named(&"needs mip levels down to 128 texels"));
        }
        Ok(Self { file, levels, size })
    }
}

/// How pages of a layer are numbered: every page of the largest level, row by
/// row, then the next level's, down to the last level with more than one
/// page. Layers follow one another.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Layout {
    /// Pages along the largest level's side.
    across: u32,
    /// Levels split into pages; the next one is the resident tail's largest.
    mips: u32,
    per_layer: u32,
}

impl Layout {
    fn new(size: u32) -> Self {
        let across = size / PAGE;
        let mips = across.ilog2();
        let per_layer = (0..mips).map(|m| (across >> m).pow(2)).sum();
        Self {
            across,
            mips,
            per_layer,
        }
    }

    fn offset(&self, mip: u32) -> u32 {
        (0..mip).map(|m| (self.across >> m).pow(2)).sum()
    }

    fn id(&self, layer: u32, mip: u32, page: UVec2) -> u32 {
        let across = self.across >> mip;
        layer * self.per_layer + self.offset(mip) + page.y * across + page.x
    }

    /// Layer, level and page of an id.
    fn page(&self, id: u32) -> (u32, u32, UVec2) {
        let (layer, mut rest) = (id / self.per_layer, id % self.per_layer);
        let mut mip = 0;
        while rest >= (self.across >> mip).pow(2) {
            rest -= (self.across >> mip).pow(2);
            mip += 1;
        }
        let across = self.across >> mip;
        (layer, mip, UVec2::new(rest % across, rest / across))
    }

    /// The page covering this one at the next level, unless that is the tail.
    fn parent(&self, id: u32) -> Option<u32> {
        let (layer, mip, page) = self.page(id);
        (mip + 1 < self.mips).then(|| self.id(layer, mip + 1, page / 2))
    }
}

/// The UASTC blocks of one page and its border, wrapping round the texture's
/// edges as tiling does.
fn gather(level: &[u8], blocks_across: u32, page: UVec2) -> Vec<u8> {
    let first = (page * (PAGE / 4)).as_ivec2() - (BORDER / 4) as i32;
    let mut out = Vec::with_capacity(PAGE_BYTES);
    for y in 0..SLOT_BLOCKS as i32 {
        let row = (first.y + y).rem_euclid(blocks_across as i32) as usize;
        for x in 0..SLOT_BLOCKS as i32 {
            let column = (first.x + x).rem_euclid(blocks_across as i32) as usize;
            let at = (row * blocks_across as usize + column) * 16;
            out.extend_from_slice(&level[at..at + 16]);
        }
    }
    out
}

/// Both maps of a page, transcoded for the GPU.
fn load_page(
    maps: &[[Map; 2]],
    layout: Layout,
    id: u32,
    format: TranscoderBlockFormat,
) -> std::result::Result<Vec<u8>, String> {
    thread_local! {
        static TRANSCODER: RefCell<Option<LowLevelUastcTranscoder>> = const { RefCell::new(None) };
    }
    let (layer, mip, page) = layout.page(id);
    let mut out = Vec::with_capacity(2 * PAGE_BYTES);
    for map in &maps[layer as usize] {
        let level = &map.file[map.levels[mip as usize].clone()];
        let blocks = gather(level, (map.size >> mip) / 4, page);
        let params = SliceParametersUastc {
            num_blocks_x: SLOT_BLOCKS,
            num_blocks_y: SLOT_BLOCKS,
            // Both maps carry data in alpha.
            has_alpha: true,
            original_width: SLOT,
            original_height: SLOT,
        };
        let transcoded = TRANSCODER.with_borrow_mut(|t| {
            let transcoder = t.get_or_insert_with(LowLevelUastcTranscoder::new);
            transcoder.transcode_slice(&blocks, params, DecodeFlags::HIGH_QUALITY, format)
        });
        let transcoded = transcoded.map_err(|e| format!("transcode page {id}: {e:?}"))?;
        let bytes = (transcoded.get(..PAGE_BYTES))
            .ok_or_else(|| format!("transcode page {id}: short output"))?;
        out.extend_from_slice(bytes);
    }
    Ok(out)
}

/// Which page each atlas slot holds, and when it was last wanted.
#[derive(Default)]
struct Slots {
    held: Vec<Option<(u32, u64)>>,
}

impl Slots {
    /// A free slot, else the one wanted longest ago before `now`.
    fn take(&self, now: u64) -> Option<usize> {
        if let Some(free) = self.held.iter().position(Option::is_none) {
            return Some(free);
        }
        (self.held.iter().enumerate())
            .filter_map(|(slot, held)| {
                held.filter(|(_, used)| *used < now)
                    .map(|(_, used)| (slot, used))
            })
            .min_by_key(|(_, used)| *used)
            .map(|(slot, _)| slot)
    }
}

pub(crate) struct TerrainLayers {
    maps: Arc<Vec<[Map; 2]>>,
    layout: Layout,
    tile_sizes: Vec<f32>,
    format: TranscoderBlockFormat,
    /// Indices in `Renderer::textures`: the two atlases, then each layer's two tails.
    atlases: [u32; 2],
    tails: Vec<[u32; 2]>,
    /// Page id to slot, or `NO_PAGE`.
    table: Vec<u32>,
    slots: Slots,
    loading: HashSet<u32>,
    sender: Sender<(u32, std::result::Result<Vec<u8>, String>)>,
    receiver: Receiver<(u32, std::result::Result<Vec<u8>, String>)>,
    /// This frame's uploads: slot and offset in the frame's staging buffer.
    pub(crate) copies: Vec<(u32, u64)>,
    stats: TerrainTextureStats,
    /// Whether a page failed to load, so it is reported once.
    failed: bool,
}

impl TerrainLayers {
    fn feedback_words(&self) -> usize {
        (self.table.len()).div_ceil(32)
    }

    /// Ids set in a read-back bitset, with their parents, coarsest level first.
    fn wanted(&self, bits: &[u32]) -> Vec<u32> {
        let mut wanted = HashSet::new();
        for (word, &value) in bits.iter().enumerate() {
            let mut value = value;
            while value != 0 {
                let mut id = Some(word as u32 * 32 + value.trailing_zeros());
                value &= value - 1;
                while let Some(page) = id.filter(|&p| (p as usize) < self.table.len()) {
                    if !wanted.insert(page) {
                        break;
                    }
                    id = self.layout.parent(page);
                }
            }
        }
        let mut wanted: Vec<u32> = wanted.into_iter().collect();
        wanted.sort_by_key(|&id| std::cmp::Reverse(self.layout.page(id).1));
        wanted
    }

    /// The table the resolve pass reads: header, layers, then page slots.
    fn words(&self, frame: u64) -> Vec<u32> {
        let mut words = vec![0; TABLE_START];
        words[..HEADER_WORDS].copy_from_slice(&[
            frame as u32,
            self.tails.len() as u32,
            self.atlases[0],
            self.atlases[1],
            self.layout.across,
            self.layout.mips,
            self.layout.per_layer,
            ATLAS_SLOTS,
        ]);
        for (layer, (tails, size)) in self.tails.iter().zip(&self.tile_sizes).enumerate() {
            let at = HEADER_WORDS + layer * LAYER_WORDS;
            words[at..at + 3].copy_from_slice(&[tails[0], tails[1], size.to_bits()]);
        }
        words.extend_from_slice(&self.table);
        words
    }
}

impl Renderer {
    /// Texture the terrain with these layers, one per [`TerrainRole`] in
    /// [`TerrainRole::ALL`] order, all the same size. Replaces any set before.
    /// [`Renderer::clear_gltf`] removes them, with every other texture.
    pub fn set_terrain_layers(&mut self, layers: &[TerrainLayer; TERRAIN_ROLES]) -> Result<()> {
        let maps: Vec<[Map; 2]> = (layers.iter())
            .map(|l| Ok([Map::open(&l.albedo)?, Map::open(&l.surface)?]))
            .collect::<Result<_>>()?;
        let size = maps[0][0].size;
        if maps.iter().flatten().any(|m| m.size != size) {
            return Err(failure("terrain layers must all be the same size"));
        }
        if let Some(bad) = layers
            .iter()
            .find(|l| !(l.tile_size.is_finite() && l.tile_size > 0.0))
        {
            return Err(failure(format!(
                "{}: tile size must be positive",
                bad.albedo.display()
            )));
        }
        let layout = Layout::new(size);
        let replacing = self.terrain_layers.take();
        let added = if replacing.is_some() {
            0
        } else {
            2 + 2 * TERRAIN_ROLES
        };
        if self.textures.len() + added > scene::MAX_TEXTURES as usize {
            return Err(failure(format!(
                "more than {} textures loaded",
                scene::MAX_TEXTURES
            )));
        }
        let options = LoadOptions {
            format: self.texture_format,
            max_texture_size: PAGE,
        };
        unsafe {
            self.device.device_wait_idle()?;
            let mut tails = Vec::with_capacity(TERRAIN_ROLES);
            for (layer, pair) in maps.iter().enumerate() {
                let mut indices = [0; 2];
                for (map_index, map) in pair.iter().enumerate() {
                    let tail = asset::decode_ktx2(&map.file, options)
                        .map_err(|e| failure(e.to_string()))?;
                    let image = self.upload_texture(&tail)?;
                    indices[map_index] = match &replacing {
                        Some(old) => {
                            let index = old.tails[layer][map_index];
                            let old = mem::replace(&mut self.textures[index as usize], image);
                            gpu::destroy_image(&self.device, self.allocator.as_mut().unwrap(), old);
                            index
                        }
                        None => {
                            self.textures.push(image);
                            self.textures.len() as u32 - 1
                        }
                    };
                }
                tails.push(indices);
            }
            let atlases = match &replacing {
                Some(old) => old.atlases,
                None => {
                    let mut atlases = [0; 2];
                    for (atlas, srgb) in atlases.iter_mut().zip([true, false]) {
                        let image = self.create_atlas(srgb)?;
                        self.textures.push(image);
                        *atlas = self.textures.len() as u32 - 1;
                    }
                    atlases
                }
            };
            self.write_texture_descriptors();
            let (sender, receiver) = channel();
            let layers = TerrainLayers {
                maps: Arc::new(maps),
                layout,
                tile_sizes: layers.iter().map(|l| l.tile_size).collect(),
                format: self.texture_format.block(),
                atlases,
                tails,
                table: vec![NO_PAGE; (layout.per_layer as usize) * TERRAIN_ROLES],
                slots: Slots {
                    held: vec![None; (ATLAS_SLOTS * ATLAS_SLOTS) as usize],
                },
                loading: HashSet::new(),
                sender,
                receiver,
                copies: Vec::new(),
                stats: TerrainTextureStats {
                    capacity: (ATLAS_SLOTS * ATLAS_SLOTS) as usize,
                    ..Default::default()
                },
                failed: false,
            };
            // Room for the table and the feedback in every frame, all zero.
            let table_bytes = (layers.words(0).len() * 4) as u64;
            let feedback_bytes = (layers.feedback_words() * 4) as u64;
            let allocator = self.allocator.as_mut().unwrap();
            for frame in &mut self.frames {
                for (buffer, name, size, location) in [
                    (
                        &mut frame.layer_table,
                        "terrain layer table",
                        table_bytes,
                        MemoryLocation::CpuToGpu,
                    ),
                    (
                        &mut frame.layer_feedback,
                        "terrain layer feedback",
                        feedback_bytes,
                        MemoryLocation::GpuToCpu,
                    ),
                ] {
                    if buffer.size < size {
                        gpu::destroy_buffer(&self.device, allocator, mem::take(buffer));
                        *buffer = gpu::create_buffer(
                            &self.device,
                            allocator,
                            name,
                            size,
                            vk::BufferUsageFlags::STORAGE_BUFFER,
                            location,
                        )?;
                    }
                    buffer.write(&vec![0; buffer.size as usize]);
                }
            }
            for index in 0..FRAMES {
                self.write_buffer_descriptors(index);
            }
            self.terrain_layers = Some(layers);
        }
        Ok(())
    }

    /// How terrain layer streaming went in the last frame; all zero without layers.
    pub fn terrain_texture_stats(&self) -> TerrainTextureStats {
        self.terrain_layers
            .as_ref()
            .map_or_else(Default::default, |l| l.stats)
    }

    /// A page atlas, laid out for sampling; its contents are undefined until pages arrive.
    unsafe fn create_atlas(&mut self, srgb: bool) -> Result<gpu::Image> {
        let side = ATLAS_SLOTS * SLOT;
        let image = gpu::create_image(
            &self.device,
            self.allocator.as_mut().unwrap(),
            "terrain page atlas",
            vk_format(self.texture_format, srgb),
            vk::Extent2D {
                width: side,
                height: side,
            },
            1,
            vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
            vk::ImageAspectFlags::COLOR,
        )?;
        let barrier = vk::ImageMemoryBarrier2::default()
            .image(image.image)
            .subresource_range(gpu::subresource(vk::ImageAspectFlags::COLOR))
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
            .dst_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ);
        let device = &self.device;
        let result = self.submit_once(|cmd| {
            device.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default().image_memory_barriers(&[barrier]),
            )
        });
        match result {
            Ok(()) => Ok(image),
            Err(e) => {
                gpu::destroy_image(&self.device, self.allocator.as_mut().unwrap(), image);
                Err(e)
            }
        }
    }

    /// Read back what the frame that last used this slot wanted, start loading
    /// it, place finished pages and write this frame's table. The frame's fence
    /// has signaled.
    pub(crate) unsafe fn stream_terrain_pages(&mut self) -> Result<()> {
        let now = self.frame_number;
        let frame = &mut self.frames[self.frame];
        let Some(layers) = &mut self.terrain_layers else {
            return Ok(());
        };
        layers.copies.clear();
        let words = layers.feedback_words();
        let bits: Vec<u32> =
            bytemuck::cast_slice(&frame.layer_feedback.read()[..words * 4]).to_vec();
        frame.layer_feedback.write(&vec![0; words * 4]);
        let wanted = layers.wanted(&bits);
        // A read-back with nothing in it says nothing about what is in view.
        if !wanted.is_empty() {
            layers.stats.wanted = wanted.len();
        }
        for &id in &wanted {
            let slot = layers.table[id as usize];
            if slot != NO_PAGE {
                layers.slots.held[slot as usize] = Some((id, now));
            }
        }
        let missing = (wanted.iter())
            .filter(|&&id| layers.table[id as usize] == NO_PAGE && !layers.loading.contains(&id));
        let room = LOADING.saturating_sub(layers.loading.len());
        for &id in missing.take(room).collect::<Vec<_>>() {
            layers.loading.insert(id);
            let (maps, layout, format) = (layers.maps.clone(), layers.layout, layers.format);
            let sender = layers.sender.clone();
            rayon::spawn(move || {
                // Only fails once the layers are gone and nobody wants the page.
                let _ = sender.send((id, load_page(&maps, layout, id, format)));
            });
        }
        let mut staging = Vec::new();
        while layers.copies.len() < UPLOAD_BUDGET {
            let Ok((id, page)) = layers.receiver.try_recv() else {
                break;
            };
            layers.loading.remove(&id);
            let page = match page {
                Ok(page) => page,
                Err(error) => {
                    if !mem::replace(&mut layers.failed, true) {
                        eprintln!("terrain layers: {error}");
                    }
                    continue;
                }
            };
            // Every slot was wanted this frame: the view needs more than fits.
            let Some(slot) = layers.slots.take(now) else {
                continue;
            };
            if let Some((old, _)) = layers.slots.held[slot] {
                layers.table[old as usize] = NO_PAGE;
            }
            layers.slots.held[slot] = Some((id, now));
            layers.table[id as usize] = slot as u32;
            layers.copies.push((slot as u32, staging.len() as u64));
            staging.extend_from_slice(&page);
        }
        layers.stats.uploaded += layers.copies.len() as u64;
        layers.stats.loading = layers.loading.len();
        layers.stats.resident = layers.slots.held.iter().flatten().count();
        frame
            .layer_table
            .write(bytemuck::cast_slice(&layers.words(now)));
        if !staging.is_empty() {
            let size = staging.len() as u64;
            if frame.layer_staging.size < size {
                let allocator = self.allocator.as_mut().unwrap();
                gpu::destroy_buffer(&self.device, allocator, mem::take(&mut frame.layer_staging));
                frame.layer_staging = gpu::create_buffer(
                    &self.device,
                    allocator,
                    "terrain page staging",
                    size.next_power_of_two(),
                    vk::BufferUsageFlags::TRANSFER_SRC,
                    MemoryLocation::CpuToGpu,
                )?;
            }
            frame.layer_staging.write(&staging);
        }
        Ok(())
    }

    /// Copy this frame's new pages into the atlases, before the resolve samples them.
    pub(crate) unsafe fn record_terrain_pages(&self, cmd: vk::CommandBuffer) {
        let Some(layers) = self
            .terrain_layers
            .as_ref()
            .filter(|l| !l.copies.is_empty())
        else {
            return;
        };
        use vk::{AccessFlags2 as A, ImageLayout as L, PipelineStageFlags2 as S};
        let atlases = layers
            .atlases
            .map(|index| self.textures[index as usize].image);
        let barriers = |old, new, src: (S, A), dst: (S, A)| {
            atlases.map(|image| {
                vk::ImageMemoryBarrier2::default()
                    .image(image)
                    .subresource_range(gpu::subresource(vk::ImageAspectFlags::COLOR))
                    .old_layout(old)
                    .new_layout(new)
                    .src_stage_mask(src.0)
                    .src_access_mask(src.1)
                    .dst_stage_mask(dst.0)
                    .dst_access_mask(dst.1)
            })
        };
        let shaders = S::COMPUTE_SHADER | S::FRAGMENT_SHADER;
        // Earlier frames may still sample the slots being replaced.
        let before = barriers(
            L::SHADER_READ_ONLY_OPTIMAL,
            L::TRANSFER_DST_OPTIMAL,
            (shaders, A::NONE),
            (S::COPY, A::TRANSFER_WRITE),
        );
        let after = barriers(
            L::TRANSFER_DST_OPTIMAL,
            L::SHADER_READ_ONLY_OPTIMAL,
            (S::COPY, A::TRANSFER_WRITE),
            (shaders, A::SHADER_SAMPLED_READ),
        );
        let device = &self.device;
        device.cmd_pipeline_barrier2(
            cmd,
            &vk::DependencyInfo::default().image_memory_barriers(&before),
        );
        let staging = self.frames[self.frame].layer_staging.buffer;
        for (map, image) in atlases.into_iter().enumerate() {
            let regions: Vec<_> = (layers.copies.iter())
                .map(|&(slot, offset)| {
                    vk::BufferImageCopy::default()
                        .buffer_offset(offset + (map * PAGE_BYTES) as u64)
                        .image_subresource(
                            vk::ImageSubresourceLayers::default()
                                .aspect_mask(vk::ImageAspectFlags::COLOR)
                                .layer_count(1),
                        )
                        .image_offset(vk::Offset3D {
                            x: (slot % ATLAS_SLOTS * SLOT) as i32,
                            y: (slot / ATLAS_SLOTS * SLOT) as i32,
                            z: 0,
                        })
                        .image_extent(vk::Extent3D {
                            width: SLOT,
                            height: SLOT,
                            depth: 1,
                        })
                })
                .collect();
            device.cmd_copy_buffer_to_image(cmd, staging, image, L::TRANSFER_DST_OPTIMAL, &regions);
        }
        device.cmd_pipeline_barrier2(
            cmd,
            &vk::DependencyInfo::default().image_memory_barriers(&after),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn page_ids_round_trip_and_climb_to_the_tail() {
        let layout = Layout::new(4096);
        assert_eq!(
            (layout.across, layout.mips, layout.per_layer),
            (32, 5, 1364)
        );
        for id in [0, 1023, 1024, 1363, 1364, 3 * 1364 + 1300] {
            let (layer, mip, page) = layout.page(id);
            assert_eq!(layout.id(layer, mip, page), id);
        }
        let mut id = layout.id(2, 0, UVec2::new(31, 17));
        let mut chain = vec![id];
        while let Some(parent) = layout.parent(id) {
            id = parent;
            chain.push(id);
        }
        let mips: Vec<_> = chain.iter().map(|&id| layout.page(id).1).collect();
        assert_eq!(mips, [0, 1, 2, 3, 4]);
        assert_eq!(layout.page(id), (2, 4, UVec2::new(1, 1)));
    }

    #[test]
    fn pages_gather_their_border_from_across_the_seam() {
        // Each block holds its own column and row.
        let across = 64u32;
        let level: Vec<u8> = (0..across * across)
            .flat_map(|i| [(i % across) as u8, (i / across) as u8].repeat(8))
            .collect();
        let blocks = gather(&level, across, UVec2::new(0, 1));
        assert_eq!(blocks.len(), PAGE_BYTES);
        let block = |x: usize, y: usize| {
            let at = (y * SLOT_BLOCKS as usize + x) * 16;
            (blocks[at], blocks[at + 1])
        };
        // One block of border: the last column, and the row above the page.
        assert_eq!(block(0, 0), (63, 31));
        assert_eq!(block(1, 1), (0, 32));
        assert_eq!(block(33, 33), (32, 64 % 64));
    }

    #[test]
    fn eviction_takes_free_slots_then_the_oldest_not_wanted_now() {
        let mut slots = Slots {
            held: vec![Some((7, 3)), None, Some((8, 1))],
        };
        assert_eq!(slots.take(5), Some(1));
        slots.held[1] = Some((9, 5));
        assert_eq!(slots.take(5), Some(2));
        slots.held[2] = Some((10, 5));
        slots.held[0] = Some((7, 5));
        assert_eq!(slots.take(5), None, "everything is in view");
    }

    #[test]
    fn converted_layers_load_pages() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../assets/packed/terrain");
        if !dir.join("Grass002_albedo.ktx2").exists() {
            return; // Converted assets are local only.
        }
        let albedo = Map::open(&dir.join("Grass002_albedo.ktx2")).unwrap();
        let surface = Map::open(&dir.join("Grass002_surface.ktx2")).unwrap();
        let layout = Layout::new(albedo.size);
        let maps = vec![[albedo, surface]];
        for id in [0, layout.per_layer - 1] {
            let page = load_page(&maps, layout, id, TranscoderBlockFormat::BC7).unwrap();
            assert_eq!(page.len(), 2 * PAGE_BYTES);
        }
    }
}
