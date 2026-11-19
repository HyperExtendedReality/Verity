"""Convert the cloud VDBs in assets/packed/clouds to what the renderer embeds.

Run with Blender 5.3+, whose Python bundles OpenVDB and numpy:

    blender -b --factory-startup --python scripts/convert_clouds.py

The source VDBs sit beside their outputs (*.vdb is gitignored). Outputs, z up
(Verity's):

- nubis_noise.ktx2: Guerrilla's 128^3 tiling detail noise, R8G8B8A8_UNORM, 3D,
  Zstd-supercompressed, a full mip chain. R, G: low and high frequency curly
  alligator (wisps); B, A: low and high frequency alligator (billows).
- <cloud>.vqcloud: a voxel cloudscape stored as VQVDB stores a grid (Zephir,
  2025): only the 8^3 bricks that hold cloud, each as 64 one-byte codes, one
  per 2^3 block, into a shared 256-entry codebook. Where VQVDB learns its
  codebook and a decoder network, this one is k-means over gain-normalized
  blocks, so decoding is a lookup and a multiply. Nubis³ (Schneider, 2023)
  keeps dense 512x512x64 NVDFs plus a dense SDF; here the bricks stand in for
  both, and a grid of distances between bricks replaces the SDF for sphere
  tracing. Channels: the dimensional profile per voxel; detail type and
  density scale per brick (they vary over hundreds of metres).

.vqcloud layout, little endian:
    b"VQCLOUD1", u32 grid[3] (bricks along x, y, z), u32 bricks, u32 codes,
    f32 voxel (metres), u32 json bytes, JSON (source credit), padded to 4,
    then the GPU payload: f32 codebook[codes * 8], u32 brick[bricks]
    (x | y << 10 | z << 20), f32 gain[bricks], u32 attributes[bricks]
    (detail type | density scale << 8), u8 code[bricks * 64]; then
    u8 distance[z][y][x] (bricks to the nearest cloudy brick, rounded down).
Codes in a brick run x fastest over its 4^3 blocks; codebook entries run x
fastest over a block's 2^3 voxels.
"""

import json
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import openvdb as vdb

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "assets" / "packed" / "clouds"
NOISE_CHANNELS = ("lf_curly_worley", "hf_curly_worley", "lf_alligator", "hf_alligator")
NVDF_CHANNELS = ("dimensional_profile", "detail_type", "density_scale")
NUBIS_CREDIT = "Nubis Voxel Clouds Pack, Guerrilla Games (Andrew Schneider), 2023"
WDAS_CREDIT = ("Walt Disney Animation Studios Cloud Data Set, Copyright 2017 Disney "
               "Enterprises, Inc., CC BY-SA 3.0")
# The quarter-resolution WDAS cloud (0.83 m voxels) is some 400 m across;
# scaled up to a 2 km cumulus congestus its voxels are 3.3 m, finer than
# Nubis' 8 m.
WDAS_SCALE = 4.0
# WDAS holds density, not a dimensional profile; Nubis derives the profile
# from the cloud's signed distance, rising from 0 at its surface to 1 this
# many metres in, so erosion frays the edges and spares the core.
PROFILE_DEPTH = 200.0
BRICK = 8
CODES = 256
VK_FORMAT_R8G8B8A8_UNORM = 37


def ktx_tool():
    found = shutil.which("ktx") or r"C:\Program Files\KTX-Software\bin\ktx.exe"
    if not Path(found).exists():
        sys.exit("ktx (KTX-Software 4.3+) not found on PATH")
    return found


def dense(path, name):
    """A grid as a float array indexed [x, y, z] in VDB index space, and its
    index-space origin and voxel size."""
    grid = vdb.read(str(path), name)
    lo, hi = grid.evalActiveVoxelBoundingBox()
    array = np.zeros([b - a + 1 for a, b in zip(lo, hi)], np.float32)
    grid.copyToArray(array, ijk=lo)
    return array, np.array(lo), grid.transform.voxelSize()[0]


def z_up(array):
    """VDB's y-up [x, y, z] to texture order [z, y, x] in Verity's z-up frame,
    where Verity (x, y, z) = VDB (x, -z, y)."""
    return np.ascontiguousarray(np.flip(np.transpose(array, (1, 2, 0)), axis=1))


def quantize(array):
    return np.clip(np.rint(array * 255.0), 0, 255).astype(np.uint8)


def downsample(level):
    """Halve a tiling [z, y, x, c] level with a 2x2x2 box filter."""
    return (level[0::2, 0::2, 0::2] + level[1::2, 0::2, 0::2] + level[0::2, 1::2, 0::2] + level[1::2, 1::2, 0::2]
            + level[0::2, 0::2, 1::2] + level[1::2, 0::2, 1::2] + level[0::2, 1::2, 1::2] + level[1::2, 1::2, 1::2]) / 8.0


def edt(inside):
    """Euclidean distance in voxels from each voxel to the nearest `inside` one,
    exact, by squared distances minimised one axis at a time."""
    big = float(sum(n * n for n in inside.shape))
    squared = np.where(inside, 0.0, big).astype(np.float32)
    for axis in range(3):
        moved = np.moveaxis(squared, axis, -1)
        n = moved.shape[-1]
        lines = moved.reshape(-1, n)
        offsets = np.arange(n, dtype=np.float32)
        gap = (offsets[:, None] - offsets[None, :]) ** 2
        result = np.empty_like(lines)
        step = max(1, (1 << 26) // (n * n))  # ~256 MB of candidates at a time
        for start in range(0, len(lines), step):
            chunk = lines[start:start + step]
            result[start:start + step] = (chunk[:, None, :] + gap[None]).min(axis=2)
        squared = np.moveaxis(result.reshape(moved.shape), -1, axis)
    return np.sqrt(squared)


def write_ktx2(path, levels, metadata):
    """Write an uncompressed R8G8B8A8_UNORM 3D KTX2; `levels` are [z, y, x, 4]
    uint8 arrays, largest first."""
    depth, height, width, _ = levels[0].shape
    samples = b"".join(
        struct.pack("<HBB4sII", 8 * index, 7, channel, bytes(4), 0, 255)
        for index, channel in enumerate((0, 1, 2, 15)))
    block = struct.pack("<IHHBBBB4s8s", 0, 2, 24 + len(samples), 1, 1, 1, 0, bytes(4),
                        bytes([4, 0, 0, 0, 0, 0, 0, 0])) + samples
    dfd = struct.pack("<I", 4 + len(block)) + block

    def entry(key, value):
        body = key.encode() + b"\0" + value.encode() + b"\0"
        return struct.pack("<I", len(body)) + body + bytes(-len(body) % 4)

    kvd = entry("KTXwriter", "Verity convert_clouds.py") + entry("verity.cloud", json.dumps(metadata))
    index_size = 80 + 24 * len(levels)
    dfd_offset = index_size
    kvd_offset = dfd_offset + len(dfd)
    data_offset = kvd_offset + len(kvd)
    data_offset += -data_offset % 4
    # Levels are stored smallest first, each 4-byte aligned.
    offsets, cursor = {}, data_offset
    for number in reversed(range(len(levels))):
        offsets[number] = cursor
        cursor += levels[number].nbytes
        cursor += -cursor % 4
    with open(path, "wb") as out:
        out.write(b"\xabKTX 20\xbb\r\n\x1a\n")
        out.write(struct.pack("<9I", VK_FORMAT_R8G8B8A8_UNORM, 1, width, height, depth, 0, 1, len(levels), 0))
        out.write(struct.pack("<4I2Q", dfd_offset, len(dfd), kvd_offset, len(kvd), 0, 0))
        for number, level in enumerate(levels):
            out.write(struct.pack("<3Q", offsets[number], level.nbytes, level.nbytes))
        out.write(dfd)
        out.write(kvd)
        for number in reversed(range(len(levels))):
            out.write(bytes(offsets[number] - out.tell()))
            out.write(levels[number].tobytes())


def publish(ktx, levels, metadata, name):
    """Write, Zstd-supercompress and validate `name`.ktx2."""
    raw = OUTPUT / f"{name}.raw.ktx2"
    target = OUTPUT / f"{name}.ktx2"
    write_ktx2(raw, levels, metadata)
    subprocess.run([ktx, "deflate", "--zstd", "19", str(raw), str(target)], check=True)
    raw.unlink()
    subprocess.run([ktx, "validate", str(target)], check=True)
    size = sum(level.nbytes for level in levels)
    print(f"  {target.name}: {levels[0].shape[2::-1]} voxels, {size / 1e6:.1f} MB on the GPU, "
          f"{target.stat().st_size / 1e6:.1f} MB on disk", flush=True)


def noise(ktx):
    path = OUTPUT / "NubisVoxelCloudNoise.vdb"
    level = np.stack([z_up(dense(path, name)[0]) for name in NOISE_CHANNELS], axis=-1)
    assert level.shape == (128, 128, 128, 4), level.shape
    levels = [level]
    while levels[-1].shape[0] > 1:
        levels.append(downsample(levels[-1]))
    metadata = {"kind": "noise", "channels": list(NOISE_CHANNELS), "source": NUBIS_CREDIT}
    publish(ktx, [quantize(level) for level in levels], metadata, "nubis_noise")


def bricks(field):
    """A [z, y, x] array padded to whole bricks, as [bz, by, bx, 8, 8, 8]."""
    field = np.pad(field, [(0, -n % BRICK) for n in field.shape])
    bz, by, bx = (n // BRICK for n in field.shape)
    return field.reshape(bz, BRICK, by, BRICK, bx, BRICK).transpose(0, 2, 4, 1, 3, 5)


def assign(vectors, codebook):
    """The nearest codebook entry to each vector."""
    out = np.empty(len(vectors), np.uint8)
    lengths = (codebook ** 2).sum(axis=1)
    for start in range(0, len(vectors), 1 << 18):
        chunk = vectors[start:start + (1 << 18)]
        out[start:start + len(chunk)] = np.argmin(lengths[None] - 2.0 * chunk @ codebook.T, axis=1)
    return out


def kmeans(vectors, rng):
    """A `CODES`-entry codebook for `vectors` by Lloyd's algorithm on a sample;
    entry 0 stays the empty block, the most common one at cloud edges."""
    sample = vectors[rng.choice(len(vectors), min(len(vectors), 400_000), replace=False)]
    codebook = sample[rng.choice(len(sample), CODES, replace=len(sample) < CODES)].copy()
    codebook[0] = 0.0
    for _ in range(40):
        labels = assign(sample, codebook)
        sums = np.zeros_like(codebook)
        np.add.at(sums, labels, sample)
        counts = np.bincount(labels, minlength=CODES)
        filled = counts > 0
        codebook[filled] = sums[filled] / counts[filled, None]
        # Reseed unused entries on random samples.
        codebook[~filled] = sample[rng.choice(len(sample), (~filled).sum())]
        codebook[0] = 0.0
    return codebook.astype(np.float32)


def vq_cloudscape(name, profile, detail_type, density_scale, voxel, credit):
    """Encode [z, y, x] fields (Verity order) as `name`.vqcloud."""
    profile = profile / profile.max()
    grids = [bricks(f) for f in (profile, detail_type, density_scale)]
    occupied = grids[0].reshape(*grids[0].shape[:3], -1).max(axis=-1) > 0.0
    # Crop to the cloudy bricks: outside the box the ray march skips ahead.
    found = np.argwhere(occupied)
    crop = tuple(slice(a, b + 1) for a, b in zip(found.min(axis=0), found.max(axis=0)))
    occupied = occupied[crop]
    grids = [g[crop] for g in grids]
    blocks, types, scales = (g[occupied] for g in grids)
    count = len(blocks)
    gain = blocks.reshape(count, -1).max(axis=1)
    normalized = blocks / gain[:, None, None, None]
    # [brick, z, y, x] -> [brick, block z, block y, block x, z, y, x]
    vectors = normalized.reshape(count, 4, 2, 4, 2, 4, 2).transpose(0, 1, 3, 5, 2, 4, 6).reshape(-1, 8)
    codebook = kmeans(vectors.astype(np.float32), np.random.default_rng(1))
    codes = assign(vectors, codebook)
    error = (codebook[codes].reshape(count, 64, 8) - vectors.reshape(count, 64, 8)) * gain[:, None, None]
    weight = blocks.reshape(count, -1).sum(axis=1) + 1e-6

    def mean(field):
        """Profile-weighted mean per brick: the fields are only meaningful inside cloud."""
        return (field.reshape(count, -1) * blocks.reshape(count, -1)).sum(axis=1) / weight

    attributes = quantize(mean(types)).astype(np.uint32) | quantize(mean(scales)).astype(np.uint32) << 8
    z, y, x = np.argwhere(occupied).T.astype(np.uint32)
    packed = x | y << 10 | z << 20
    # Center to center distance to the nearest cloudy brick, less a full
    # diagonal, bounds the distance from anywhere in one brick to the other.
    distance = np.clip(np.floor(edt(occupied) - np.sqrt(3.0)), 0, 255).astype(np.uint8)
    distance[occupied] = 0
    grid = occupied.shape[::-1]
    assert max(grid) < 1024, grid
    meta = json.dumps({"source": credit}).encode()
    meta += b" " * (-len(meta) % 4)
    body = (b"VQCLOUD1" + struct.pack("<5If", *grid, count, CODES, voxel) + struct.pack("<I", len(meta)) + meta
            + codebook.tobytes() + packed.tobytes() + gain.astype(np.float32).tobytes()
            + attributes.tobytes() + codes.tobytes() + distance.tobytes())
    target = OUTPUT / f"{name}.vqcloud"
    target.write_bytes(body)
    sparse = count * BRICK ** 3
    print(f"  {target.name}: {grid} bricks of {voxel:.2f} m voxels, {count} cloudy "
          f"({100 * count / occupied.size:.1f}%), {target.stat().st_size / 1e6:.2f} MB "
          f"(sparse 8-bit: {sparse / 1e6:.1f} MB, dense 8-bit: {occupied.size * BRICK ** 3 / 1e6:.1f} MB), "
          f"RMS error {np.sqrt((error ** 2).mean()):.4f} of peak", flush=True)


def nubis(name):
    path = OUTPUT / f"{name}Cloud.vdb"
    # The fields are sparse and start at different indices: place them in the profile's box.
    profile, lo, voxel = dense(path, "dimensional_profile")
    fields = []
    for channel in NVDF_CHANNELS[1:]:
        field, at, _ = dense(path, channel)
        whole = np.zeros_like(profile)
        offset = at - lo
        src = tuple(slice(max(-o, 0), min(n, s - o)) for o, n, s in zip(offset, field.shape, profile.shape))
        dst = tuple(slice(max(o, 0), max(o, 0) + (b.stop - b.start)) for o, b in zip(offset, src))
        whole[dst] = field[src]
        fields.append(whole)
    vq_cloudscape(name.lower(), *(z_up(f) for f in (profile, *fields)), voxel, NUBIS_CREDIT)


def disney():
    density, _, voxel = dense(OUTPUT / "wdas_cloud_quarter.vdb", "density")
    voxel *= WDAS_SCALE
    inside = density > 0.1
    # Depth below the surface at half resolution, which the smooth ramp allows.
    depth = edt(~np.pad(inside, 1)[::2, ::2, ::2])[1:, 1:, 1:] * 2.0 * voxel
    for axis in range(3):
        depth = np.repeat(depth, 2, axis=axis)
    depth = depth[tuple(slice(0, n) for n in inside.shape)]
    profile = z_up(np.where(inside, np.clip(depth / PROFILE_DEPTH, 0.0, 1.0), 0.0).astype(np.float32))
    ones = np.ones_like(profile)
    vq_cloudscape("wdas", profile, ones, ones, voxel, WDAS_CREDIT)


def main():
    # Self-check: a lone cloudy brick is 0 from itself, and its distance bound never exceeds the true gap.
    point = np.zeros((5, 5, 5), bool)
    point[2, 2, 2] = True
    gaps = np.floor(edt(point) - np.sqrt(3.0))
    assert edt(point)[2, 2, 2] == 0.0 and gaps[2, 2, 4] <= 1.0 and gaps[0, 2, 2] <= 1.0
    OUTPUT.mkdir(parents=True, exist_ok=True)
    noise(ktx_tool())
    nubis("Parkouring")
    nubis("Stormbird")
    disney()


main()
