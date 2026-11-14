"""Convert the cloud datasets in assets/ to 3D KTX2 volumes under assets/packed/clouds.

Run with Blender 5.3+, whose Python bundles OpenVDB and numpy:

    blender -b --factory-startup --python scripts/convert_clouds.py

Sources are the float VDBs, not the 8-bit TGA slices: the Nubis signed distance
field in particular needs the precision. Every output is R8G8B8A8_UNORM, 3D,
Zstd-supercompressed, z up (Verity's), with a "verity.cloud" JSON key:

- nubis_noise.ktx2: Guerrilla's 128^3 tiling detail noise with a full mip chain.
  R, G: low and high frequency curly Worley (wisps); B, A: low and high
  frequency alligator (billows).
- <cloud>.ktx2: a voxel cloudscape (Nubis NVDF layout). R: dimensional profile;
  G: detail type (0 wispy, 1 billowy); B: density scale; A: signed distance to
  the cloud in metres, mapped from the "sdf" range. Cropped to where the profile
  is nonzero plus a margin: the sampler skips zero profile, and outside the box
  the distance to the box is a safe lower bound for sphere tracing.

Why uncompressed 8-bit: KTX tools cannot encode 3D textures to BC formats, the
noise is sampled at full detail next to the camera where block artifacts show,
and cropping already shrinks each cloudscape from 64 MB to a few MB of VRAM.
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
ASSETS = ROOT / "assets"
OUTPUT = ASSETS / "packed" / "clouds"
NUBIS = ASSETS / "Nubis Voxel Clouds Pack"
NOISE_CHANNELS = ("lf_curly_worley", "hf_curly_worley", "lf_alligator", "hf_alligator")
NVDF_CHANNELS = ("dimensional_profile", "detail_type", "density_scale")
NUBIS_CREDIT = "Nubis Voxel Clouds Pack, Guerrilla Games (Andrew Schneider), 2023"
WDAS_CREDIT = ("Walt Disney Animation Studios Cloud Data Set, Copyright 2017 Disney "
               "Enterprises, Inc., CC BY-SA 3.0")
# Voxels kept around the cloud so the stored distance covers its surroundings.
MARGIN = 4
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
        for start in range(0, len(lines), 2048):
            chunk = lines[start:start + 2048]
            result[start:start + 2048] = (chunk[:, None, :] + gap[None]).min(axis=2)
        squared = np.moveaxis(result.reshape(moved.shape), -1, axis)
    return np.sqrt(squared)


def signed_distance(inside, voxel):
    """Metres to the surface of `inside`, negative within it; the surface sits
    half a voxel past the last inside voxel."""
    outside = edt(inside) - 0.5
    within = edt(~inside) - 0.5
    return np.where(inside, -within, outside) * voxel


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
    path = NUBIS / "Noise" / "Examples" / "VDB" / "NubisVoxelCloudNoise.vdb"
    level = np.stack([z_up(dense(path, name)[0]) for name in NOISE_CHANNELS], axis=-1)
    assert level.shape == (128, 128, 128, 4), level.shape
    levels = [level]
    while levels[-1].shape[0] > 1:
        levels.append(downsample(levels[-1]))
    metadata = {"kind": "noise", "channels": list(NOISE_CHANNELS), "source": NUBIS_CREDIT}
    publish(ktx, [quantize(level) for level in levels], metadata, "nubis_noise")


def cloudscape(ktx, name, profile, lo, voxel, fields, sdf, credit):
    """Crop to the profile plus `MARGIN` and publish. `profile`, every field and
    `sdf` (metres) are [x, y, z] VDB-index arrays starting at index `lo`."""
    occupied = np.argwhere(profile > 0.0)
    start = np.maximum(occupied.min(axis=0) - MARGIN, 0)
    stop = np.minimum(occupied.max(axis=0) + MARGIN + 1, profile.shape)
    crop = tuple(slice(a, b) for a, b in zip(start, stop))
    sdf = sdf[crop]
    # Rounded down, so the stored distance never oversteps a sphere trace.
    sdf_range = [float(np.floor(sdf.min())), float(np.ceil(sdf.max()))]
    encoded_sdf = np.floor((sdf - sdf_range[0]) / (sdf_range[1] - sdf_range[0]) * 255.0)
    channels = [quantize(z_up(field[crop])) for field in (profile, *fields)]
    channels.append(z_up(np.clip(encoded_sdf, 0, 255).astype(np.uint8)))
    level = np.stack(channels, axis=-1)
    # The box's low corner in metres: VDB index i is a voxel centred on i.
    first, last = lo + start, lo + stop - 1
    metadata = {
        "kind": "cloudscape",
        "voxel": voxel,
        "min": [(first[0] - 0.5) * voxel, -(last[2] + 0.5) * voxel, (first[1] - 0.5) * voxel],
        "sdf": sdf_range,
        "source": credit,
    }
    publish(ktx, [level], metadata, name)


def nubis(ktx, name, file):
    path = NUBIS / "NVDFs" / "Examples" / name / "VDB" / file
    # The dense fields share the full grid; the profile is sparse, so place it there.
    sdf, lo, voxel = dense(path, "sdf")
    profile = np.zeros_like(sdf)
    sparse, at, _ = dense(path, "dimensional_profile")
    offset = at - lo
    profile[tuple(slice(o, o + n) for o, n in zip(offset, sparse.shape))] = sparse
    fields = []
    for channel in NVDF_CHANNELS[1:]:
        field, at, _ = dense(path, channel)
        whole = np.zeros_like(sdf)
        offset = at - lo
        whole[tuple(slice(o, o + n) for o, n in zip(offset, field.shape))] = field
        fields.append(whole)
    cloudscape(ktx, name.removesuffix("Cloud").lower(), profile, lo, voxel, fields, sdf, NUBIS_CREDIT)


def disney(ktx):
    # The sixteenth level (3.3 m voxels) is a Nubis-style low-resolution
    # profile; the up-rez adds the fine detail.
    density, lo, voxel = dense(ASSETS / "wdas_cloud" / "wdas_cloud_sixteenth.vdb", "density")
    pad = MARGIN + 1
    density = np.pad(density, pad)
    lo = lo - pad
    sdf = signed_distance(density > 0.0, voxel)
    ones = np.ones_like(density)
    cloudscape(ktx, "wdas", density, lo, voxel, (ones, ones), sdf, WDAS_CREDIT)


def main():
    ktx = ktx_tool()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    # Self-check: a single inside voxel is half a voxel inside, its neighbour half outside.
    point = np.zeros((5, 5, 5), bool)
    point[2, 2, 2] = True
    check = signed_distance(point, 2.0)
    assert check[2, 2, 2] == -1.0 and check[2, 2, 3] == 1.0 and abs(check[0, 2, 2] - 3.0) < 1e-6
    noise(ktx)
    nubis(ktx, "ParkouringCloud", "ParkouringCloud.vdb")
    nubis(ktx, "StormbirdCloud", "StormbirdCloud.vdb")
    disney(ktx)


main()
