"""Convert the glTF scenes in assets/ to runtime .glb files with UASTC KTX2 textures.

gltfpack first optimizes each scene's meshes (vertex cache, overdraw and
fetch order; primitives merged per material) with float attributes kept
(-noq) and no meshopt compression, so any glTF loader reads the result.
Each scene then becomes assets/packed/<name>.glb: the JSON and binary buffer
in one file, with every texture pointing at an external assets/packed/textures/*.ktx2
through KHR_texture_basisu (required, no PNG fallback). Textures are encoded
once with KTX-Software's `ktx create` as UASTC + Zstd with full mip chains:
base color as sRGB, normal and metallic-roughness maps as linear data.
Only referenced images are converted; the source files are never modified.

    python scripts/convert_assets.py           # convert, skipping up-to-date textures
    python scripts/convert_assets.py --verify  # check the outputs only
"""

import argparse
import json
import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "assets"
OUTPUT = SOURCE / "packed"
SCENES = {
    "NewSponza_Main.glb": "NewSponza_Main_glTF_003.gltf",
    "NewSponza_Curtains.glb": "NewSponza_Curtains_glTF.gltf",
    "NewSponza_IvyGrowth.glb": "NewSponza_IvyGrowth_glTF.gltf",
}
GLB_MAGIC, GLB_JSON, GLB_BIN = 0x46546C67, 0x4E4F534A, 0x004E4942
# Settings for each material role: (VkFormat, transfer function, extra flags).
ROLES = {
    "color": ("R8G8B8A8_SRGB", "srgb", []),
    "data": ("R8G8B8_UNORM", "linear", []),
    "normal": ("R8G8B8_UNORM", "linear", ["--normalize"]),
}


def ktx_tool():
    found = shutil.which("ktx") or r"C:\Program Files\KTX-Software\bin\ktx.exe"
    if not Path(found).exists():
        sys.exit("ktx (KTX-Software 4.3+) not found on PATH")
    return found


def optimize(source_name, name):
    """Run gltfpack into an intermediate .gltf + .bin, keeping texture paths (-tr)."""
    gltfpack = shutil.which("gltfpack") or sys.exit("gltfpack not found on PATH (npm install -g gltfpack)")
    target = OUTPUT / ".mesh" / Path(name).with_suffix(".gltf")
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([gltfpack, "-i", str(SOURCE / source_name), "-o", str(target), "-noq", "-tr"], check=True)
    return target


def texture_roles(gltf):
    """Map each image index to its role, from how materials use the textures."""
    roles = {}

    def mark(info, role):
        if info is None:
            return
        image = gltf["textures"][info["index"]]["source"]
        if roles.setdefault(image, role) != role:
            sys.exit(f"image {gltf['images'][image]['uri']} is used as both {roles[image]} and {role}")

    for material in gltf.get("materials", []):
        pbr = material.get("pbrMetallicRoughness", {})
        mark(pbr.get("baseColorTexture"), "color")
        mark(material.get("emissiveTexture"), "color")
        mark(pbr.get("metallicRoughnessTexture"), "data")
        mark(material.get("occlusionTexture"), "data")
        mark(material.get("normalTexture"), "normal")
    return roles


def encode(ktx, source, target, role):
    if target.exists() and target.stat().st_mtime >= source.stat().st_mtime:
        return False
    vk_format, transfer, extra = ROLES[role]
    partial = target.with_suffix(".ktx2.partial")
    subprocess.run(
        [ktx, "create", "--format", vk_format, "--assign-tf", transfer, *extra,
         "--encode", "uastc", "--uastc-quality", "2", "--zstd", "18",
         "--generate-mipmap", str(source), str(partial)],
        check=True,
    )
    # Rename last so an interrupted run never leaves a truncated texture behind.
    os.replace(partial, target)
    return True


def pack(gltf, binary):
    """GLB bytes: a JSON chunk padded with spaces, then a zero-padded BIN chunk."""
    text = json.dumps(gltf, separators=(",", ":")).encode()
    text += b" " * (-len(text) % 4)
    binary += b"\0" * (-len(binary) % 4)
    length = 12 + 8 + len(text) + 8 + len(binary)
    return b"".join([
        struct.pack("<III", GLB_MAGIC, 2, length),
        struct.pack("<II", len(text), GLB_JSON), text,
        struct.pack("<II", len(binary), GLB_BIN), binary,
    ])


def convert_scene(ktx, name, source_name):
    optimized = optimize(source_name, name)
    gltf = json.loads(optimized.read_text(encoding="utf-8"))
    if len(gltf["buffers"]) != 1:
        sys.exit(f"{source_name}: expected one buffer")
    buffer = gltf["buffers"][0]
    binary_path = optimized.parent / unquote(buffer.pop("uri"))
    binary = binary_path.read_bytes()
    if len(binary) != buffer["byteLength"]:
        sys.exit(f"{source_name}: buffer is {len(binary)} bytes, glTF says {buffer['byteLength']}")

    roles = texture_roles(gltf)
    images = []
    remap = {}
    for index, image in enumerate(gltf.get("images", [])):
        if index not in roles:
            continue
        # -tr keeps URIs as written in the source scene, so they resolve from SOURCE.
        png = SOURCE / unquote(image["uri"])
        target = OUTPUT / "textures" / (png.stem + ".ktx2")
        print(f"  {'encode' if encode(ktx, png, target, roles[index]) else 'keep  '} {target.name} ({roles[index]})", flush=True)
        remap[index] = len(images)
        images.append({"uri": f"textures/{target.name}", "mimeType": "image/ktx2"})
    gltf["images"] = images
    textures = []
    texture_remap = {}
    for index, texture in enumerate(gltf.get("textures", [])):
        source = texture.pop("source")
        if source in remap:
            texture_remap[index] = len(textures)
            texture.setdefault("extensions", {})["KHR_texture_basisu"] = {"source": remap[source]}
            textures.append(texture)
    gltf["textures"] = textures

    def retarget(info):
        if info is not None:
            info["index"] = texture_remap[info["index"]]

    for material in gltf.get("materials", []):
        pbr = material.get("pbrMetallicRoughness", {})
        for info in (pbr.get("baseColorTexture"), pbr.get("metallicRoughnessTexture"),
                     material.get("normalTexture"), material.get("occlusionTexture"),
                     material.get("emissiveTexture")):
            retarget(info)
    for key in ("extensionsUsed", "extensionsRequired"):
        gltf[key] = sorted(set(gltf.get(key, [])) | {"KHR_texture_basisu"})

    target = OUTPUT / name
    partial = target.with_suffix(".glb.partial")
    partial.write_bytes(pack(gltf, binary))
    os.replace(partial, target)
    optimized.unlink()
    binary_path.unlink()


def verify(name):
    """Check the GLB framing, buffer bounds and that every texture file exists."""
    data = (OUTPUT / name).read_bytes()
    magic, version, length = struct.unpack_from("<III", data, 0)
    assert (magic, version, length) == (GLB_MAGIC, 2, len(data)), "bad GLB header"
    json_length, json_type = struct.unpack_from("<II", data, 12)
    assert json_type == GLB_JSON and json_length % 4 == 0
    gltf = json.loads(data[20:20 + json_length])
    offset = 20 + json_length
    bin_length, bin_type = struct.unpack_from("<II", data, offset)
    assert bin_type == GLB_BIN and offset + 8 + bin_length == len(data)
    buffer = gltf["buffers"][0]
    assert "uri" not in buffer and buffer["byteLength"] <= bin_length
    for view in gltf["bufferViews"]:
        assert view["byteOffset"] + view["byteLength"] <= buffer["byteLength"] if "byteOffset" in view else view["byteLength"] <= buffer["byteLength"]
    for image in gltf["images"]:
        ktx2 = OUTPUT / image["uri"]
        assert ktx2.read_bytes()[:12] == b"\xabKTX 20\xbb\r\n\x1a\n", f"{ktx2} is not KTX2"
    for texture in gltf["textures"]:
        assert "source" not in texture and texture["extensions"]["KHR_texture_basisu"]["source"] < len(gltf["images"])
    assert "KHR_texture_basisu" in gltf["extensionsRequired"]
    print(f"  ok {name}: {len(gltf['meshes'])} meshes, {len(gltf['images'])} KTX2 textures, {len(data) / 1e6:.0f} MB")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--verify", action="store_true", help="only check existing outputs")
    args = parser.parse_args()
    if not args.verify:
        ktx = ktx_tool()
        (OUTPUT / "textures").mkdir(parents=True, exist_ok=True)
        for name, source in SCENES.items():
            print(f"{source} -> packed/{name}", flush=True)
            convert_scene(ktx, name, source)
    for name in SCENES:
        verify(name)


if __name__ == "__main__":
    main()
