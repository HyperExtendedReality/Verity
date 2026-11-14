"""Convert the terrain layer PNGs in assets/textures/ to virtual-texture KTX2 files.

Each layer named `<Layer>_4K-PNG_<Map>.png` becomes two files in
assets/packed/terrain/:

    <Layer>_albedo.ktx2   RGB base color (sRGB) with height in alpha (linear)
    <Layer>_surface.ktx2  RG normal (OpenGL, +Y up), B roughness, A ambient occlusion

ffmpeg packs the channels and KTX-Software's `ktx create` encodes UASTC with a
full mip chain. Unlike the scene textures these are not Zstd supercompressed:
the renderer maps the files and reads one 128x128 page of blocks at a time,
which needs each level stored as plain blocks. A layer without an ambient
occlusion map gets a white one.

    python scripts/convert_terrain_layers.py           # convert, skipping up-to-date layers
    python scripts/convert_terrain_layers.py --jobs 2  # layers encoded at once
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "assets" / "textures"
OUTPUT = ROOT / "assets" / "packed" / "terrain"
NAME = re.compile(r"^(?P<layer>.+?)_\dK-PNG_(?P<map>[A-Za-z]+)\.png$")


def tool(name, fallback):
    found = shutil.which(name) or fallback
    if not found or not Path(found).exists():
        sys.exit(f"{name} not found on PATH")
    return found


def layers():
    """Each layer's maps, by map name."""
    found = {}
    for path in sorted(SOURCE.glob("*.png")):
        match = NAME.match(path.name)
        if match:
            found.setdefault(match["layer"], {})[match["map"]] = path
    return found


def encode(ktx, source, target, vk_format, transfer):
    partial = target.with_suffix(".ktx2.partial")
    subprocess.run(
        [ktx, "create", "--format", vk_format, "--assign-tf", transfer,
         "--encode", "uastc", "--uastc-quality", "2",
         "--generate-mipmap", str(source), str(partial)],
        check=True,
    )
    # Rename last so an interrupted run never leaves a truncated texture behind.
    os.replace(partial, target)


def size_of(ffprobe, path):
    out = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "v", "-show_entries", "stream=width,height",
         "-of", "csv=p=0", str(path)],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    width, height = map(int, out.split(","))
    if width != height or width & (width - 1):
        sys.exit(f"{path.name}: expected a square power-of-two image, got {width}x{height}")
    return width


def convert(tools, layer, maps):
    ffmpeg, ffprobe, ktx = tools
    missing = {"Color", "Displacement", "NormalGL", "Roughness"} - maps.keys()
    if missing:
        print(f"skip {layer}: no {', '.join(sorted(missing))} map")
        return
    sources = [maps[m] for m in ("Color", "Displacement", "NormalGL", "Roughness", "AmbientOcclusion") if m in maps]
    newest = max(p.stat().st_mtime for p in sources)
    albedo = OUTPUT / f"{layer}_albedo.ktx2"
    surface = OUTPUT / f"{layer}_surface.ktx2"
    if all(t.exists() and t.stat().st_mtime >= newest for t in (albedo, surface)):
        return
    size = size_of(ffprobe, maps["Color"])
    work = OUTPUT / ".work"
    work.mkdir(parents=True, exist_ok=True)
    flat = "setsar=1,"
    # mergeplanes takes planes in gbrap order: green, blue, red, alpha.
    jobs = [
        (albedo, "R8G8B8A8_SRGB", "srgb", [maps["Color"], maps["Displacement"]],
         f"[0:v]{flat}format=gbrp,extractplanes=r+g+b[r][g][b];[1:v]{flat}format=gray[a];"
         "[g][b][r][a]mergeplanes=0x00102030:gbrap,format=rgba"),
        (surface, "R8G8B8A8_UNORM", "linear",
         [maps["NormalGL"], maps["Roughness"], maps.get("AmbientOcclusion")],
         f"[0:v]{flat}format=gbrp,extractplanes=r+g[r][g];[1:v]{flat}format=gray[b];"
         f"[2:v]{flat}format=gray[a];[g][b][r][a]mergeplanes=0x00102030:gbrap,format=rgba"),
    ]
    for target, vk_format, transfer, inputs, graph in jobs:
        packed = work / f"{target.stem}.png"
        args = [ffmpeg, "-v", "error", "-y"]
        for source in inputs:
            if source is None:
                args += ["-f", "lavfi", "-i", f"color=white:s={size}x{size}"]
            else:
                args += ["-i", str(source)]
        subprocess.run([*args, "-filter_complex", graph, "-frames:v", "1", str(packed)], check=True)
        encode(ktx, packed, target, vk_format, transfer)
        packed.unlink()
    print(f"converted {layer}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--jobs", type=int, default=2, help="layers encoded at once")
    args = parser.parse_args()
    tools = (
        tool("ffmpeg", None),
        tool("ffprobe", None),
        tool("ktx", r"C:\Program Files\KTX-Software\bin\ktx.exe"),
    )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    found = layers()
    if not found:
        sys.exit(f"no layer PNGs in {SOURCE}")
    with ThreadPoolExecutor(max_workers=max(args.jobs, 1)) as pool:
        for future in [pool.submit(convert, tools, layer, maps) for layer, maps in found.items()]:
            future.result()
    shutil.rmtree(OUTPUT / ".work", ignore_errors=True)


if __name__ == "__main__":
    main()
