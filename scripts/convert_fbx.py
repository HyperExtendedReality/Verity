"""Convert the FBX models in assets/ to runtime .glb files, KTX2 textures and editor scenes.

For each model:
  1. Blender (headless, scripts/fbx_to_gltf_blender.py) exports a separate glTF.
  2. The materials are rebuilt from the texture names (`*_BaseColor`, `*_Normal`,
     `*_Specular` = occlusion/roughness/metallic, `*_Emissive`): Blender reads the
     Specular map as a specular color, which is wrong for these assets. Alpha
     becomes MASK only when the BaseColor texture really has transparent texels.
  3. The DDS (or TGA) textures are decoded with ImageMagick (normals get their
     Z rebuilt from the BC5 X/Y), then convert_assets.py optimizes the meshes with
     gltfpack and encodes UASTC + Zstd KTX2 into assets/packed/textures/<set>/.
  4. assets/scenes/<model>.verity references the .glb with a camera framing it.

    python scripts/convert_fbx.py                 # all models
    python scripts/convert_fbx.py BistroInterior  # one model
"""

import json
import re
import shutil
import struct
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import unquote

import convert_assets as ca

ROOT = ca.ROOT
WORK = ROOT / "tmp" / "fbx"
SCENES = ROOT / "assets" / "scenes"
BLENDER = shutil.which("blender") or r"C:\Program Files\Blender Foundation\Blender 5.3\blender.exe"
# glb name -> (FBX, texture set, root scale). Blender leaves the Bistro nodes 1e-4 too small
# (so 100 gives meters); the MEASURE files are used as Blender imports them (unit unverified).
MODELS = {
    "BistroExterior": ("Bistro_v5_2/BistroExterior.fbx", "Bistro", 100.0),
    "BistroInterior": ("Bistro_v5_2/BistroInterior.fbx", "Bistro", 100.0),
    "BistroInterior_Wine": ("Bistro_v5_2/BistroInterior_Wine.fbx", "Bistro", 100.0),
    "MEASURE_ONE": ("MEASURE_ONE/MEASURE_ONE.fbx", "MEASURE_ONE", 1.0),
    "MEASURE_SEVEN": ("MEASURE_SEVEN/MEASURE_SEVEN.fbx", "MEASURE_SEVEN", 1.0),
    "MEASURE_SEVEN_COLORED_LIGHTS": ("MEASURE_SEVEN/MEASURE_SEVEN_COLORED_LIGHTS.fbx", "MEASURE_SEVEN", 1.0),
}
SUFFIX = re.compile(r"_(BaseColor|Normal|Specular|Emissive\d*)$", re.I)
JOBS = 8


def export(name, fbx):
    out = WORK / f"{name}.gltf"
    if not out.exists():
        subprocess.run([BLENDER, "-b", "--python", str(ROOT / "scripts" / "fbx_to_gltf_blender.py"),
                        "--", str(ROOT / "assets" / fbx), str(out)], check=True, capture_output=True)
    return json.loads(out.read_text(encoding="utf-8"))


def decode(source, target, kind):
    """DDS/TGA -> PNG (mip 0). The PNG keeps alpha only for a BaseColor with transparent texels."""
    if target.exists() and target.stat().st_mtime >= source.stat().st_mtime:
        return
    args = []
    if kind == "Normal":
        # BC5 stores X and Y only; rebuild Z so the glTF normal map is a full RGB one.
        args = ["-channel", "B", "-fx", "sqrt(max(0,1-(2*r-1)*(2*r-1)-(2*g-1)*(2*g-1)))/2+0.5", "+channel"]
    opaque = kind != "BaseColor"
    if not opaque:
        low = subprocess.run(["magick", f"{source}[0]", "-format", "%[fx:minima.a]", "info:"],
                             check=True, capture_output=True, text=True).stdout
        opaque = float(low) >= 0.98
    if opaque:
        args = ["-alpha", "off", *args]
    subprocess.run(["magick", f"{source}[0]", *args, str(target)], check=True)


def has_alpha(png):
    return png.read_bytes()[:26][25] in (4, 6)  # PNG color type


def convert(name, fbx, texture_set, scale, ktx):
    gltf = export(name, fbx)
    work = WORK / texture_set
    (work / "png").mkdir(parents=True, exist_ok=True)

    # Rebuild materials from the texture names.
    images = gltf["images"]
    kinds = {}
    for index, image in enumerate(images):
        match = SUFFIX.search(Path(unquote(image["uri"])).stem)
        kinds[index] = {"basecolor": "BaseColor", "normal": "Normal", "specular": "Specular"}.get(match.group(1).lower()) if match else None
    source_of = lambda tex: gltf["textures"][tex["index"]]["source"]
    for material in gltf["materials"]:
        pbr = material.setdefault("pbrMetallicRoughness", {})
        spec = material.get("extensions", {}).pop("KHR_materials_specular", None)
        if not material.get("extensions", True):
            del material["extensions"]
        if spec and "specularColorTexture" in spec:
            orm = spec["specularColorTexture"]
            pbr["metallicRoughnessTexture"] = dict(orm)
            material["occlusionTexture"] = dict(orm)
            pbr["metallicFactor"] = pbr["roughnessFactor"] = 1.0
        base = pbr.get("baseColorTexture")
        material["alphaMode"] = "OPAQUE"
        material["doubleSided"] = False
        material["_base"] = source_of(base) if base else None
    gltf["extensionsUsed"] = [e for e in gltf.get("extensionsUsed", []) if e != "KHR_materials_specular"]
    if not gltf["extensionsUsed"]:
        del gltf["extensionsUsed"]

    roles = ca.texture_roles(gltf)
    jobs = []
    for index in roles:
        uri = unquote(images[index]["uri"])
        source = (WORK / uri).resolve()
        png = work / "png" / (source.stem + ".png")
        jobs.append((index, source, png, kinds[index] or "Other"))
    with ThreadPoolExecutor(JOBS) as pool:
        list(pool.map(lambda j: decode(j[1], j[2], j[3]), jobs))
    transparent = {index: has_alpha(png) for index, _, png, _ in jobs}
    for material in gltf["materials"]:
        image = material.pop("_base")
        if image is not None and transparent.get(image):
            material.update(alphaMode="MASK", alphaCutoff=0.5, doubleSided=True)
    # Point at the decoded PNGs; convert_scene reads these paths and swaps in the KTX2 files.
    for index, source, png, _ in jobs:
        images[index]["uri"] = f"png/{png.name}"
        images[index]["mimeType"] = "image/png"
    gltf["buffers"][0]["uri"] = f"../{name}.bin"
    (work / f"{name}.gltf").write_text(json.dumps(gltf), encoding="utf-8")

    with ThreadPoolExecutor(JOBS) as pool:
        list(pool.map(lambda j: ca.encode(ktx, j[2], ca.OUTPUT / "textures" / texture_set / (j[2].stem + ".ktx2"),
                                          roles[j[0]]), jobs))
    ca.convert_scene(ktx, f"{name}.glb", f"{name}.gltf", source=work, textures=f"textures/{texture_set}", scale=scale)


def matmul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4)] for i in range(4)]


def local_matrix(node):
    if "matrix" in node:
        m = node["matrix"]  # column-major
        return [[m[c * 4 + r] for c in range(4)] for r in range(4)]
    (x, y, z, w), (sx, sy, sz), t = node.get("rotation", [0, 0, 0, 1]), node.get("scale", [1, 1, 1]), node.get("translation", [0, 0, 0])
    r = [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
         [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
         [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]
    return [[r[i][0] * sx, r[i][1] * sy, r[i][2] * sz, t[i]] for i in range(3)] + [[0, 0, 0, 1]]


def bounds(name):
    """Z-up world bounds (min, max) of the packed glb, from its nodes' meshes and transforms."""
    data = (ca.OUTPUT / f"{name}.glb").read_bytes()
    length = struct.unpack_from("<I", data, 12)[0]
    gltf = json.loads(data[20:20 + length])
    boxes = []
    # The renderer turns Y-up into Z-up: (x, y, z) -> (x, -z, y).
    stack = [(n, [[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]) for n in gltf["scenes"][0]["nodes"]]
    while stack:
        index, parent = stack.pop()
        node = gltf["nodes"][index]
        world = matmul(parent, local_matrix(node))
        stack += [(c, world) for c in node.get("children", [])]
        if "mesh" not in node:
            continue
        for primitive in gltf["meshes"][node["mesh"]]["primitives"]:
            a = gltf["accessors"][primitive["attributes"]["POSITION"]]
            corners = [[a["max"][i] if c >> i & 1 else a["min"][i] for i in range(3)] + [1] for c in range(8)]
            points = [[sum(world[axis][k] * p[k] for k in range(4)) for axis in range(3)] for p in corners]
            boxes.append(([min(q[i] for q in points) for i in range(3)], [max(q[i] for q in points) for i in range(3)]))
    # shortcut: 1st/99th percentile, so a few stray or degenerate meshes do not push the camera away.
    cut = lambda values, at: sorted(values)[min(len(values) - 1, int(len(values) * at))]
    return ([cut([b[0][i] for b in boxes], 0.01) for i in range(3)],
            [cut([b[1][i] for b in boxes], 0.99) for i in range(3)])


def write_scene(name):
    lo, hi = bounds(name)
    center = [(a + b) / 2 for a, b in zip(lo, hi)]
    size = [b - a for a, b in zip(lo, hi)]
    scene = json.loads((SCENES / "sponza.verity").read_text())
    scene["assets"] = [f"assets/packed/{name}.glb"]
    scene["cameras"] = [{
        "name": "Overview",
        "position": [center[0], lo[1] - 0.6 * max(size[0], size[1]), center[2] + 0.2 * size[2]],
        "target": center,
        "vertical_fov": 60.0,
    }]
    (SCENES / f"{name.lower().replace('_', '-')}.verity").write_text(json.dumps(scene, indent=2) + "\n")
    print(f"  scene {name.lower().replace('_', '-')}.verity, bounds {[round(v, 1) for v in lo]} .. {[round(v, 1) for v in hi]}")


def main():
    names = sys.argv[1:] or list(MODELS)
    ktx = ca.ktx_tool()
    for name in names:
        fbx, texture_set, scale = MODELS[name]
        print(f"{fbx} -> packed/{name}.glb", flush=True)
        convert(name, fbx, texture_set, scale, ktx)
        ca.verify(f"{name}.glb")
        write_scene(name)


if __name__ == "__main__":
    main()
