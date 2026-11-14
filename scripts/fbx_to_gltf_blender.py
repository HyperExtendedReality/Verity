"""Blender (headless) helper: import one FBX and export it as a separate glTF.

    blender -b --python scripts/fbx_to_gltf_blender.py -- <in.fbx> <out.gltf>

Textures are not re-encoded here: the exporter keeps the original files and
writes their paths, which convert_fbx.py then turns into KTX2.
"""
import sys

import bpy

source, target = sys.argv[sys.argv.index("--") + 1:][:2]
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.wm.fbx_import(filepath=source)
print("objects", len(bpy.data.objects), "materials", len(bpy.data.materials), "images", len(bpy.data.images))
bpy.ops.export_scene.gltf(
    filepath=target,
    export_format="GLTF_SEPARATE",
    export_keep_originals=True,
    export_image_format="AUTO",
    export_apply=True,
    export_cameras=False,
    export_lights=False,
    export_yup=True,
)
