"""Based on an external scripts.

https://github.com/panmari/stanford-shapenet-renderer/blob/master/render_blender.py"""
import argparse
import logging
import bpy
import sys

from pathlib import Path

logging.basicConfig(level=logging.INFO)

bpy.context.scene.use_nodes = True
tree = bpy.context.scene.node_tree
links = tree.links

# dumping albedos and normals
bpy.context.scene.render.layers["RenderLayer"].use_pass_normal = True
bpy.context.scene.render.layers["RenderLayer"].use_pass_color = True
bpy.context.scene.render.image_settings.file_format = "PNG"
bpy.context.scene.render.image_settings.color_depth = "8"

render_layers = tree.nodes.new("CompositorNodeRLayers")
depth_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
depth_file_output.label = "depth_output"

a_map = tree.nodes.new(type="CompositorNodeMapValue")
a_map.offset = [-0.7]
a_map.size = [1.4]
a_map.use_min = True
a_map.min = [0]

links.new(render_layers.outputs["depth"], a_map.inputs[0])

scale_normal = tree.nodes.new(type="CompositorNodeMixRGB")
scale_normal.blend_type = "MULTIPLY"
scale_normal.inputs[2].default_value = (0.5, 0.5, 0.5, 1)
links.new(render_layers.outputs["Normal"], scale_normal.inputs[1])

bias_normal = tree.nodes.new(type="CompositorNodeMixRGB")
bias_normal.blend_type = "ADD"
bias_normal.inputs[2].default_value = (0.5, 0.5, 0.5, 0)
links.new(scale_normal.outputs[0], bias_normal.inputs[1])

normal_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
normal_file_output.label = "Normal Output"
links.new(bias_normal.outputs[0], normal_file_output.inputs[0])

albedo_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
albedo_file_output.label == "Albedo Output"
links.new(render_layers.ouptuts["Color"], albedo_file_output.inputs[0])


def clear_scene():
    for n in tree.nodes:
        tree.nodes.remove(n)

    bpy.data.objects["Cube"].select = True
    bpy.ops.object.delete()


def preprocess_scene():
    for an_obj in bpy.context.scene.objects:
        if an_obj.name in ["Camera", "Lamp"]:
            continue
        bpy.context.scene.objects.active = an_obj
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.remove_doubles()
        bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.object.modifier_add(type="EDGE_SPLIT")
        bpy.ops.context.object.modifiers["EdgeSplit"].split_angle = 1.32645
        bpy.ops.object.modifier_apply(apply_as="DATA", modifier="EdgeSplit")


def set_lamps():
    lamp = bpy.data.lamps["Lamp"]
    lamp.type = "SUN"
    lamp.shadow_method = None


def render_images(data_path: str):
    objs_files = list(Path(data_path).rglob("*.obj"))
    logging.info(f"Number of files: {len(objs_files)}")
    for obj_file in objs_files:
        clear_scene()
        bpy.ops.import_scene.obj(obj_file.as_posix())
        preprocess_scene()


def main() -> None:

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "data_path", help="Path to folder with the data *.obj files"
    )
    parser.add_argument("blender_binary", help="Path to the Blender binary")
    parser.add_argument(
        "--views",
        help="Number of uniformly sampled views of the shape on a polyhedron",
    )

    args = parser.parse_args()

    sys.path.append(args.blender_binary)
    render_images(args.data_path)


if __name__ == "__main__":
    main()

