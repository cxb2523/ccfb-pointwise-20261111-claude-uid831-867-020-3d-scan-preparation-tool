"""Deep verification of batch outputs (independent of the service code)."""
import io
import json
import os
import struct
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from PIL import Image

from scanprep3d import obj_io, glb_io

OUT = sys.argv[1]
obj_path = os.path.join(OUT, "materials_grid.obj")
mtl_path = os.path.join(OUT, "materials.mtl")
glb_path = os.path.join(OUT, "spheres.glb")

failures = []


def check(cond, label):
    print(("PASS" if cond else "FAIL"), label)
    if not cond:
        failures.append(label)


# --- OBJ ------------------------------------------------------------------- #
mesh = obj_io.parse_obj(obj_path)
group_faces = {g.material: len(g.faces) for g in mesh.groups}
print("OBJ groups:", group_faces)
check(set(group_faces) == {"mat_png", "mat_tga", "mat_jpg"}, "three material groups kept")
check(sum(group_faces.values()) == 864, "OBJ total faces 864")
# No cross-material welding: groups keep disjoint vertex blocks.
group_vertex_ranges = []
for g in mesh.groups:
    used = {vi for vi, _t, _n in g.instances}
    group_vertex_ranges.append((min(used), max(used)))
print("group vertex ranges:", group_vertex_ranges)
check(group_vertex_ranges[0][1] < group_vertex_ranges[1][0]
      and group_vertex_ranges[1][1] < group_vertex_ranges[2][0],
      "material groups never share vertices (no cross-material weld)")

mtl = obj_io.parse_mtl(mtl_path)
refs = {}
for name, entries in mtl.materials.items():
    for key, _opts, token, _raw in entries:
        if token is not None:
            refs[name] = token
print("MTL refs:", refs)
check(refs.get("mat_png", "").replace("\\", "/") == "textures/tex_a.webp", "PNG rewritten to WebP in MTL")
check(refs.get("mat_tga", "").replace("\\", "/") == "textures/tex_b.webp", "TGA rewritten to WebP in MTL")
check(refs.get("mat_jpg", "").replace("\\", "/") == "textures/tex_c.jpg", "JPEG kept as passthrough in MTL")

for rel in ("textures/tex_a.webp", "textures/tex_b.webp", "textures/tex_c.jpg"):
    p = os.path.join(OUT, rel)
    check(os.path.isfile(p), f"output exists: {rel}")
    with Image.open(p) as im:
        im.load()
        check(im.size == (32, 32), f"{rel} decodes at 32x32 (mode {im.mode})")
check(open(os.path.join(OUT, "textures", "tex_a.webp"), "rb").read(4) == b"RIFF",
      "tex_a.webp has RIFF/WebP header")

# --- GLB ------------------------------------------------------------------- #
container = glb_io.read_glb(glb_path)
header = container.header
glb_faces = 0
for mesh_ in header["meshes"]:
    for prim in mesh_["primitives"]:
        glb_faces += header["accessors"][prim["indices"]]["count"] // 3
check(glb_faces == 1280, "GLB total faces 1280")
check(len(header.get("materials", [])) == 2, "GLB keeps two materials")
check("EXT_texture_webp" in header.get("extensionsRequired", []),
      "GLB requires EXT_texture_webp")
for tex in header.get("textures", []):
    check("EXT_texture_webp" in tex.get("extensions", {}),
          "texture entry carries EXT_texture_webp source")
for image in header.get("images", []):
    check(image.get("mimeType") == "image/webp", "GLB image mime is image/webp")
    blob = container.view_bytes(image["bufferView"])
    check(blob[:4] == b"RIFF", "GLB embedded image bytes are WebP")
    Image.open(io.BytesIO(blob)).load()

# Source files must be untouched and present.
src = os.path.join(os.path.dirname(OUT), "multi_material")
check(os.path.isfile(os.path.join(src, "materials_grid.obj")), "source OBJ untouched")
check(os.path.isfile(os.path.join(src, "tex_a.png")), "source PNG untouched")
check(not any(name.startswith(".staging") for name in os.listdir(OUT)),
      "no staging directories left behind")

print()
print("FAILURES:", failures if failures else "none")
sys.exit(1 if failures else 0)
