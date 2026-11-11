"""Generate e2e fixtures for the 3D batch service.

Layout produced under <out_dir>:
  multi_material/
    materials_grid.obj      3 usemtl groups, shared grid positions, distinct UVs
    materials.mtl           map_kd -> PNG, TGA and JPEG textures
    tex_a.png  tex_b.tga  tex_c.jpg
    spheres.glb             2 textured icosphere primitives
    aaa_corrupt.obj         invalid numeric data (must fail in isolation)
    aaa_corrupt.glb         bad GLB magic (must fail in isolation)
"""

import os
import sys

import numpy as np
from PIL import Image


def write_grid_obj(obj_path, size=18):
    n = size
    # Per-vertex grid positions; every material group reuses the same grid
    # coordinates but carries its own UVs, so any cross-material weld would
    # collapse the groups together (the service must never do that).
    positions = [(x / (n - 1), y / (n - 1), 0.0)
                 for y in range(n) for x in range(n)]
    uvs_a = [(x / (n - 1), y / (n - 1))
             for y in range(n) for x in range(n)]
    uvs_b = [(1.0 - x / (n - 1), 1.0 - y / (n - 1))
             for y in range(n) for x in range(n)]
    uvs_c = [((x / (n - 1)) * 0.5, (y / (n - 1)) * 0.5)
             for y in range(n) for x in range(n)]

    faces = []
    for y in range(n - 1):
        for x in range(n - 1):
            a = y * n + x + 1
            b = a + 1
            d = (y + 1) * n + x + 1
            c = d + 1
            faces.append((a, b, c, d))

    lines = ["# e2e multi-material fixture", "mtllib materials.mtl"]
    # Three independent vertex/uv blocks, one per material group.
    for _ in range(3):
        for x, y, z in positions:
            lines.append(f"v {x:.6f} {y:.6f} {z:.6f}")
    for uvs in (uvs_a, uvs_b, uvs_c):
        for u, v in uvs:
            lines.append(f"vt {u:.6f} {v:.6f}")

    group_specs = [
        ("mat_png", 1, len(positions), 1),
        ("mat_tga", len(positions) + 1, 2 * len(positions), len(positions) + 1),
        ("mat_jpg", 2 * len(positions) + 1, 3 * len(positions),
         2 * len(positions) + 1),
    ]
    for material, v0, _v1, vt0 in group_specs:
        lines.append(f"usemtl {material}")
        lines.append(f"g {material}")
        for a, b, c, d in faces:
            lines.append(
                f"f {v0 + a - 1}/{vt0 + a - 1} {v0 + b - 1}/{vt0 + b - 1} "
                f"{v0 + c - 1}/{vt0 + c - 1} {v0 + d - 1}/{vt0 + d - 1}")
    with open(obj_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")


def write_mtl(mtl_path):
    lines = [
        "newmtl mat_png",
        "Ka 0.2 0.2 0.2",
        "Kd 0.8 0.2 0.2",
        "map_kd tex_a.png",
        "",
        "newmtl mat_tga",
        "Ka 0.2 0.2 0.2",
        "Kd 0.2 0.8 0.2",
        "map_kd tex_b.tga",
        "",
        "newmtl mat_jpg",
        "Ka 0.2 0.2 0.2",
        "Kd 0.2 0.2 0.8",
        "map_kd tex_c.jpg",
        "",
    ]
    with open(mtl_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines))


def write_textures(folder):
    rng = np.random.RandomState(7)
    Image.fromarray((rng.rand(32, 32, 3) * 255).astype(np.uint8)).save(
        os.path.join(folder, "tex_a.png"), format="PNG")
    Image.fromarray((rng.rand(32, 32, 3) * 255).astype(np.uint8)).save(
        os.path.join(folder, "tex_b.tga"), format="TGA")
    Image.fromarray((rng.rand(32, 32, 3) * 255).astype(np.uint8)).save(
        os.path.join(folder, "tex_c.jpg"), format="JPEG", quality=90)


def write_glb(glb_path):
    import trimesh

    rng = np.random.RandomState(11)
    spheres = []
    for index, color in enumerate((200, 60)):
        mesh = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
        uv = rng.rand(len(mesh.vertices), 2)
        texture = Image.fromarray(
            np.full((32, 32, 3), color, dtype=np.uint8))
        mesh.visual = trimesh.visual.texture.TextureVisuals(
            uv=uv,
            material=trimesh.visual.material.PBRMaterial(
                baseColorTexture=texture,
                name=f"sphere_mat_{index}",
                metallicFactor=0.0,
                roughnessFactor=0.9,
            ),
        )
        spheres.append(mesh)
    scene = trimesh.Scene({"sphere_a": spheres[0], "sphere_b": spheres[1]})
    scene.export(glb_path)


def main(out_dir):
    folder = os.path.join(out_dir, "multi_material")
    os.makedirs(folder, exist_ok=True)
    write_textures(folder)
    write_mtl(os.path.join(folder, "materials.mtl"))
    write_grid_obj(os.path.join(folder, "materials_grid.obj"))
    write_glb(os.path.join(folder, "spheres.glb"))

    with open(os.path.join(folder, "aaa_corrupt.obj"), "w",
              encoding="utf-8") as f:
        f.write("v not numbers here\nf 1 2 3\n")
    with open(os.path.join(folder, "aaa_corrupt.glb"), "wb") as f:
        f.write(b"NOTGLB" + os.urandom(64))
    print(folder)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: make_fixtures.py <out_dir>")
        sys.exit(1)
    main(sys.argv[1])
