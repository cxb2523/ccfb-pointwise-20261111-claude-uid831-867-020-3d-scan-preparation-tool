"""Multi-material OBJ/MTL parser and writer.

The parser keeps every ``usemtl`` group as an independent vertex set, which is
what lets :mod:`scanprep3d.decimate` guarantee that no vertex is ever welded
across a material boundary (cross-material welding is what causes texture
bleeding between material groups).

Only the subset needed by scan assets is handled: ``v``, ``vt``, ``vn``,
``f`` (triangulating n-gons), ``usemtl``, ``mtllib`` and group/object names.
"""

import os
from collections import OrderedDict

import numpy as np


class ObjGroup:
    """One ``usemtl`` partition with its own corner-indexed vertex arrays."""

    def __init__(self, material):
        self.material = material
        # Corner instances: OBJ allows the same position to carry several UVs
        # or normals, so the working vertex is a (vi, ti, ni) triple.
        self.instances = []          # list of (vi, ti, ni)
        self.instance_index = {}     # (vi, ti, ni) -> row in self.instances
        self.faces = []              # list of triples of instance rows

    def add_corner(self, vi, ti, ni):
        key = (vi, ti, ni)
        row = self.instance_index.get(key)
        if row is None:
            row = len(self.instances)
            self.instance_index[key] = row
            self.instances.append(key)
        return row


class ObjMesh:
    def __init__(self):
        self.vertices = []    # list of (x, y, z)
        self.uvs = []         # list of (u, v)
        self.normals = []     # list of (x, y, z)
        self.groups = []      # list of ObjGroup
        self.mtllib = None
        self.object_names = []

    def group_for(self, material):
        if self.groups and self.groups[-1].material == material:
            return self.groups[-1]
        group = ObjGroup(material)
        self.groups.append(group)
        return group


DEFAULT_MATERIAL = "__scanprep_default__"


def _resolve_index(index, count):
    """OBJ supports negative (relative) indices."""
    if index > 0:
        return index - 1
    return count + index


def _parse_face_corner(token):
    # Forms: v | v/vt | v//vn | v/vt/vn
    parts = token.split("/")
    vi = int(parts[0])
    ti = int(parts[1]) if len(parts) > 1 and parts[1] else 0
    ni = int(parts[2]) if len(parts) > 2 and parts[2] else 0
    return vi, ti, ni


def parse_obj(obj_path):
    mesh = ObjMesh()
    current_material = DEFAULT_MATERIAL
    v_count = 0
    vt_count = 0
    vn_count = 0

    with open(obj_path, "r", encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            tokens = line.split()
            tag = tokens[0]
            if tag == "v":
                mesh.vertices.append(tuple(float(x) for x in tokens[1:4]))
                v_count += 1
            elif tag == "vt":
                mesh.uvs.append((float(tokens[1]), float(tokens[2]) if len(tokens) > 2 else 0.0))
                vt_count += 1
            elif tag == "vn":
                mesh.normals.append(tuple(float(x) for x in tokens[1:4]))
                vn_count += 1
            elif tag in ("o", "g") and len(tokens) > 1:
                mesh.object_names.append(" ".join(tokens[1:]))
            elif tag == "mtllib" and len(tokens) > 1:
                mesh.mtllib = " ".join(tokens[1:]).strip().strip('"')
            elif tag == "usemtl":
                current_material = " ".join(tokens[1:]).strip() or DEFAULT_MATERIAL
            elif tag == "f":
                corners = [_parse_face_corner(tok) for tok in tokens[1:]]
                corners = [
                    (_resolve_index(vi, v_count),
                     _resolve_index(ti, vt_count) if ti else -1,
                     _resolve_index(ni, vn_count) if ni else -1)
                    for vi, ti, ni in corners
                ]
                group = mesh.group_for(current_material)
                # Fan triangulation preserves every corner's v/vt/vn pairing.
                rows = [group.add_corner(vi, ti, ni) for vi, ti, ni in corners]
                for i in range(1, len(rows) - 1):
                    mesh_group_face(group, rows, i)
    return mesh


def mesh_group_face(group, rows, i):
    group.faces.append((rows[0], rows[i], rows[i + 1]))


# --------------------------------------------------------------------------- #
# MTL
# --------------------------------------------------------------------------- #

# map_* keys whose final token is a texture file path.
TEXTURE_MAP_KEYS = {
    "map_kd", "map_ka", "map_ks", "map_ke", "map_ns", "map_d",
    "map_bump", "bump", "map_normal", "map_pr", "map_pm", "map_ps",
    "norm", "map_orm",
}


class MtlLibrary:
    def __init__(self):
        # name -> list of (raw_key, options_prefix, texture_token_or_None, raw_line)
        self.materials = OrderedDict()
        self.current = None

    def start(self, name):
        self.current = name
        self.materials.setdefault(name, [])

    def add_line(self, key, raw_line):
        if self.current is None:
            self.start(DEFAULT_MATERIAL)
        texture_token = None
        options = ""
        low = key.lower()
        if low in TEXTURE_MAP_KEYS:
            tokens = raw_line.split(None, 1)[1] if " " in raw_line else ""
            pieces = tokens.split()
            if pieces:
                texture_token = pieces[-1].strip('"')
                options = " ".join(pieces[:-1])
        self.materials[self.current].append((key, options, texture_token, raw_line))


def parse_mtl(mtl_path):
    lib = MtlLibrary()
    if not os.path.isfile(mtl_path):
        return lib
    with open(mtl_path, "r", encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            tokens = line.split(None, 1)
            key = tokens[0]
            if key.lower() == "newmtl":
                lib.start(tokens[1].strip() if len(tokens) > 1 else DEFAULT_MATERIAL)
            else:
                lib.add_line(key, line)
    return lib


def rewrite_mtl(lib, texture_resolver, out_mtl_path):
    """Write a new MTL, replacing texture paths through the shared manifest.

    ``texture_resolver`` maps an absolute source texture path to the new
    path *relative to the MTL file*. Lines without textures are preserved
    verbatim.
    """
    lines = ["# Rewritten by scanprep3d - texture paths come from the shared manifest"]
    for name, entries in lib.materials.items():
        lines.append(f"newmtl {name}")
        for key, options, texture_token, raw_line in entries:
            if texture_token is None:
                lines.append(raw_line)
                continue
            new_ref = texture_resolver(texture_token)
            if new_ref is None:
                lines.append(raw_line)  # unresolved: keep original reference
            else:
                prefix = f"{key} {options} " if options else f"{key} "
                lines.append(prefix + new_ref.replace("\\", "/"))
        lines.append("")
    tmp = out_mtl_path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines))
    os.replace(tmp, out_mtl_path)


# --------------------------------------------------------------------------- #
# Writer
# --------------------------------------------------------------------------- #

def write_obj(out_path, group_meshes, mtl_name, materials_used):
    """Write an OBJ from decimated group meshes.

    ``group_meshes`` is a list of (material_name, vertices, uvs, faces,
    normals) where faces index into that group's arrays. Groups are emitted
    as separate ``usemtl`` blocks with independent index ranges.
    """
    lines = ["# Written by scanprep3d"]
    if mtl_name:
        lines.append(f"mtllib {mtl_name}")

    vertex_offset = 0
    uv_offset = 0
    normal_offset = 0
    for material, verts, uvs, faces, normals in group_meshes:
        if mtl_name and material != DEFAULT_MATERIAL and material in materials_used:
            lines.append(f"usemtl {material}")
        lines.append(f"g {_safe_name(material)}")
        for x, y, z in verts:
            lines.append(f"v {x:.6f} {y:.6f} {z:.6f}")
        has_uv = uvs is not None and len(uvs)
        if has_uv:
            for u, v in uvs:
                lines.append(f"vt {u:.6f} {v:.6f}")
        has_normal = normals is not None and len(normals)
        if has_normal:
            for x, y, z in normals:
                lines.append(f"vn {x:.6f} {y:.6f} {z:.6f}")
        for face in faces:
            corner_tokens = []
            for row in face:
                v_idx = vertex_offset + row + 1
                if has_uv and has_normal:
                    corner_tokens.append(
                        f"{v_idx}/{uv_offset + row + 1}/{normal_offset + row + 1}")
                elif has_uv:
                    corner_tokens.append(f"{v_idx}/{uv_offset + row + 1}")
                elif has_normal:
                    corner_tokens.append(f"{v_idx}//{normal_offset + row + 1}")
                else:
                    corner_tokens.append(str(v_idx))
            lines.append("f " + " ".join(corner_tokens))
        vertex_offset += len(verts)
        uv_offset += len(uvs) if has_uv else 0
        normal_offset += len(normals) if has_normal else 0

    tmp = out_path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    os.replace(tmp, out_path)


def _safe_name(name):
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(name))[:64]


def build_group_arrays(mesh):
    """Expand parsed groups into numpy (vertices, uvs, normals, faces) arrays."""
    result = []
    for group in mesh.groups:
        verts = np.empty((len(group.instances), 3), dtype=np.float64)
        uvs = np.full((len(group.instances), 2), np.nan, dtype=np.float64)
        normals = np.full((len(group.instances), 3), np.nan, dtype=np.float64)
        for row, (vi, ti, ni) in enumerate(group.instances):
            verts[row] = mesh.vertices[vi]
            if ti >= 0 and ti < len(mesh.uvs):
                uvs[row] = mesh.uvs[ti]
            if ni >= 0 and ni < len(mesh.normals):
                normals[row] = mesh.normals[ni]
        if np.isnan(uvs).all():
            uvs = None
        if np.isnan(normals).all():
            normals = None
        faces = np.array(group.faces, dtype=np.int64) if group.faces \
            else np.empty((0, 3), dtype=np.int64)
        result.append((group.material, verts, uvs, normals, faces))
    return result
