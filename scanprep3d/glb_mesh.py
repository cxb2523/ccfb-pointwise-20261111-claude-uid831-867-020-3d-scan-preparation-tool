"""Per-primitive GLB geometry rebuild with material-group-locked decimation.

Instead of round-tripping through a generic exporter (which can reorder or
drop textures), this module decodes glTF accessors itself, decimates each
primitive independently - primitives never share a simplifier call, so
material groups cannot be welded together - and then rebuilds the binary
chunk. Material, texture and image JSON entries are preserved; only the
buffer views they point at are replaced.
"""

import numpy as np

from . import decimate as decim
from .glb_io import EXT_BASISU, EXT_WEBP, GlbContainer, MIME_JPEG, MIME_KTX2, MIME_PNG, MIME_WEBP, write_glb

_COMPONENT = {
    5120: (np.int8, 1),
    5121: (np.uint8, 1),
    5122: (np.int16, 2),
    5123: (np.uint16, 2),
    5125: (np.uint32, 4),
    5126: (np.float32, 4),
}
_DTYPE_COMPONENT = {
    np.dtype(np.int8): 5120,
    np.dtype(np.uint8): 5121,
    np.dtype(np.int16): 5122,
    np.dtype(np.uint16): 5123,
    np.dtype(np.uint32): 5125,
    np.dtype(np.int32): 5125,
    np.dtype(np.float32): 5126,
}
_TYPE_COMPONENTS = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}
_FLOAT_TYPES = {5126}


class GlbGeometryError(Exception):
    """Raised for accessor features the rebuilder does not support."""


# --------------------------------------------------------------------------- #
# Decoding
# --------------------------------------------------------------------------- #

def _view_blob(container, view_index):
    view = container.header["bufferViews"][view_index]
    start = view.get("byteOffset", 0)
    return container.bin_blob[start:start + view["byteLength"]], view


def decode_accessor(container, accessor_index):
    """Decode an accessor into a ``(count, components)`` numpy array."""
    accessor = container.header["accessors"][accessor_index]
    if "sparse" in accessor:
        raise GlbGeometryError("sparse accessors are not supported")
    dtype, component_size = _COMPONENT[accessor["componentType"]]
    components = _TYPE_COMPONENTS[accessor["type"]]
    count = accessor["count"]

    view_index = accessor.get("bufferView")
    if view_index is None:
        return np.zeros((count, components), dtype=dtype)
    blob, view = _view_blob(container, view_index)
    stride = view.get("byteStride") or (component_size * components)
    base = accessor.get("byteOffset", 0)

    rows = np.empty((count, components), dtype=dtype)
    for row in range(count):
        offset = base + row * stride
        rows[row] = np.frombuffer(
            blob[offset:offset + component_size * components],
            dtype=dtype, count=components)
    return rows


def primitive_arrays(container, primitive):
    """Return ``(positions, uvs_or_None, attr_specs, faces)``.

    ``attr_specs`` maps each non-POSITION/UV attribute to
    ``(values, source_accessor)`` so encoding can keep the original dtype and
    normalized flag.
    """
    attributes = primitive["attributes"]
    positions = decode_accessor(container, attributes["POSITION"]).astype(np.float64)
    uvs = None
    if "TEXCOORD_0" in attributes:
        uvs = decode_accessor(
            container, attributes["TEXCOORD_0"]).astype(np.float64)

    attr_specs = {}
    for name, accessor_index in attributes.items():
        if name in ("POSITION", "TEXCOORD_0"):
            continue
        attr_specs[name] = (
            decode_accessor(container, accessor_index),
            container.header["accessors"][accessor_index],
        )

    if "indices" in primitive:
        indices = decode_accessor(
            container, primitive["indices"]).astype(np.int64).ravel()
        faces = indices.reshape(-1, 3)
    else:
        faces = np.arange(len(positions), dtype=np.int64).reshape(-1, 3)
    return positions, uvs, attr_specs, faces


# --------------------------------------------------------------------------- #
# Encoding
# --------------------------------------------------------------------------- #

class _BlobBuilder:
    def __init__(self):
        self.pieces = []
        self.length = 0

    def add(self, blob):
        padding = (-self.length) % 4
        if padding:
            self.pieces.append(b"\x00" * padding)
            self.length += padding
        # Offset must point at the data itself, i.e. after any alignment pad.
        offset = self.length
        self.pieces.append(blob)
        self.length += len(blob)
        return offset

    def finish(self):
        padding = (-self.length) % 4
        if padding:
            self.pieces.append(b"\x00" * padding)
            self.length += padding
        return b"".join(self.pieces)


def _area_weighted_normals(vertices, faces):
    normals = np.zeros_like(vertices)
    tris = vertices[faces]
    crossed = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    for face_index, face in enumerate(faces):
        normals[face] += crossed[face_index]
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    lengths[lengths == 0] = 1.0
    return normals / lengths


def _add_float_view(builder, header, array, accessor_type, target=None):
    components = _TYPE_COMPONENTS[accessor_type]
    flat = np.ascontiguousarray(array, dtype=np.float32).reshape(-1, components)
    offset = builder.add(flat.tobytes())
    view_index = len(header["bufferViews"])
    view = {"buffer": 0, "byteOffset": offset, "byteLength": flat.nbytes}
    if target is not None:
        view["target"] = target
    header["bufferViews"].append(view)
    accessor = {
        "bufferView": view_index, "componentType": 5126,
        "count": len(flat), "type": accessor_type,
    }
    if accessor_type == "VEC3":
        accessor["min"] = flat.min(axis=0).tolist()
        accessor["max"] = flat.max(axis=0).tolist()
    header["accessors"].append(accessor)
    return len(header["accessors"]) - 1


def _add_generic_view(builder, header, array, source_accessor, target=None):
    flat = np.ascontiguousarray(array)
    component_type = _DTYPE_COMPONENT.get(flat.dtype)
    if component_type is None:
        flat = flat.astype(np.float32)
        component_type = 5126
    offset = builder.add(flat.tobytes())
    view_index = len(header["bufferViews"])
    view = {"buffer": 0, "byteOffset": offset, "byteLength": flat.nbytes}
    if target is not None:
        view["target"] = target
    header["bufferViews"].append(view)
    accessor = {
        "bufferView": view_index,
        "componentType": component_type,
        "count": len(flat),
        "type": source_accessor["type"],
    }
    if source_accessor.get("normalized"):
        accessor["normalized"] = True
    header["accessors"].append(accessor)
    return len(header["accessors"]) - 1


def _add_index_view(builder, header, faces):
    flat = np.ascontiguousarray(faces, dtype=np.uint32).reshape(-1)
    offset = builder.add(flat.tobytes())
    view_index = len(header["bufferViews"])
    header["bufferViews"].append({
        "buffer": 0, "byteOffset": offset, "byteLength": flat.nbytes,
        "target": 34963,
    })
    header["accessors"].append({
        "bufferView": view_index, "componentType": 5125,
        "count": len(flat), "type": "SCALAR",
        "min": [int(flat.min())], "max": [int(flat.max())],
    })
    return len(header["accessors"]) - 1


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #

def rebuild_glb_geometry(container, target_reduction):
    """Decimate every primitive in-place on ``container``.

    Returns per-primitive stats. Image buffer views are preserved byte-for-byte
    and re-laid after the geometry views; the service swaps their contents
    afterwards from the shared texture manifest.
    """
    header = container.header

    # Phase 1: decode every primitive against the ORIGINAL accessor tables and
    # snapshot image bytes. Nothing is mutated yet, so indices stay valid.
    decoded = []
    input_face_total = 0
    for mesh in header.get("meshes", []):
        for primitive in mesh.get("primitives", []):
            if "targets" in primitive:
                # Morph targets cannot be remapped cheaply; fail this one file
                # loudly rather than silently corrupting blend shapes.
                raise GlbGeometryError("morph target primitives are not supported")
            positions, uvs, attr_specs, faces = primitive_arrays(
                container, primitive)
            decoded.append((mesh, primitive, positions, uvs, attr_specs, faces))
            input_face_total += len(faces)

    original_image_blobs = {}
    for image in header.get("images", []):
        view_index = image.get("bufferView")
        if view_index is not None:
            original_image_blobs[view_index] = container.view_bytes(view_index)

    # Phase 2: rebuild buffer views and accessors from the decoded data.
    builder = _BlobBuilder()
    header["bufferViews"] = []
    header["accessors"] = []
    stats = []

    for mesh, primitive, positions, uvs, attr_specs, faces in decoded:
        material = primitive.get("material")
        out_v, out_uv, out_f, group_stats, indice_map = decim.decimate_group(
            positions, uvs, faces, target_reduction)
        group_stats["material_index"] = material
        new_count = len(out_v)

        new_attributes = {}
        new_attributes["POSITION"] = _add_float_view(
            builder, header, out_v, "VEC3", target=34962)
        if out_uv is not None and len(out_uv):
            new_attributes["TEXCOORD_0"] = _add_float_view(
                builder, header, out_uv, "VEC2", target=34962)

        for name, (values, source_accessor) in attr_specs.items():
            if name == "NORMAL":
                remapped = _area_weighted_normals(out_v, out_f)
                new_attributes[name] = _add_float_view(
                    builder, header, remapped, "VEC3", target=34962)
                continue
            remapped = decim.remap_corner_array(
                values, indice_map, new_count)
            new_attributes[name] = _add_generic_view(
                builder, header, remapped, source_accessor, target=34962)

        new_primitive = {
            "attributes": new_attributes,
            "indices": _add_index_view(builder, header, out_f),
            "mode": primitive.get("mode", 4),
        }
        if material is not None:
            new_primitive["material"] = material
        mesh.setdefault("_scanprep_new_primitives", []).append(new_primitive)
        stats.append(group_stats)

    for mesh in header.get("meshes", []):
        if "_scanprep_new_primitives" in mesh:
            mesh["primitives"] = mesh.pop("_scanprep_new_primitives")

    # Re-append the preserved image views and repoint image entries at them.
    for image in header.get("images", []):
        old_view = image.get("bufferView")
        if old_view is None:
            continue
        blob = original_image_blobs[old_view]
        offset = builder.add(blob)
        new_view = len(header["bufferViews"])
        header["bufferViews"].append({
            "buffer": 0, "byteOffset": offset, "byteLength": len(blob),
        })
        image["bufferView"] = new_view

    container.bin_blob = builder.finish()
    header["buffers"][0]["byteLength"] = len(container.bin_blob)
    return stats


def _register_extension(header, name):
    used = header.setdefault("extensionsUsed", [])
    if name not in used:
        used.append(name)
    required = header.setdefault("extensionsRequired", [])
    if name not in required:
        required.append(name)


def replace_glb_images(container, image_payloads):
    """Swap image bytes from the manifest and rewrite texture extensions.

    ``image_payloads`` maps image index -> ``(bytes, mime, target_format)``.
    One final BIN rebuild means either every replacement lands or the caller
    aborts before :func:`write_glb` and nothing on disk changes.
    """
    header = container.header
    image_views = {
        img["bufferView"]: image_index
        for image_index, img in enumerate(header.get("images", []))
        if img.get("bufferView") is not None
    }

    builder = _BlobBuilder()
    for view_index, view in enumerate(header["bufferViews"]):
        if view_index in image_views:
            payload = image_payloads.get(image_views[view_index])
            blob = payload[0] if payload is not None else container.view_bytes(view_index)
        else:
            blob = container.view_bytes(view_index)
        offset = builder.add(blob)
        view["byteOffset"] = offset
        view["byteLength"] = len(blob)

    for image_index, payload in image_payloads.items():
        if payload is None:
            continue
        data, mime, target_format = payload
        image = header["images"][image_index]
        image["mimeType"] = mime
        for texture in header.get("textures", []):
            references = (
                texture.get("source") == image_index
                or texture.get("extensions", {}).get(EXT_WEBP, {}).get("source") == image_index
                or texture.get("extensions", {}).get(EXT_BASISU, {}).get("source") == image_index
            )
            if not references:
                continue
            texture.pop("source", None)
            texture.setdefault("extensions", {})
            if target_format == "webp":
                texture["extensions"][EXT_WEBP] = {"source": image_index}
                _register_extension(header, EXT_WEBP)
            elif target_format == "ktx2":
                texture["extensions"][EXT_BASISU] = {"source": image_index}
                _register_extension(header, EXT_BASISU)

    container.bin_blob = builder.finish()
    header["buffers"][0]["byteLength"] = len(container.bin_blob)


MIME_BY_EXT = {".png": MIME_PNG, ".jpg": MIME_JPEG, ".jpeg": MIME_JPEG,
               ".webp": MIME_WEBP, ".ktx2": MIME_KTX2}
