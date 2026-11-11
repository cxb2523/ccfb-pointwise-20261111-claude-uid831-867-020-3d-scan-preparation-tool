"""GLB chunk reader/writer.

Geometry accessors, buffer views and material parameters live in this
container; :mod:`scanprep3d.glb_mesh` mutates them in place and calls
:func:`write_glb` once, so a mid-pipeline failure never leaves a GLB whose
JSON points at missing or mismatched bytes.
"""

import json
import os
import struct

GLB_MAGIC = 0x46546C67          # "glTF"
GLB_VERSION = 2
CHUNK_JSON = 0x4E4F534A         # "JSON"
CHUNK_BIN = 0x004E4942          # "BIN\0"

MIME_WEBP = "image/webp"
MIME_KTX2 = "image/ktx2"
MIME_PNG = "image/png"
MIME_JPEG = "image/jpeg"

EXT_WEBP = "EXT_texture_webp"
EXT_BASISU = "KHR_texture_basisu"


class GlbContainer:
    def __init__(self, header, bin_blob, source_path):
        self.header = header
        self.bin_blob = bin_blob
        self.source_path = source_path

    def view_bytes(self, view_index):
        view = self.header["bufferViews"][view_index]
        start = view.get("byteOffset", 0)
        return self.bin_blob[start:start + view["byteLength"]]

    def textures_using_image(self, image_index):
        """Return texture indexes that reference the given image."""
        result = []
        for texture_index, texture in enumerate(self.header.get("textures", [])):
            if texture.get("source") == image_index:
                result.append(texture_index)
            ext = texture.get("extensions", {})
            if ext.get(EXT_WEBP, {}).get("source") == image_index:
                result.append(texture_index)
            if ext.get(EXT_BASISU, {}).get("source") == image_index:
                result.append(texture_index)
        return sorted(set(result))


def read_glb(glb_path):
    with open(glb_path, "rb") as f:
        raw = f.read()
    magic, version, _total = struct.unpack("<III", raw[:12])
    if magic != GLB_MAGIC or version != GLB_VERSION:
        raise ValueError("not a GLB 2.0 file")
    offset = 12
    header = None
    bin_blob = b""
    while offset < len(raw):
        chunk_length, chunk_type = struct.unpack("<II", raw[offset:offset + 8])
        chunk_data = raw[offset + 8:offset + 8 + chunk_length]
        offset += 8 + chunk_length
        if chunk_type == CHUNK_JSON:
            header = json.loads(chunk_data.decode("utf-8"))
        elif chunk_type == CHUNK_BIN:
            bin_blob = chunk_data
    if header is None:
        raise ValueError("GLB missing JSON chunk")
    return GlbContainer(header, bin_blob, glb_path)


def write_glb(container, out_path):
    """Serialize the current header + BIN blob atomically."""
    header = container.header
    json_blob = json.dumps(header, separators=(",", ":")).encode("utf-8")
    json_blob += b" " * ((-len(json_blob)) % 4)
    bin_blob = container.bin_blob
    bin_blob += b"\x00" * ((-len(bin_blob)) % 4)
    total = 12 + 8 + len(json_blob) + 8 + len(bin_blob)
    tmp_path = out_path + ".tmp"
    with open(tmp_path, "wb") as f:
        f.write(struct.pack("<III", GLB_MAGIC, GLB_VERSION, total))
        f.write(struct.pack("<II", len(json_blob), CHUNK_JSON))
        f.write(json_blob)
        f.write(struct.pack("<II", len(bin_blob), CHUNK_BIN))
        f.write(bin_blob)
    os.replace(tmp_path, out_path)
