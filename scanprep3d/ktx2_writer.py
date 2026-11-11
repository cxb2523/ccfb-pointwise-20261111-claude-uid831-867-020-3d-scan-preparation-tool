"""Minimal KTX2 writer.

The deployment image does not ship a Basis/BC7 encoder (no ``toktx`` binary,
no ``imagecodecs`` BC7 support, no installable Python wheel), so by default we
write a standards-valid *uncompressed* KTX2 (R8G8B8A8 sRGB, vkFormat 43). When
a real encoder is available its path can be supplied through
``SCANPREP_KTX2_ENCODER`` (an executable taking ``input output``) and the
compressed output replaces the fallback. Either way the consumer receives a
``.ktx2`` file and the glTF/MTL reference is identical.

Layout follows the Khronos KTX v2 spec: 72-byte header, Data Format
Descriptor, Key/Value Data, then the single mip level (no Level Index for
levelCount == 1 with no supercompression).
"""

import os
import struct
import subprocess
import tempfile

from PIL import Image

# KTX2 constants (Khronos KTX v2 spec).
KTX2_IDENTIFIER = bytes([
    0xAB, 0x4B, 0x54, 0x58, 0x20, 0x32, 0x30, 0xBB,
    0x0D, 0x0A, 0x1A, 0x0A,
])
KTX2_HEADER_SIZE = 72
VK_FORMAT_R8G8B8A8_UNORM = 37
VK_FORMAT_R8G8B8A8_SRGB = 43

# DFD constants.
DFD_VENDOR_ID = 0
DFD_DESCRIPTOR_TYPE_BASIC = 0
DFD_VERSION = 2
DFD_BLOCK_TYPE_BASIC = 0
DFD_MODEL_RGBSDA = 2
DFD_PRIMARIES_SRGB = 1
DFD_TRANSFER_SRGB = 2
DFD_TRANSFER_LINEAR = 1


def _build_dfd(srgb=True):
    """Build a 4-byte-aligned DFD describing one 32-bit RGBA sample."""
    sample = struct.pack(
        "<HHBBII",
        0,                 # bitOffset (start of R)
        32,                # bitLength: R,G,B,A as one 32-bit texel sample
        0,                 # channelType: sampled color signal
        0,                 # samplePosition (0,0)
        0,                 # sampleLower qualifier
        0,                 # sampleUpper qualifier
    )
    block_size = 8 + 4 + 8 + len(sample)   # header + texelDim + bytesPlane + sample
    descriptor_block = struct.pack(
        "<HHBBBB",
        DFD_BLOCK_TYPE_BASIC,
        block_size,
        DFD_MODEL_RGBSDA,
        DFD_PRIMARIES_SRGB,
        DFD_TRANSFER_SRGB if srgb else DFD_TRANSFER_LINEAR,
        0,  # flags
    ) + struct.pack("<III", 0, 0, 0) + sample  # texelBlockDimension, bytesPlane[2]

    body = struct.pack(
        "<IIII",
        DFD_VENDOR_ID,
        DFD_DESCRIPTOR_TYPE_BASIC,
        DFD_VERSION,
        0,  # total size placeholder, patched after alignment
    ) + descriptor_block
    # The DFD byte length must be a multiple of 4.
    body += b"\x00" * ((-len(body)) % 4)
    total_size = len(body)
    body = struct.pack(
        "<IIII", DFD_VENDOR_ID, DFD_DESCRIPTOR_TYPE_BASIC, DFD_VERSION,
        total_size) + body[16:]
    return body


def _kv_entry(key, value):
    """One keyAndValueByteSize entry, padded to 4-byte alignment."""
    if isinstance(key, str):
        key = key.encode("utf-8")
    if isinstance(value, str):
        value = value.encode("utf-8")
    payload = key + b"\x00" + value
    block = struct.pack("<I", len(payload)) + payload
    return block + (b"\x00" * ((-len(block)) % 4))


def write_uncompressed_ktx2(image, out_path, srgb=True):
    """Write an RGBA PIL image as an uncompressed single-level KTX2."""
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    width, height = image.size
    pixels = image.tobytes()  # tightly packed RGBA rows
    level_byte_length = len(pixels)

    dfd = _build_dfd(srgb=srgb)
    kvd = _kv_entry("KTXwriter", "scanprep3d uncompressed fallback")

    dfd_offset = KTX2_HEADER_SIZE
    kvd_offset = dfd_offset + len(dfd)
    level_offset = kvd_offset + len(kvd)
    level_offset += (-level_offset) % 4  # level data is 4-byte aligned

    header = struct.pack(
        "<12sIIIIIIIII",
        KTX2_IDENTIFIER,
        VK_FORMAT_R8G8B8A8_SRGB if srgb else VK_FORMAT_R8G8B8A8_UNORM,
        1,            # typeSize (one byte per component)
        width,
        height,
        0,            # pixelDepth
        0,            # layerCount
        0,            # faceCount (0 means a regular 2D texture)
        1,            # levelCount
        0,            # supercompressionScheme (none)
    ) + struct.pack(
        "<IIIIII",
        dfd_offset, len(dfd),
        kvd_offset, len(kvd),
        0, 0,         # no supercompression global data
    )
    assert len(header) == KTX2_HEADER_SIZE

    tmp_path = out_path + ".tmp"
    with open(tmp_path, "wb") as f:
        f.write(header)
        f.write(dfd)
        f.write(kvd)
        f.write(b"\x00" * (level_offset - (kvd_offset + len(kvd))))
        # Single level: byteLength pair, then the texel bytes.
        f.write(struct.pack("<QQ", level_byte_length, level_byte_length))
        f.write(pixels)
    os.replace(tmp_path, out_path)
    return out_path


def encode_ktx2(image, out_path, srgb=True, external_encoder=None):
    """Encode to KTX2, preferring an external BasisU/toktx encoder.

    Returns ``(path, engine)`` where engine is "external" or "uncompressed".
    """
    external_encoder = external_encoder or os.environ.get("SCANPREP_KTX2_ENCODER")
    if external_encoder and os.path.isfile(external_encoder):
        with tempfile.TemporaryDirectory(prefix="scanprep_ktx2_") as tmp:
            png_in = os.path.join(tmp, "source.png")
            image.save(png_in, format="PNG")
            try:
                subprocess.run(
                    [external_encoder, png_in, out_path],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=120,
                )
                if os.path.isfile(out_path):
                    return out_path, "external"
            except (subprocess.SubprocessError, OSError):
                pass  # fall through to the in-process writer
    return write_uncompressed_ktx2(image, out_path, srgb=srgb), "uncompressed"


def ktx2_encoder_available(external_encoder=None):
    external_encoder = external_encoder or os.environ.get("SCANPREP_KTX2_ENCODER")
    return bool(external_encoder and os.path.isfile(external_encoder))
