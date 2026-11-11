"""ScanPrep 3D batch service package.

Modules:
  events       - NDJSON event protocol shared with the Electron child process
  textures     - PNG/TGA -> WebP/KTX2 transcoding with a shared resource manifest
  ktx2_writer  - minimal uncompressed KTX2 container writer (+ external encoder hook)
  obj_io       - multi-material OBJ/MTL parser and writer (texture references preserved)
  glb_io       - GLB chunk repacking so image buffers can be swapped without geometry loss
  decimate     - quadric decimation locked to UV borders and material groups
  service      - recursive discovery, bounded queue runner, per-file failure isolation
"""

__all__ = [
    "events",
    "textures",
    "ktx2_writer",
    "obj_io",
    "glb_io",
    "decimate",
    "service",
]
