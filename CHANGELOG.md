# Changelog

## Unreleased

- Added a 3D model batch service for OBJ/GLB scan folders (`scanprep3d/`,
  `scanprep3d_service.py`, also exposed as `--batch-service`). The Electron
  panel spawns it as a child process and renders NDJSON progress/result
  events through a restricted preload bridge; the renderer never touches the
  filesystem.
- Recursive model discovery feeds a bounded queue: serial by default, hard
  concurrency cap of 2 because 3D decoding and texture compression are
  memory-heavy. Scene and image objects are released explicitly after every
  task.
- Decimation runs per material group with UV borders locked; vertices are
  never welded across materials, preventing texture color bleed.
- PNG/TGA textures transcode to WebP or KTX2 through one shared resource
  manifest, and MTL/glTF references are rewritten only after every conversion
  has landed in staging, so mid-run failures cannot leave half-rewritten
  models. KTX2 uses an uncompressed standards-valid fallback when no BasisU
  encoder is installed.
- Per-file failures emit `task_failed` with details and are isolated from the
  rest of the batch. Outputs go to sibling `*_processed` folders; name
  collisions get numeric suffixes and sources are never overwritten.
- Added `npm run e2e:batch` covering a multi-material OBJ, a textured GLB,
  corrupt-file isolation and continuation, face counts, texture loading and
  rerun naming.

## 1.0.1

- Fixed 360 video extraction when using Every Nth frame. Sparse 360 extraction now uses the faster seek-based extractor instead of scanning the full video stream.
- Improved preview-column layout on smaller Windows displays and non-maximized windows.
- Redirected AI/library cache and config folders to the user's app data area so the app should not need administrator mode for normal use.
- Hid noisy first-run Ultralytics/YOLO settings messages from normal Runtime Notes.

## 1.0.0

- Initial public release.
