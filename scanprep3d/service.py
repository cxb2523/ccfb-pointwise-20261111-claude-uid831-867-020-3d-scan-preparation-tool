"""Batch service: recursive discovery, bounded queue runner, per-file pipeline.

Concurrency is deliberately conservative because 3D decoding and texture
compression are memory-heavy: the default is serial (1 worker) and the
configuration clamps at 2. Each task owns its scene/image objects and
releases them explicitly in ``finally`` blocks, so a long batch of large
scans does not accumulate resident memory.

A failure in one file is recorded with full detail and surfaced as a
``task_failed`` event; it never aborts the queue.
"""

import gc
import os
import shutil
import sys
import tempfile
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from . import decimate as decim
from . import events
from . import glb_io
from . import glb_mesh
from . import obj_io
from .textures import TextureManifest, transcode_bytes

MODEL_EXTS = (".obj", ".glb")
MAX_CONCURRENCY = 2
DEFAULT_CONCURRENCY = 1
PROCESSED_SUFFIX = "_processed"


class BatchCancelled(Exception):
    pass


# Module-level cancel flag flipped by the stdin watcher (CLI) or a host API.
_cancel_event = threading.Event()
_commit_lock = threading.Lock()


def cancel():
    """Request cooperative cancellation; tasks stop at their next stage."""
    _cancel_event.set()


def is_cancelled():
    return _cancel_event.is_set()


def start_stdin_cancel_watcher():
    """Cooperatively cancel the batch when the host writes ``cancel``."""
    def watch():
        try:
            if sys.stdin is None or sys.stdin.closed:
                return
            for line in sys.stdin:
                if line.strip().lower() in {"cancel", "stop", "quit"}:
                    cancel()
                    break
        except Exception:
            pass

    thread = threading.Thread(target=watch, name="scanprep3d-cancel", daemon=True)
    thread.start()
    return thread


# --------------------------------------------------------------------------- #
# Discovery and output paths
# --------------------------------------------------------------------------- #

def discover_models(input_path, recursive=True):
    """Return a sorted list of OBJ/GLB files, skipping processed output trees."""
    if os.path.isfile(input_path):
        return [os.path.abspath(input_path)] \
            if input_path.lower().endswith(MODEL_EXTS) else []

    found = []
    for root, dirs, files in os.walk(input_path):
        # Never descend into processed output, caches or VCS folders.
        dirs[:] = sorted(
            d for d in dirs
            if not d.startswith(".")
            and not d.lower().endswith(PROCESSED_SUFFIX)
            and d.lower() not in {"__pycache__", "node_modules"}
        )
        for name in sorted(files):
            if name.lower().endswith(MODEL_EXTS) and not name.startswith("~"):
                found.append(os.path.abspath(os.path.join(root, name)))
        if not recursive:
            break
    return sorted(found)


def plan_output_dir(input_path, source_file, configured_output=None):
    """Choose the sibling ``*_processed`` directory for one source file."""
    if configured_output:
        output_root = os.path.abspath(configured_output)
    elif os.path.isfile(input_path):
        # Single-file input: a *_processed folder next to the file.
        stem = os.path.splitext(os.path.basename(input_path))[0]
        output_root = os.path.abspath(
            os.path.join(os.path.dirname(input_path), stem + PROCESSED_SUFFIX))
    else:
        # Folder input: a sibling folder named <folder>_processed.
        input_abs = os.path.abspath(input_path)
        parent = os.path.dirname(input_abs)
        output_root = os.path.join(
            parent, os.path.basename(input_abs.rstrip(os.sep)) + PROCESSED_SUFFIX)

    if os.path.isfile(input_path):
        relative_dir = ""
    else:
        relative_dir = os.path.relpath(os.path.dirname(source_file),
                                       os.path.abspath(input_path))
        if relative_dir == ".":
            relative_dir = ""
    return os.path.join(output_root, relative_dir) if relative_dir else output_root


def unique_output_path(target_path):
    """Append _1, _2, ... until the path is free. Never overwrites."""
    if not os.path.exists(target_path):
        return target_path
    stem, ext = os.path.splitext(target_path)
    counter = 1
    while True:
        candidate = f"{stem}_{counter}{ext}"
        if not os.path.exists(candidate):
            return candidate
        counter += 1


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #

def _smooth_normals(vertices, faces):
    normals = np.zeros_like(vertices)
    if not len(faces):
        return normals
    tris = vertices[faces]
    crossed = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    for face_index, face in enumerate(faces):
        normals[face] += crossed[face_index]
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    lengths[lengths == 0] = 1.0
    return normals / lengths


def _release(*objects):
    for obj in objects:
        try:
            del obj
        except Exception:
            pass
    gc.collect()


# --------------------------------------------------------------------------- #
# OBJ pipeline
# --------------------------------------------------------------------------- #

def _resolve_texture(mtl_path, token):
    if os.path.isabs(token):
        return os.path.normpath(token)
    return os.path.normpath(os.path.join(os.path.dirname(mtl_path), token))


def _process_obj(source_file, staging_dir, config, progress_cb):
    progress_cb("parse", 10, "Parsing OBJ/MTL")
    mesh = obj_io.parse_obj(source_file)
    groups = obj_io.build_group_arrays(mesh)
    if not groups:
        raise ValueError("OBJ contains no faces or material groups")

    input_faces = sum(len(g[4]) for g in groups)

    progress_cb("decimate", 30, "Decimating material groups")
    decimated_groups = []
    decimation_stats = []
    for material, vertices, uvs, normals, faces in groups:
        out_v, out_uv, out_f, stats, _remap = decim.decimate_group(
            vertices, uvs, faces, config["decimate_ratio"])
        stats["material"] = material
        out_normals = _smooth_normals(out_v, out_f)
        decimated_groups.append((material, out_v, out_uv, out_f, out_normals))
        decimation_stats.append(stats)
    output_faces = sum(len(g[3]) for g in decimated_groups)

    # --- textures: one manifest drives every MTL path rewrite -------------- #
    manifest = TextureManifest(
        staging_dir,
        target_format=config["texture_format"],
        webp_quality=config["webp_quality"],
        external_ktx2_encoder=config.get("ktx2_encoder"),
    )
    mtllib = mesh.mtllib
    mtl_lib = None
    mtl_source_path = None
    texture_warnings = []
    if mtllib:
        mtl_source_path = _resolve_texture(source_file, mtllib)
        mtl_lib = obj_io.parse_mtl(mtl_source_path)
        for _name, entries in mtl_lib.materials.items():
            for _key, _options, texture_token, _raw in entries:
                if texture_token is None:
                    continue
                texture_abs = _resolve_texture(mtl_source_path, texture_token)
                if not os.path.isfile(texture_abs):
                    texture_warnings.append(
                        f"missing texture kept as-is: {texture_token}")
                    continue
                manifest.add(texture_abs)

    if manifest.records():
        progress_cb("textures", 55, "Transcoding textures")
        manifest.prepare_all(
            progress_cb=lambda index, total, name:
                progress_cb("textures", 55 + int(30 * index / total),
                            f"Texture {index}/{total}: {name}"))

    # --- write MTL + OBJ into staging -------------------------------------- #
    progress_cb("write", 88, "Writing OBJ/MTL")
    materials_used = {g[0] for g in decimated_groups}
    mtl_output_name = None
    if mtl_lib is not None:
        mtl_output_name = os.path.splitext(os.path.basename(
            mtl_source_path or "materials.mtl"))[0] + ".mtl"
        staged_mtl = os.path.join(staging_dir, mtl_output_name)

        def resolver(texture_token, _mtl_src=mtl_source_path):
            texture_abs = _resolve_texture(_mtl_src, texture_token)
            new_rel = manifest.rewrite_for_source(texture_abs)
            return new_rel  # already relative to staging root == MTL location

        obj_io.rewrite_mtl(mtl_lib, resolver, staged_mtl)

    staged_obj = os.path.join(staging_dir, os.path.basename(source_file))
    obj_io.write_obj(staged_obj, decimated_groups, mtl_output_name, materials_used)

    _release(mesh, groups, decimated_groups, mtl_lib, manifest)
    return {
        "input_faces": int(input_faces),
        "output_faces": int(output_faces),
        "materials": len(materials_used),
        "decimation": decimation_stats,
        "textures": _texture_summary(manifest) if manifest.records() else [],
        "warnings": texture_warnings,
    }


def _texture_summary(manifest):
    return [{
        "source": os.path.basename(record["source"]),
        "output": record["output_rel"],
        "engine": record["engine"],
        "transcoded": record["transcoded"],
        "width": record["width"],
        "height": record["height"],
    } for record in manifest.records()]


# --------------------------------------------------------------------------- #
# GLB pipeline
# --------------------------------------------------------------------------- #

def _process_glb(source_file, staging_dir, config, progress_cb):
    progress_cb("parse", 10, "Reading GLB container")
    container = glb_io.read_glb(source_file)
    header = container.header

    input_faces = 0
    for mesh in header.get("meshes", []):
        for primitive in mesh.get("primitives", []):
            if "indices" in primitive:
                input_faces += len(glb_mesh.decode_accessor(
                    container, primitive["indices"])) // 3
            else:
                input_faces += header["accessors"][
                    primitive["attributes"]["POSITION"]]["count"] // 3

    progress_cb("decimate", 30, "Decimating primitives")
    decimation_stats = glb_mesh.rebuild_glb_geometry(
        container, config["decimate_ratio"])
    output_faces = sum(s["output_faces"] for s in decimation_stats)

    texture_results = []
    if config["texture_format"] != "none" and header.get("images"):
        progress_cb("textures", 55, "Transcoding embedded textures")
        payloads = {}
        total_images = len(header["images"])
        for image_index, image in enumerate(header["images"]):
            view_index = image.get("bufferView")
            if view_index is None:
                continue
            source_bytes = container.view_bytes(view_index)
            mime_in = image.get("mimeType", "")
            # Spec scope: transcode PNG (and TGA, impossible inside GLB);
            # JPEG and already-compressed images pass through unchanged.
            if mime_in != glb_io.MIME_PNG:
                texture_results.append({
                    "source": f"image[{image_index}]",
                    "output": f"image[{image_index}]",
                    "engine": "passthrough",
                    "transcoded": False,
                    "mime": mime_in,
                })
                continue
            out_bytes, mime, engine, width, height = transcode_bytes(
                source_bytes,
                config["texture_format"],
                webp_quality=config["webp_quality"],
                external_ktx2_encoder=config.get("ktx2_encoder"),
            )
            payloads[image_index] = (out_bytes, mime, config["texture_format"])
            texture_results.append({
                "source": f"image[{image_index}]",
                "output": f"image[{image_index}]",
                "engine": engine,
                "transcoded": True,
                "mime": mime,
                "width": width,
                "height": height,
            })
            progress_cb("textures", 55 + int(30 * (image_index + 1) /
                                             max(1, total_images)),
                        f"Texture {image_index + 1}/{total_images}")
        if payloads:
            glb_mesh.replace_glb_images(container, payloads)

    progress_cb("write", 88, "Writing GLB")
    staged_glb = os.path.join(staging_dir, os.path.basename(source_file))
    glb_io.write_glb(container, staged_glb)
    _release(container)
    return {
        "input_faces": int(input_faces),
        "output_faces": int(output_faces),
        "materials": len(header.get("materials", [])),
        "decimation": decimation_stats,
        "textures": texture_results,
        "warnings": [],
    }


# --------------------------------------------------------------------------- #
# Task runner
# --------------------------------------------------------------------------- #

def _commit_staging(staging_dir, output_dir, file_name):
    os.makedirs(output_dir, exist_ok=True)
    # Lock so two concurrent tasks targeting the same output dir cannot race
    # on unique-name allocation.
    with _commit_lock:
        final_path = unique_output_path(os.path.join(output_dir, file_name))
        # Staging holds one model plus its textures/ folder and manifest.
        for name in os.listdir(staging_dir):
            staged = os.path.join(staging_dir, name)
            target = unique_output_path(os.path.join(output_dir, name))
            if os.path.isdir(staged):
                # Merge into an existing textures/ folder without overwriting.
                os.makedirs(target, exist_ok=True)
                for sub in os.listdir(staged):
                    sub_target = unique_output_path(os.path.join(target, sub))
                    os.replace(os.path.join(staged, sub), sub_target)
            else:
                os.replace(staged, target)
    return final_path


def process_one(source_file, input_path, config, task_index):
    """Run one model end-to-end. Never raises; returns a result dict."""
    ext = os.path.splitext(source_file)[1].lower()
    result = {
        "task": task_index,
        "source": source_file,
        "status": "failed",
        "output": None,
        "error": None,
    }
    staging_root = None
    try:
        if is_cancelled():
            raise BatchCancelled("batch cancelled before task start")

        output_dir = plan_output_dir(
            input_path, source_file, config.get("output_dir"))
        if os.path.abspath(output_dir) == os.path.dirname(source_file):
            raise ValueError("refusing to write into the source directory")
        os.makedirs(output_dir, exist_ok=True)
        staging_root = tempfile.mkdtemp(
            prefix=f".staging-{task_index}-", dir=output_dir)

        def progress_cb(stage, percent, message):
            if is_cancelled():
                raise BatchCancelled("batch cancelled during processing")
            events.emit(
                events.TASK_PROGRESS,
                task=task_index, source=source_file,
                stage=stage, percent=int(percent), message=message)

        if ext == ".obj":
            details = _process_obj(
                source_file, staging_root, config, progress_cb)
        elif ext == ".glb":
            details = _process_glb(
                source_file, staging_root, config, progress_cb)
        else:
            raise ValueError(f"unsupported model extension: {ext}")

        file_name = os.path.basename(source_file)
        final_path = _commit_staging(staging_root, output_dir, file_name)
        result.update(status="completed", output=final_path, **details)
        events.emit(events.TEXTURES_PREPARED,
                    task=task_index, source=source_file,
                    textures=details.get("textures", []),
                    warnings=details.get("warnings", []))
        events.emit(events.TASK_COMPLETED, **result)
    except BatchCancelled as exc:
        result["error"] = str(exc)
        result["status"] = "cancelled"
        events.emit(events.TASK_CANCELLED, **result,
                    error_type="cancelled")
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc(limit=6)
        # Single-file failure: record detail, emit failed, keep the batch alive.
        events.emit(events.TASK_FAILED, **result,
                    error_type=type(exc).__name__)
    finally:
        if staging_root and os.path.isdir(staging_root):
            shutil.rmtree(staging_root, ignore_errors=True)
        gc.collect()
    return result


def run_batch(config):
    """Entry point used by the ``--batch-service`` CLI command."""
    input_path = config["input_path"]
    if not os.path.exists(input_path):
        events.emit(events.ERROR, message=f"input path does not exist: {input_path}")
        return 1

    config = dict(config)
    config.setdefault("recursive", True)
    config.setdefault("decimate_ratio", 0.5)
    config.setdefault("texture_format", "webp")
    config.setdefault("webp_quality", 90)
    if config["texture_format"] not in ("webp", "ktx2", "none"):
        events.emit(events.ERROR,
                    message=f"invalid texture_format: {config['texture_format']}")
        return 1

    try:
        concurrency = int(config.get("concurrency", DEFAULT_CONCURRENCY))
    except (TypeError, ValueError):
        concurrency = DEFAULT_CONCURRENCY
    concurrency = max(1, min(MAX_CONCURRENCY, concurrency))
    config["concurrency"] = concurrency

    sources = discover_models(input_path, recursive=bool(config["recursive"]))
    if not sources:
        events.emit(events.STATUS,
                    message="No OBJ/GLB models found under the input path.")
        events.emit(events.BATCH_COMPLETED, total=0, completed=0, failed=0,
                    cancelled=0, results=[])
        return 0

    events.emit(events.BATCH_STARTED,
                input_path=os.path.abspath(input_path),
                total=len(sources), concurrency=concurrency,
                texture_format=config["texture_format"],
                decimate_ratio=config["decimate_ratio"])
    for index, source in enumerate(sources):
        events.emit(events.TASK_QUEUED, task=index, source=source)

    results = []
    results_lock = threading.Lock()

    def worker(task_index, source):
        events.emit(events.TASK_STARTED, task=task_index, source=source)
        task_result = process_one(source, input_path, config, task_index)
        with results_lock:
            results.append(task_result)
        return task_result

    if concurrency == 1:
        for index, source in enumerate(sources):
            # process_one short-circuits to a "cancelled" result after cancel.
            worker(index, source)
    else:
        with ThreadPoolExecutor(max_workers=concurrency,
                                thread_name_prefix="scanprep3d") as pool:
            futures = [pool.submit(worker, index, source)
                       for index, source in enumerate(sources)]
            for future in futures:
                # per-task exceptions are already contained; result() re-raises
                # only infrastructure errors, which should surface.
                future.result()

    completed = sum(1 for r in results if r["status"] == "completed")
    failed = sum(1 for r in results if r["status"] == "failed")
    cancelled = sum(1 for r in results if r["status"] == "cancelled")

    if is_cancelled():
        events.emit(events.BATCH_CANCELLED,
                    total=len(sources), completed=completed, failed=failed,
                    cancelled=cancelled, results=results)
        return 2

    # Per-file failures are data, not service failures: the batch itself ran
    # to completion, so exit 0 and let the failed counts/events drive the UI.
    events.emit(events.BATCH_COMPLETED,
                total=len(sources), completed=completed, failed=failed,
                cancelled=cancelled, results=results)
    return 0
