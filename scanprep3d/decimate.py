"""Quadric decimation locked to UV borders and material groups.

Two invariants are enforced:

1. Material groups are decimated independently. Vertices from different
   ``usemtl`` partitions (or different glTF primitives) never enter the same
   simplifier call, so cross-material welding - and the texture color bleed it
   causes - is structurally impossible.
2. Within a group, corner vertices sharing a position but carrying different
   UVs stay as separate rows, and ``preserve_border`` keeps seam edges fixed.
   The simplifier's collapse history is replayed via
   :func:`fast_simplification.replay_simplification`, which yields an exact
   old-row -> new-row index map, so every surviving vertex inherits the UV of
   its pre-collapse corner without any positional guessing.
"""

import numpy as np

try:
    import fast_simplification as _fs
    SIMPLIFIER_AVAILABLE = True
    SIMPLIFIER_ERROR = ""
except Exception as exc:  # pragma: no cover - environment dependent
    _fs = None
    SIMPLIFIER_AVAILABLE = False
    SIMPLIFIER_ERROR = str(exc)


def decimate_group(vertices, uvs, faces, target_reduction):
    """Decimate one material group.

    Returns ``(vertices, uvs, faces, stats, indice_map)`` where
    ``indice_map[i]`` is the new row for original vertex ``i`` (or ``-1`` for
    vertices not referenced by any face). On simplifier failure or when
    decimation is disabled, the original arrays are returned with
    ``indice_map=None`` so the batch can continue.
    """
    vertices = np.ascontiguousarray(vertices, dtype=np.float64)
    faces = np.ascontiguousarray(faces, dtype=np.int64)
    stats = {"input_faces": int(len(faces)), "output_faces": int(len(faces)),
             "skipped": False, "reason": ""}

    if not SIMPLIFIER_AVAILABLE:
        stats["skipped"] = True
        stats["reason"] = f"simplifier unavailable: {SIMPLIFIER_ERROR}"
        return vertices, uvs, faces, stats, None

    if len(faces) < 8 or target_reduction is None or target_reduction <= 0.0:
        stats["skipped"] = True
        stats["reason"] = "mesh below threshold or reduction disabled"
        return vertices, uvs, faces, stats, None

    reduction = float(np.clip(target_reduction, 0.01, 0.95))
    try:
        _points, _faces, collapses = _fs.simplify(
            vertices, faces,
            target_reduction=reduction,
            preserve_border=True,
            return_collapses=True,
        )
        # Replay gives a deterministic old->new index map (and identical
        # geometry) without relying on vertex-position lookups.
        out_vertices, out_faces, indice_map = _fs.replay_simplification(
            vertices, faces, collapses)
    except Exception as exc:
        stats["skipped"] = True
        stats["reason"] = f"simplifier error, kept original: {type(exc).__name__}: {exc}"
        return vertices, uvs, faces, stats, None

    if len(out_faces) == 0:
        stats["skipped"] = True
        stats["reason"] = "simplifier produced no faces, kept original"
        return vertices, uvs, faces, stats, None

    # Replay may leave -1 entries for originally unreferenced vertices.
    valid = ((out_faces[:, 0] != out_faces[:, 1])
             & (out_faces[:, 1] != out_faces[:, 2])
             & (out_faces[:, 0] != out_faces[:, 2]))
    out_faces = np.ascontiguousarray(out_faces[valid], dtype=np.int64)
    if len(out_faces) == 0:
        stats["skipped"] = True
        stats["reason"] = "all simplified faces degenerate, kept original"
        return vertices, uvs, faces, stats, None

    out_uvs = None
    if uvs is not None and len(uvs):
        out_uvs = np.zeros((len(out_vertices), 2), dtype=np.float64)
        for old_index, new_index in enumerate(indice_map):
            if new_index >= 0:
                out_uvs[new_index] = uvs[old_index]

    stats["output_faces"] = int(len(out_faces))
    return out_vertices, out_uvs, out_faces, stats, indice_map


def remap_corner_array(values, indice_map, new_count):
    """Apply an old->new index map to any per-vertex attribute."""
    if indice_map is None:
        return values
    out = np.zeros((new_count, values.shape[1]), dtype=values.dtype)
    for old_index, new_index in enumerate(indice_map):
        if new_index >= 0:
            out[new_index] = values[old_index]
    return out


def decimate_groups(groups, target_reduction):
    """Run :func:`decimate_group` over ``[(material, v, uv, f), ...]``.

    The list order and material names are preserved exactly.
    """
    decimated = []
    all_stats = []
    for material, vertices, uvs, faces in groups:
        out_v, out_uv, out_f, stats, _indice_map = decimate_group(
            vertices, uvs, faces, target_reduction)
        stats["material"] = material
        all_stats.append(stats)
        decimated.append((material, out_v, out_uv, out_f))
    return decimated, all_stats
