"""Texture transcoding driven by one shared resource manifest.

PNG/TGA sources are converted to WebP or KTX2 *before* any MTL/glTF
reference is rewritten. Every conversion lands in a staging directory and
the manifest is the single source of truth used by both the OBJ and GLB
writers, so a mid-run failure leaves the originals untouched and produces
no half-rewritten model (no dangling ``map_*`` entries).
"""

import gc
import json
import os

from PIL import Image

from . import ktx2_writer

TRANSCODABLE_EXTS = {".png", ".tga"}
WEB_P = "webp"
KTX2 = "ktx2"
KEEP_ORIGINAL = "none"
TARGET_FORMATS = (WEB_P, KTX2, KEEP_ORIGINAL)


class TextureManifest:
    """Collects texture jobs, transcodes them, and records path rewrites."""

    def __init__(self, staging_dir, target_format=WEB_P, webp_quality=90,
                 external_ktx2_encoder=None):
        if target_format not in TARGET_FORMATS:
            raise ValueError(f"Unsupported texture target format: {target_format}")
        self.staging_dir = staging_dir
        self.texture_dir = os.path.join(staging_dir, "textures")
        os.makedirs(self.texture_dir, exist_ok=True)
        self.target_format = target_format
        self.webp_quality = int(webp_quality)
        self.external_ktx2_encoder = external_ktx2_encoder
        # source absolute path -> manifest record
        self._records_by_source = {}
        self._used_names = set()

    def _unique_output_name(self, source_path):
        stem = os.path.splitext(os.path.basename(source_path))[0]
        candidate = f"{stem}.{self.target_format}"
        counter = 1
        while candidate.lower() in self._used_names:
            candidate = f"{stem}_{counter}.{self.target_format}"
            counter += 1
        self._used_names.add(candidate.lower())
        return candidate

    def add(self, source_path):
        """Register a texture source; returns its record (creating if needed).

        Non-transcodable files (e.g. JPEG when targeting WebP) are recorded as
        passthrough and copied byte-for-byte so references stay valid.
        """
        source_abs = os.path.normpath(os.path.abspath(source_path))
        if source_abs in self._records_by_source:
            return self._records_by_source[source_abs]

        ext = os.path.splitext(source_abs)[1].lower()
        # "none" means keep every source texture byte-for-byte.
        record = {
            "source": source_abs,
            "source_ext": ext,
            "transcoded": ext in TRANSCODABLE_EXTS and self.target_format != KEEP_ORIGINAL,
            "output_rel": None,
            "output_abs": None,
            "engine": "passthrough",
            "width": None,
            "height": None,
        }
        if record["transcoded"]:
            record["output_rel"] = os.path.join("textures", self._unique_output_name(source_abs))
        else:
            # Keep the original file name for passthrough assets.
            passthrough_name = os.path.basename(source_abs)
            if passthrough_name.lower() in self._used_names:
                stem, old_ext = os.path.splitext(passthrough_name)
                counter = 1
                while f"{stem}_{counter}{old_ext}".lower() in self._used_names:
                    counter += 1
                passthrough_name = f"{stem}_{counter}{old_ext}"
            self._used_names.add(passthrough_name.lower())
            record["output_rel"] = os.path.join("textures", passthrough_name)
        record["output_abs"] = os.path.join(self.staging_dir, record["output_rel"])
        self._records_by_source[source_abs] = record
        return record

    def records(self):
        return list(self._records_by_source.values())

    def rewrite_for_source(self, source_path):
        """Return the manifest output relative path for a registered source."""
        record = self._records_by_source.get(
            os.path.normpath(os.path.abspath(source_path)))
        if record is None:
            return None
        return record["output_rel"]

    def prepare_all(self, progress_cb=None):
        """Transcode/copy every registered texture into staging.

        Raises on the first failure; nothing is committed to the final output
        directory and the caller discards the whole staging tree.
        """
        records = self.records()
        total = max(1, len(records))
        for index, record in enumerate(records, start=1):
            if progress_cb:
                progress_cb(index, total, os.path.basename(record["source"]))
            os.makedirs(os.path.dirname(record["output_abs"]), exist_ok=True)
            if not record["transcoded"]:
                _copy_bytes(record["source"], record["output_abs"])
                with Image.open(record["source"]) as probe:
                    record["width"], record["height"] = probe.size
                continue

            image = Image.open(record["source"])
            try:
                image.load()
                record["width"], record["height"] = image.size
                if self.target_format == WEB_P:
                    self._write_webp(image, record)
                else:
                    _, engine = ktx2_writer.encode_ktx2(
                        image,
                        record["output_abs"],
                        srgb=True,
                        external_encoder=self.external_ktx2_encoder,
                    )
                    record["engine"] = engine
            finally:
                # Explicit image release: 4K RGBA textures hold large buffers.
                image.close()
                del image
                gc.collect()

        # Persist the manifest beside the staged model so a crashed process
        # never leaves references that disagree with the files on disk.
        manifest_path = os.path.join(self.staging_dir, "texture_manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump({"target_format": self.target_format,
                       "records": records}, f, indent=2)
        return records

    def _write_webp(self, image, record):
        if image.mode not in ("RGB", "RGBA"):
            image = image.convert("RGBA" if "A" in image.getbands() else "RGB")
        tmp_path = record["output_abs"] + ".tmp"
        image.save(
            tmp_path,
            format="WEBP",
            quality=self.webp_quality,
            method=6,
            alpha_quality=100,
        )
        os.replace(tmp_path, record["output_abs"])
        record["engine"] = "pillow-webp"


def _copy_bytes(source, dest):
    with open(source, "rb") as src, open(dest + ".tmp", "wb") as dst:
        while True:
            chunk = src.read(1 << 20)
            if not chunk:
                break
            dst.write(chunk)
    os.replace(dest + ".tmp", dest)


def transcode_bytes(source_bytes, target_format, webp_quality=90,
                    external_ktx2_encoder=None):
    """Transcode an in-memory encoded image (used for embedded GLB images).

    Returns ``(out_bytes, mime, engine, width, height)``. JPEG sources are
    returned untouched when no compression gain is available, mirroring the
    file-manifest passthrough policy.
    """
    import io

    image = Image.open(io.BytesIO(source_bytes))
    image.load()
    width, height = image.size
    try:
        if target_format == KEEP_ORIGINAL:
            return source_bytes, "application/octet-stream", "passthrough", width, height
        if target_format == WEB_P:
            if image.format == "WEBP":
                return source_bytes, "image/webp", "passthrough", width, height
            if image.mode not in ("RGB", "RGBA"):
                image = image.convert("RGBA" if "A" in image.getbands() else "RGB")
            buffer = io.BytesIO()
            image.save(buffer, format="WEBP", quality=int(webp_quality),
                       method=6, alpha_quality=100)
            return buffer.getvalue(), "image/webp", "pillow-webp", width, height

        if target_format == KTX2:
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".ktx2", delete=False) as tmp_out:
                tmp_path = tmp_out.name
            try:
                _, engine = ktx2_writer.encode_ktx2(
                    image, tmp_path, srgb=True,
                    external_encoder=external_ktx2_encoder)
                with open(tmp_path, "rb") as f:
                    out_bytes = f.read()
            finally:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            return out_bytes, "image/ktx2", engine, width, height
    finally:
        image.close()
        del image
        gc.collect()

    raise ValueError(f"unsupported texture target: {target_format}")
