"""Local multimodal distillation: ffmpeg + Pillow (+ faster-whisper via ``Transcriber``).

For every canonical media item this module produces what the vision stage
needs, and nothing more:

* **videos** (Reels, Stories, Highlights, TikTok posts/reposts): 4-6
  scene-change keyframes stitched into one labeled high-resolution contact
  sheet, plus a local transcript of the audio track;
* **carousels / photo-mode posts**: a labeled slide sheet (video slides are
  represented by a frame) plus transcripts of any video slides;
* **single photos**: a normalized JPEG sized for the vision model.

Everything is cached by input fingerprint, so re-running is cheap.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import re
import shutil
import statistics
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

from config.settings import MediaSettings, Settings
from intelligence.schemas import DerivedMedia, MediaItem, ProfileManifest, Transcript
from pipeline.transcriber import Transcriber
from storage.cache_manager import CacheManager, file_fingerprint, stable_hash

log = logging.getLogger(__name__)

PROCESSOR_VERSION = "2"


class MediaProcessingError(RuntimeError):
    pass


@dataclass
class ProbeInfo:
    duration: float | None
    width: int | None
    height: int | None
    has_video: bool
    has_audio: bool


# ---------------------------------------------------------------------------
# ffmpeg helpers
# ---------------------------------------------------------------------------


def _run(cmd: list[str], timeout: float = 600) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    if proc.returncode != 0:
        raise MediaProcessingError(f"{Path(cmd[0]).name} failed ({proc.returncode}): {proc.stderr[-400:]}")
    return proc


def check_binaries(media: MediaSettings) -> dict[str, bool]:
    return {
        name: shutil.which(binary) is not None
        for name, binary in (("ffmpeg", media.ffmpeg_bin), ("ffprobe", media.ffprobe_bin))
    }


def probe(path: Path, media: MediaSettings) -> ProbeInfo:
    proc = _run([media.ffprobe_bin, "-v", "error", "-print_format", "json", "-show_streams", "-show_format", str(path)])
    data = json.loads(proc.stdout or "{}")
    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    duration = data.get("format", {}).get("duration") or (video or {}).get("duration")
    return ProbeInfo(
        duration=float(duration) if duration not in (None, "N/A") else None,
        width=int(video["width"]) if video and video.get("width") else None,
        height=int(video["height"]) if video and video.get("height") else None,
        has_video=video is not None,
        has_audio=audio is not None,
    )


_PTS_RE = re.compile(r"pts_time:([0-9.]+)")
_SCORE_RE = re.compile(r"lavfi\.scene_score=([0-9.]+)")


def scene_scores(path: Path, media: MediaSettings) -> list[tuple[float, float]]:
    """(timestamp, scene-change score) for frames sampled at ``scene_analysis_fps``."""

    vf = f"fps={media.scene_analysis_fps},scale=320:-2,select='gte(scene,0)',metadata=print:file=-"
    proc = _run([media.ffmpeg_bin, "-hide_banner", "-nostats", "-i", str(path), "-vf", vf, "-an", "-f", "null", "-"])
    scores: list[tuple[float, float]] = []
    current_t: float | None = None
    for line in proc.stdout.splitlines():
        if (m := _PTS_RE.search(line)) is not None:
            current_t = float(m.group(1))
        elif (m := _SCORE_RE.search(line)) is not None and current_t is not None:
            scores.append((current_t, float(m.group(1))))
            current_t = None
    return scores


def choose_keyframe_times(
    scores: list[tuple[float, float]],
    duration: float,
    n_min: int,
    n_max: int,
    threshold: float,
    min_gap: float,
) -> list[float]:
    """Pick 4-6 representative timestamps: the hook, then the strongest scene cuts.

    Frames are taken shortly *after* a cut so transitions/motion blur are avoided.
    Remaining budget goes to the midpoints of the longest unsampled stretches, so
    videos with few detectable cuts (talking heads, slow pans) are still covered.
    """

    if duration <= 0:
        return [0.0]
    end = max(0.0, duration - 0.1)
    settle = min(0.4, min_gap / 2)
    hook = min(0.5, duration * 0.05)
    picked: list[float] = [hook]
    gap = min(min_gap, duration / (n_max + 1))

    def far_enough(t: float) -> bool:
        return all(abs(t - p) >= gap for p in picked)

    for t, score in sorted(scores, key=lambda ts: ts[1], reverse=True):
        if score < threshold or len(picked) >= n_max:
            break
        candidate = min(end, t + settle)
        if far_enough(candidate):
            picked.append(candidate)

    # Fill the largest temporal gaps: always up to n_min, and up to n_max while a
    # stretch longer than ~1.5 "fair shares" of the video remains unsampled.
    fair_share = duration / n_max
    while len(picked) < n_max:
        points = sorted([0.0, *picked, end])
        left, right = max(zip(points, points[1:], strict=False), key=lambda ab: ab[1] - ab[0])
        if right - left < 2 * gap or (len(picked) >= n_min and right - left < 1.5 * fair_share):
            break
        picked.append((left + right) / 2)
    return sorted(round(t, 2) for t in picked)[:n_max]


def extract_frame(path: Path, t: float, out: Path, media: MediaSettings) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            media.ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{t:.3f}",
            "-i",
            str(path),
            "-frames:v",
            "1",
            "-q:v",
            "2",
            str(out),
        ]
    )
    if not out.exists():  # seeking past the last decodable frame
        _run(
            [
                media.ffmpeg_bin,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-sseof",
                "-0.5",
                "-i",
                str(path),
                "-frames:v",
                "1",
                "-q:v",
                "2",
                str(out),
            ]
        )
    return out


def extract_audio(path: Path, out: Path, media: MediaSettings) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            media.ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(out),
        ]
    )
    return out


def extract_keyframes(
    path: Path, out_dir: Path, media: MediaSettings, info: ProbeInfo | None = None
) -> list[tuple[float, Path]]:
    info = info or probe(path, media)
    duration = info.duration or 0.0
    try:
        scores = scene_scores(path, media) if duration > 1.0 else []
    except MediaProcessingError as exc:
        log.debug("Scene analysis failed for %s: %s", path, exc)
        scores = []
    times = choose_keyframe_times(
        scores, duration, media.keyframes_min, media.keyframes_max, media.scene_threshold, media.min_keyframe_gap_s
    )
    frames: list[tuple[float, Path]] = []
    for idx, t in enumerate(times):
        frame = extract_frame(path, t, out_dir / f"frame_{idx:02d}.jpg", media)
        if frame.exists() and frame.stat().st_size > 0:
            frames.append((t, frame))
    if not frames:
        raise MediaProcessingError(f"No frames could be extracted from {path.name}")
    return frames


# ---------------------------------------------------------------------------
# Pillow helpers
# ---------------------------------------------------------------------------


def _font(size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    for name in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf", "Arial.ttf", "Helvetica.ttc"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def _fit_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_w: int) -> str:
    if draw.textlength(text, font=font) <= max_w:
        return text
    while text and draw.textlength(text + "…", font=font) > max_w:
        text = text[:-1]
    return text + "…"


def save_jpeg(img: Image.Image, out: Path, quality: int, max_bytes: int) -> Path:
    """Save as JPEG, stepping quality (then size) down until under ``max_bytes``."""

    out.parent.mkdir(parents=True, exist_ok=True)
    img = img.convert("RGB")
    q = quality
    while True:
        img.save(out, "JPEG", quality=q, optimize=True, progressive=True)
        if out.stat().st_size <= max_bytes:
            return out
        if q > 60:
            q -= 8
        else:
            img = img.resize((max(1, int(img.width * 0.85)), max(1, int(img.height * 0.85))), Image.LANCZOS)


def normalize_image(src: Path, out: Path, media: MediaSettings) -> Path:
    with Image.open(src) as im:
        img = ImageOps.exif_transpose(im).convert("RGB")
    img.thumbnail((media.image_max_edge, media.image_max_edge), Image.LANCZOS)
    return save_jpeg(img, out, media.jpeg_quality, media.max_image_bytes)


def build_contact_sheet(
    frames: list[Path],
    labels: list[str],
    header: str,
    out: Path,
    media: MediaSettings,
) -> Path:
    """Grid of frames with a header bar and a caption bar under each tile.

    Tiles are kept as large as the sheet's long-edge budget allows so burned-in
    captions and on-screen text stay legible to the vision model.
    """

    if not frames:
        raise MediaProcessingError("contact sheet needs at least one frame")
    images = []
    for f in frames:
        with Image.open(f) as im:
            images.append(ImageOps.exif_transpose(im).convert("RGB"))
    n = len(images)
    aspect = statistics.median(img.height / img.width for img in images)
    if aspect > 1.2:  # portrait (most short-form video)
        cols = min(n, 4 if n == 4 else media.contact_sheet_columns)
    else:
        cols = min(n, 2 if n <= 4 else media.contact_sheet_columns)
    rows = math.ceil(n / cols)

    max_edge = media.contact_sheet_max_edge
    pad = max(6, max_edge // 300)
    # Solve for tile width so the sheet's long edge fits the budget.
    label_ratio, header_ratio = 0.06, 0.06
    width_units = cols  # sheet width ~= cols * tile_w
    height_units = rows * (aspect + label_ratio) + header_ratio * cols
    tile_w = int((max_edge - pad * (max(cols, rows) + 1)) / max(width_units, height_units))
    tile_h = int(tile_w * aspect)
    label_h = max(26, int(tile_w * label_ratio))
    header_h = max(30, int(tile_w * cols * header_ratio * 0.6))
    sheet_w = cols * tile_w + pad * (cols + 1)
    sheet_h = header_h + rows * (tile_h + label_h) + pad * (rows + 1)

    sheet = Image.new("RGB", (sheet_w, sheet_h), (18, 18, 18))
    draw = ImageDraw.Draw(sheet)
    header_font = _font(max(14, int(header_h * 0.42)))
    label_font = _font(max(14, int(label_h * 0.6)))
    draw.text(
        (pad, header_h // 2),
        _fit_text(draw, header, header_font, sheet_w - 2 * pad),
        fill=(240, 240, 240),
        font=header_font,
        anchor="lm",
    )

    for idx, img in enumerate(images):
        r, c = divmod(idx, cols)
        x = pad + c * (tile_w + pad)
        y = header_h + pad + r * (tile_h + label_h + pad)
        tile = ImageOps.contain(img, (tile_w, tile_h), Image.LANCZOS)
        sheet.paste(tile, (x + (tile_w - tile.width) // 2, y + (tile_h - tile.height) // 2))
        draw.rectangle([x, y + tile_h, x + tile_w, y + tile_h + label_h], fill=(40, 40, 40))
        label = labels[idx] if idx < len(labels) else f"{idx + 1}/{n}"
        draw.text(
            (x + 8, y + tile_h + label_h // 2),
            _fit_text(draw, label, label_font, tile_w - 16),
            fill=(255, 214, 102),
            font=label_font,
            anchor="lm",
        )
    return save_jpeg(sheet, out, media.jpeg_quality, media.max_image_bytes)


def _fmt_ts(t: float) -> str:
    minutes, seconds = divmod(t, 60)
    return f"{int(minutes):02d}:{seconds:04.1f}"


# ---------------------------------------------------------------------------
# Per-item distillation
# ---------------------------------------------------------------------------


class MediaProcessor:
    """Turns downloaded assets into contact sheets, normalized images and transcripts."""

    def __init__(self, settings: Settings, cache: CacheManager, transcriber: Transcriber | None = None) -> None:
        self.settings = settings
        self.media = settings.media
        self.cache = cache
        self.transcriber = (
            transcriber
            if transcriber is not None
            else (Transcriber(settings.whisper) if settings.whisper.enabled else None)
        )

    def _transcribe_video(self, video: Path, workdir: Path, info: ProbeInfo) -> Transcript | None:
        if self.transcriber is None or not info.has_audio:
            return None
        key = stable_hash(file_fingerprint(video), self.transcriber.model_name, self.settings.whisper.language)
        cached = self.cache.get_model("transcripts", key, Transcript)
        if cached is not None:
            return cached
        wav = extract_audio(video, workdir / f"{video.stem}.wav", self.media)
        try:
            transcript = self.transcriber.transcribe(wav)
        finally:
            wav.unlink(missing_ok=True)
        self.cache.set("transcripts", key, transcript)
        return transcript

    def _header(self, item: MediaItem, extra: str = "") -> str:
        when = item.taken_at.strftime("%Y-%m-%d") if item.taken_at else "undated"
        parts = [item.item_id, item.surface.label, when]
        if item.highlight_title:
            parts.append(f"Highlight: {item.highlight_title}")
        if extra:
            parts.append(extra)
        return "  |  ".join(parts)

    def process_item(self, item: MediaItem, out_root: Path, namespace: str) -> DerivedMedia:
        """Synchronous: run in a worker thread."""

        local = [a for a in item.assets if a.local_path and Path(a.local_path).exists()]
        workdir = out_root / stable_hash(item.item_id)
        key = stable_hash(
            PROCESSOR_VERSION,
            item.item_id,
            [file_fingerprint(a.local_path) for a in local],
            self.media.model_dump(mode="json"),
            self.settings.whisper.model_size,
            self.transcriber is not None,
        )
        cached = self.cache.get_model(namespace, key, DerivedMedia)
        if cached is not None and all(
            Path(p).exists() for p in ([cached.contact_sheet] if cached.contact_sheet else []) + cached.images
        ):
            return cached

        derived = DerivedMedia()
        videos = sorted((a for a in local if a.kind == "video"), key=lambda a: a.index)
        stills = sorted(
            (a for a in local if a.kind == "image" and a.role in ("primary", "slide")), key=lambda a: a.index
        )
        covers = {a.index: a for a in local if a.kind == "image" and a.role == "cover"}

        try:
            is_multi = len({a.index for a in videos + stills}) > 1
            if videos and not is_multi:
                self._process_video(item, Path(videos[0].local_path), workdir, derived)  # type: ignore[arg-type]
            elif is_multi:
                self._process_slides(item, videos, stills, covers, workdir, derived)
            elif stills:
                derived.images = [
                    str(normalize_image(Path(stills[0].local_path), workdir / "image_00.jpg", self.media))
                ]  # type: ignore[arg-type]
            elif covers:
                cover = next(iter(covers.values()))
                derived.images = [str(normalize_image(Path(cover.local_path), workdir / "cover_00.jpg", self.media))]  # type: ignore[arg-type]
                derived.errors.append("primary media unavailable; analyzed cover image only")
            else:
                derived.errors.append("no downloaded media")
        except (MediaProcessingError, OSError, subprocess.TimeoutExpired) as exc:
            derived.errors.append(str(exc)[:300])
            log.warning("Processing %s failed: %s", item.item_id, exc)
            if not derived.vision_images and covers:
                cover = next(iter(covers.values()))
                try:
                    derived.images = [
                        str(normalize_image(Path(cover.local_path), workdir / "cover_00.jpg", self.media))
                    ]  # type: ignore[arg-type]
                except OSError:
                    pass
        self.cache.set(namespace, key, derived)
        return derived

    def _process_video(self, item: MediaItem, video: Path, workdir: Path, derived: DerivedMedia) -> None:
        info = probe(video, self.media)
        derived.duration_s = info.duration
        derived.has_audio = info.has_audio
        if not info.has_video:
            raise MediaProcessingError("file has no video stream")
        frames = extract_keyframes(video, workdir / "frames", self.media, info)
        n = len(frames)
        labels = [f"{i + 1}/{n}  @ {_fmt_ts(t)}" for i, (t, _) in enumerate(frames)]
        duration = f"{info.duration:.1f}s" if info.duration else "?s"
        sheet = build_contact_sheet(
            [p for _, p in frames], labels, self._header(item, duration), workdir / "contact_sheet.jpg", self.media
        )
        derived.contact_sheet = str(sheet)
        derived.keyframe_times = [t for t, _ in frames]
        try:
            derived.transcript = self._transcribe_video(video, workdir, info)
        except Exception as exc:  # transcription must never sink the visual analysis
            derived.errors.append(f"transcription failed: {exc}"[:300])
            log.warning("Transcription failed for %s: %s", item.item_id, exc)

    def _process_slides(self, item, videos, stills, covers, workdir, derived) -> None:  # type: ignore[no-untyped-def]
        slots: dict[int, tuple[Path, bool]] = {}
        transcripts: list[str] = []
        for asset in stills:
            slots[asset.index] = (Path(asset.local_path), False)
        for asset in videos:
            video = Path(asset.local_path)
            info = probe(video, self.media)
            if asset.index in covers:
                frame = Path(covers[asset.index].local_path)
            else:
                mid = (info.duration or 0) / 2
                frame = extract_frame(video, mid, workdir / f"slide_{asset.index:02d}.jpg", self.media)
            slots[asset.index] = (frame, True)
            try:
                tr = self._transcribe_video(video, workdir, info)
                if tr and tr.text:
                    transcripts.append(f"[slide {asset.index + 1}] {tr.text}")
            except Exception as exc:
                derived.errors.append(f"transcription failed (slide {asset.index + 1}): {exc}"[:300])
        ordered = [slots[i] for i in sorted(slots)][: self.media.carousel_sheet_max_slides]
        total = len(slots)
        labels = [
            f"slide {i + 1}/{total}" + (" (video)" if is_video else "") for i, (_, is_video) in enumerate(ordered)
        ]
        extra = f"{total} slides" + (f" (sheet shows first {len(ordered)})" if total > len(ordered) else "")
        sheet = build_contact_sheet(
            [p for p, _ in ordered], labels, self._header(item, extra), workdir / "slide_sheet.jpg", self.media
        )
        derived.contact_sheet = str(sheet)
        if transcripts:
            derived.transcript = Transcript(text="\n".join(transcripts))


async def distill_manifest(
    manifest: ProfileManifest,
    settings: Settings,
    cache: CacheManager,
    progress: Callable[[int, int, str], None] | None = None,
    transcriber: Transcriber | None = None,
) -> ProfileManifest:
    """Process every item concurrently (ffmpeg in threads; whisper serialized)."""

    processor = MediaProcessor(settings, cache, transcriber)
    out_root = settings.derived_dir(manifest.run_key)
    out_root.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(max(1, settings.media.workers))
    done = 0
    total = len(manifest.items)

    async def _one(item: MediaItem) -> None:
        nonlocal done
        async with sem:
            item.derived = await asyncio.to_thread(
                processor.process_item, item, out_root, f"{manifest.run_key}:derived"
            )
            if item.derived.duration_s and not item.duration_s:
                item.duration_s = item.derived.duration_s
        done += 1
        if progress:
            progress(done, total, item.item_id)

    await asyncio.gather(*(_one(item) for item in manifest.items))
    return manifest
