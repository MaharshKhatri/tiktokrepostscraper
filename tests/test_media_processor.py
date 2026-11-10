from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
from PIL import Image

from intelligence.schemas import MediaAsset, MediaItem, MediaKind, ProfileManifest, Surface, Transcript
from pipeline.media_processor import (
    MediaProcessor,
    build_contact_sheet,
    choose_keyframe_times,
    distill_manifest,
    extract_keyframes,
    probe,
    save_jpeg,
)
from tests.conftest import make_test_video, needs_ffmpeg


class StubTranscriber:
    model_name = "stub"

    def __init__(self) -> None:
        self.calls = 0

    def transcribe(self, path):  # type: ignore[no-untyped-def]
        self.calls += 1
        assert Path(path).exists() and Path(path).suffix == ".wav"
        return Transcript(text="hey guys welcome back", language="en", segments=[(0.0, 1.5, "hey guys welcome back")])


def _item(item_id: str, surface: Surface, kind: MediaKind, assets: list[MediaAsset]) -> MediaItem:
    platform = "tiktok" if surface.value.startswith("tt_") else "instagram"
    return MediaItem(
        item_id=item_id,
        platform=platform,
        surface=surface,
        native_id=item_id.split(":")[-1],
        media_kind=kind,
        taken_at=datetime(2026, 5, 1, tzinfo=timezone.utc),
        assets=assets,
    )


def test_choose_keyframe_times_prefers_strong_cuts_and_fills() -> None:
    scores = [(0.25, 0.01), (1.0, 0.9), (1.25, 0.85), (3.0, 0.7), (4.5, 0.5), (5.0, 0.05)]
    times = choose_keyframe_times(scores, duration=6.0, n_min=4, n_max=6, threshold=0.28, min_gap=0.75)
    assert 4 <= len(times) <= 6
    assert times == sorted(times)
    assert times[0] <= 0.5  # hook frame
    assert any(1.0 < t < 1.6 for t in times)  # strongest cut (settled after the cut)
    assert all(b - a >= 0.5 for a, b in zip(times, times[1:], strict=False))
    # No cuts at all -> evenly spaced fill
    flat = choose_keyframe_times([], duration=20.0, n_min=4, n_max=6, threshold=0.28, min_gap=0.75)
    assert len(flat) >= 4
    assert choose_keyframe_times([], duration=0, n_min=4, n_max=6, threshold=0.3, min_gap=0.75) == [0.0]


def test_contact_sheet_respects_long_edge(tmp_path: Path, settings) -> None:  # type: ignore[no-untyped-def]
    frames = []
    for i, color in enumerate(["red", "green", "blue", "orange", "purple", "gray"]):
        p = tmp_path / f"f{i}.jpg"
        Image.new("RGB", (1080, 1920), color).save(p)
        frames.append(p)
    out = build_contact_sheet(
        frames, [f"{i + 1}/6" for i in range(6)], "header text", tmp_path / "sheet.jpg", settings.media
    )
    with Image.open(out) as im:
        assert max(im.size) <= settings.media.contact_sheet_max_edge
        assert im.height > im.width  # 3x2 grid of portrait frames
        assert im.width >= 1500  # tiles stay large enough to read overlays


def test_save_jpeg_enforces_byte_budget(tmp_path: Path) -> None:
    import random

    img = Image.new("RGB", (1500, 1500))
    img.putdata([(random.randrange(256), random.randrange(256), random.randrange(256)) for _ in range(1500 * 1500)])
    out = save_jpeg(img, tmp_path / "noisy.jpg", quality=95, max_bytes=400_000)
    assert out.stat().st_size <= 400_000


@needs_ffmpeg
def test_keyframes_from_scene_cuts(tmp_path: Path, settings) -> None:  # type: ignore[no-untyped-def]
    video = make_test_video(tmp_path / "v.mp4", seconds=6)
    info = probe(video, settings.media)
    assert info.has_video and info.has_audio and info.duration and 5.5 < info.duration < 6.5
    frames = extract_keyframes(video, tmp_path / "frames", settings.media, info)
    assert settings.media.keyframes_min <= len(frames) <= settings.media.keyframes_max
    assert all(p.exists() for _, p in frames)


@needs_ffmpeg
def test_process_items_video_carousel_photo(tmp_path: Path, settings, cache) -> None:  # type: ignore[no-untyped-def]
    video = make_test_video(tmp_path / "reel.mp4", seconds=6)
    silent = make_test_video(tmp_path / "slide.mp4", seconds=4, with_audio=False)
    photo = tmp_path / "photo.jpg"
    Image.new("RGB", (3000, 2000), "teal").save(photo)

    reel = _item(
        "instagram:ig_reel:1", Surface.IG_REEL, MediaKind.VIDEO, [MediaAsset(kind="video", local_path=str(video))]
    )
    carousel = _item(
        "instagram:ig_post:2",
        Surface.IG_POST,
        MediaKind.CAROUSEL,
        [
            MediaAsset(kind="image", local_path=str(photo), role="slide", index=0),
            MediaAsset(kind="video", local_path=str(silent), role="slide", index=1),
            MediaAsset(kind="video", local_path=str(video), role="slide", index=2),
        ],
    )
    single = _item(
        "instagram:ig_post:3", Surface.IG_POST, MediaKind.IMAGE, [MediaAsset(kind="image", local_path=str(photo))]
    )
    missing = _item(
        "tiktok:tt_video:4",
        Surface.TT_VIDEO,
        MediaKind.VIDEO,
        [MediaAsset(kind="video", local_path=str(tmp_path / "nope.mp4"))],
    )
    manifest = ProfileManifest(run_key="creator", items=[reel, carousel, single, missing])

    stub = StubTranscriber()
    seen = []
    await_result = distill_manifest(
        manifest, settings, cache, progress=lambda d, t, i: seen.append(i), transcriber=stub
    )  # type: ignore[arg-type]
    import asyncio

    asyncio.run(await_result)
    assert len(seen) == 4

    assert reel.derived.contact_sheet and Path(reel.derived.contact_sheet).exists()
    assert 4 <= len(reel.derived.keyframe_times) <= 6
    assert reel.derived.transcript and reel.derived.transcript.text.startswith("hey guys")
    assert reel.derived.has_audio is True

    assert carousel.derived.contact_sheet and carousel.derived.contact_sheet.endswith("slide_sheet.jpg")
    assert carousel.derived.transcript and "[slide 3]" in carousel.derived.transcript.text

    assert single.derived.images and max(Image.open(single.derived.images[0]).size) <= settings.media.image_max_edge
    assert missing.derived.errors == ["no downloaded media"]

    calls_before = stub.calls
    processor = MediaProcessor(settings, cache, stub)  # type: ignore[arg-type]
    again = processor.process_item(reel, settings.derived_dir("creator"), "creator:derived")
    assert again.contact_sheet == reel.derived.contact_sheet
    assert stub.calls == calls_before  # fully cached


@pytest.mark.whisper
def test_real_whisper_on_tone(tmp_path: Path, settings) -> None:  # type: ignore[no-untyped-def]
    pytest.importorskip("faster_whisper")
    from pipeline.media_processor import extract_audio
    from pipeline.transcriber import Transcriber

    settings.whisper.model_size = "tiny"
    video = make_test_video(tmp_path / "v.mp4", seconds=4)
    wav = extract_audio(video, tmp_path / "a.wav", settings.media)
    try:
        transcript = Transcriber(settings.whisper).transcribe(wav)
    except Exception as exc:  # model download blocked offline
        pytest.skip(f"whisper model unavailable: {exc}")
    assert isinstance(transcript.text, str)
