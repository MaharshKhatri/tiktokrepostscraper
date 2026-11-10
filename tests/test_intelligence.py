from __future__ import annotations

import asyncio
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

from PIL import Image

from config.prompts import ALL_SECTIONS
from intelligence.report_synthesizer import ReportSynthesizer, compute_stats
from intelligence.schemas import (
    BatchObservation,
    BrandMention,
    CommentSample,
    DerivedMedia,
    ItemObservation,
    MediaItem,
    MediaKind,
    Metrics,
    PlatformProfile,
    ProfileManifest,
    Surface,
    SurfaceDigest,
    Transcript,
    VisualAesthetic,
)
from intelligence.vision_analyzer import ClaudeGateway, VisionAnalyzer, plan_batches, render_item_header
from tests.conftest import fake_client, fake_message

ITEM_RE = re.compile(r"^ITEM (\S+)$", re.M)


def _observation(item_id: str) -> ItemObservation:
    return ItemObservation(
        item_id=item_id,
        summary="Creator pours latte art in a sunny kitchen.",
        content_format="tutorial",
        content_pillars=["coffee"],
        setting="home kitchen",
        visual_aesthetic=VisualAesthetic(
            color_palette=["cream", "walnut"],
            lighting="window light",
            composition="overhead",
            editing_style="jump cuts",
            mood="cozy",
        ),
        on_screen_text=["3 tips"],
        voice_and_tone="warm",
        persona_signals=["friendly expert"],
        interest_signals=["coffee"],
        brands_and_products=[BrandMention(name="BrewCo", context="tagged", evidence="tag")],
        commercial_signals=[],
        production_quality="polished",
        brand_safety_flags=[],
        notable_quotes=["tip one"],
        confidence="high",
    )


def _manifest(tmp_path: Path, n_reels: int = 10) -> ProfileManifest:
    img = tmp_path / "sheet.jpg"
    Image.new("RGB", (800, 1200), "beige").save(img, "JPEG")
    now = datetime(2026, 10, 1, tzinfo=timezone.utc)
    items = []
    for i in range(n_reels):
        items.append(
            MediaItem(
                item_id=f"instagram:ig_reel:{i}",
                platform="instagram",
                surface=Surface.IG_REEL,
                native_id=str(i),
                media_kind=MediaKind.VIDEO,
                taken_at=now - timedelta(days=3 * i),
                caption=f"tip {i} #coffee",
                hashtags=["coffee"],
                metrics=Metrics(likes=1000 + i * 10, comments=20, views=20000 + i * 500),
                comments=[CommentSample(text="love this", likes=2)],
                derived=DerivedMedia(
                    contact_sheet=str(img),
                    keyframe_times=[0.5, 2.0, 4.0, 6.0],
                    transcript=Transcript(text="today three tips for latte art", language="en"),
                ),
            )
        )
    items.append(
        MediaItem(
            item_id="instagram:ig_tagged:t1",
            platform="instagram",
            surface=Surface.IG_TAGGED,
            native_id="t1",
            media_kind=MediaKind.IMAGE,
            author_handle="brewco",
            author_is_profile_owner=False,
            author_is_verified=True,
            taken_at=now,
            derived=DerivedMedia(images=[str(img)]),
        )
    )
    items.append(
        MediaItem(
            item_id="tiktok:tt_repost:r1",
            platform="tiktok",
            surface=Surface.TT_REPOST,
            native_id="r1",
            media_kind=MediaKind.VIDEO,
            author_handle="coffeescience",
            author_is_profile_owner=False,
            taken_at=now,
        )
    )
    profiles = [
        PlatformProfile(platform="instagram", username="creator", followers=50000, posts_count=400),
        PlatformProfile(platform="tiktok", username="creator", followers=120000, total_likes=3_000_000),
    ]
    return ProfileManifest(run_key="creator", profiles=profiles, items=items)


def _echo_parse(kwargs: dict) -> object:
    """Fake vision/digest responses derived from the request itself."""

    fmt = kwargs["output_format"]
    content = kwargs["messages"][0]["content"]
    if fmt is BatchObservation:
        ids = [m for block in content if block["type"] == "text" for m in ITEM_RE.findall(block["text"])]
        return fake_message(
            parsed=BatchObservation(items=[_observation(i) for i in ids], batch_patterns=["overhead shots"])
        )
    assert fmt is SurfaceDigest
    surface = re.search(r"Surface: (.+)", content[0]["text"]).group(1)  # type: ignore[union-attr]
    return fake_message(
        parsed=SurfaceDigest(
            surface=surface,
            item_count=1,
            overview="Cozy coffee tutorials.",
            recurring_themes=["latte art"],
            format_patterns=["overhead tutorial"],
            aesthetic_patterns=["warm neutrals"],
            voice_and_persona_patterns=["warm"],
            interest_and_lifestyle_patterns=["coffee"],
            brand_and_commercial_patterns=["BrewCo tags"],
            audience_response_patterns=["saves"],
            brand_safety_notes=[],
            standout_items=[],
            notable_quotes=[],
        )
    )


def test_plan_batches_respects_limits(tmp_path: Path, settings) -> None:  # type: ignore[no-untyped-def]
    manifest = _manifest(tmp_path, n_reels=10)
    settings.anthropic.vision_items_per_batch = 4
    settings.anthropic.vision_max_images_per_request = 3
    batches = plan_batches(manifest.items, settings)
    reel_batches = [b for b in batches if b.surface == Surface.IG_REEL]
    assert all(b.image_count <= 3 for b in batches)
    assert sum(len(b.items) for b in reel_batches) == 10
    assert [b.batch_no for b in reel_batches] == list(range(1, len(reel_batches) + 1))
    assert len({b.key for b in batches}) == len(batches)
    # newest first within a surface
    first = reel_batches[0].items
    assert first[0].taken_at >= first[-1].taken_at  # type: ignore[operator]


def test_item_header_contents(tmp_path: Path, settings) -> None:  # type: ignore[no-untyped-def]
    manifest = _manifest(tmp_path, n_reels=1)
    header = render_item_header(manifest.items[0], settings)
    assert header.startswith("ITEM instagram:ig_reel:0")
    assert "speech transcript (automatic, language en)" in header
    assert "contact sheet of 4 keyframes" in header
    tagged = render_item_header(manifest.items[1], settings)
    assert "(another account)" in tagged
    no_media = render_item_header(manifest.items[2], settings)
    assert "media unavailable" in no_media


def test_vision_analyzer_caches_and_splits_on_refusal(tmp_path: Path, settings, cache) -> None:  # type: ignore[no-untyped-def]
    manifest = _manifest(tmp_path, n_reels=4)
    settings.anthropic.vision_items_per_batch = 4

    def handler(kwargs: dict) -> object:
        content = kwargs["messages"][0]["content"]
        ids = [m for block in content if block["type"] == "text" for m in ITEM_RE.findall(block["text"])]
        if "instagram:ig_reel:2" in ids:
            msg = fake_message(stop_reason="refusal")
            return msg
        return _echo_parse(kwargs)

    client = fake_client(handler)
    gateway = ClaudeGateway(settings, client=client)  # type: ignore[arg-type]
    outcome = asyncio.run(VisionAnalyzer(settings, cache, gateway).analyze(manifest))
    assert outcome.refused_items == ["instagram:ig_reel:2"]
    assert set(outcome.observations) == {i.item_id for i in manifest.items} - {"instagram:ig_reel:2"}
    first_calls = len(client.beta.messages.parse_calls)
    call = client.beta.messages.parse_calls[0]
    assert call["fallbacks"] == "default" and call["betas"] == [settings.anthropic.fallback_beta]
    assert call["output_config"] == {"effort": settings.anthropic.vision_effort}
    assert any(b["type"] == "image" for b in call["messages"][0]["content"])
    assert (settings.run_dir("creator") / "observations.jsonl").exists()

    # Second run: everything except the declined item is served from the cache.
    outcome2 = asyncio.run(VisionAnalyzer(settings, cache, gateway).analyze(manifest))
    assert set(outcome2.observations) == set(outcome.observations)
    assert len(client.beta.messages.parse_calls) - first_calls <= 1


def test_vision_analyzer_repairs_mangled_ids(tmp_path: Path, settings, cache) -> None:  # type: ignore[no-untyped-def]
    manifest = _manifest(tmp_path, n_reels=2)
    manifest.items = manifest.items[:2]

    def handler(kwargs: dict) -> object:
        content = kwargs["messages"][0]["content"]
        ids = [m for block in content if block["type"] == "text" for m in ITEM_RE.findall(block["text"])]
        return fake_message(parsed=BatchObservation(items=[_observation(f"{i}-typo") for i in ids], batch_patterns=[]))

    gateway = ClaudeGateway(settings, client=fake_client(handler))  # type: ignore[arg-type]
    outcome = asyncio.run(VisionAnalyzer(settings, cache, gateway).analyze(manifest))
    assert set(outcome.observations) == {"instagram:ig_reel:0", "instagram:ig_reel:1"}


def test_compute_stats(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, n_reels=10)
    stats = compute_stats(manifest)
    reels = stats["surfaces"]["ig_reel"]
    assert reels["items"] == 10 and reels["median_views"] == 22250.0
    assert reels["posts_per_week"] > 2 and reels["top_hashtags"][0] == "#coffee (10)"
    assert reels["median_engagement_rate_pct"] is not None
    assert stats["surfaces"]["ig_tagged"]["top_source_accounts"] == ["@brewco (1, verified)"]
    assert stats["platforms"]["tiktok"]["followers"] == 120000


def test_full_synthesis_writes_report(tmp_path: Path, settings, cache) -> None:  # type: ignore[no-untyped-def]
    manifest = _manifest(tmp_path, n_reels=5)

    def stream_handler(kwargs: dict) -> object:
        request = kwargs["messages"][0]["content"][-1]["text"]
        title = re.search(r'"([^"]+)"', request).group(1)  # type: ignore[union-attr]
        return fake_message(text=f"Body for {title} [instagram:ig_reel:0].")

    client = fake_client(_echo_parse, stream_handler)
    gateway = ClaudeGateway(settings, client=client)  # type: ignore[arg-type]
    outcome = asyncio.run(VisionAnalyzer(settings, cache, gateway).analyze(manifest))
    report, path = asyncio.run(ReportSynthesizer(settings, cache, gateway).synthesize(manifest, outcome))

    assert path == settings.reports_dir / "creator_full_report.md"
    text = path.read_text()
    for spec in ALL_SECTIONS:
        assert f"## {spec.number}. {spec.title}" in text
        assert f"Body for {spec.title}" in text
    assert "### A. Data coverage" in text and "### D. Methodology & limitations" in text
    assert (settings.reports_dir / "creator_item_index.csv").exists()
    assert len(report.sections) == 10

    stream_calls = client.beta.messages.stream_calls
    assert len(stream_calls) == 10
    evidence_block = stream_calls[0]["messages"][0]["content"][0]
    assert evidence_block["cache_control"] == {"type": "ephemeral"}
    assert all(c["messages"][0]["content"][0]["text"] == evidence_block["text"] for c in stream_calls)
    assert "<surface_digests>" in evidence_block["text"] and "Cozy coffee tutorials." in evidence_block["text"]
    summary_call = stream_calls[-1]["messages"][0]["content"][1]["text"]
    assert "Executive Summary" in summary_call and "Body for Brand Partnerships" in summary_call

    # Re-running reuses cached sections (no new streaming calls).
    asyncio.run(ReportSynthesizer(settings, cache, gateway).synthesize(manifest, outcome))
    assert len(client.beta.messages.stream_calls) == 10
