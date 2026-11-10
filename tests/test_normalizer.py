from __future__ import annotations

from intelligence.schemas import MediaKind, PlatformScrapeResult, RawItem, Surface
from pipeline.normalizer import build_manifest, load_manifest, parse_tags, save_manifest
from scrapers.instagram_scraper import classify_feed_surface, extract_assets, find_media_nodes, media_pk
from scrapers.tiktok_scraper import extract_tt_assets, tt_item_url
from tests.conftest import load_fixture


def _ig_result() -> PlatformScrapeResult:
    feed = load_fixture("ig_feed_page.json")
    stories = load_fixture("ig_story_reel.json")
    items = []
    for node in find_media_nodes(feed["items"]):
        raw = RawItem(
            surface=classify_feed_surface(node),
            native_id=media_pk(node),
            raw=node,
            assets=extract_assets(node),
            extra={"shortcode": node.get("code")},
        )
        if raw.native_id.endswith("1"):
            raw.extra["comments"] = [
                {"text": "need this mug!", "created_at": 1730000500, "likes": 3, "by_owner": False},
                {"text": "link in bio 💛", "created_at": 1730000600, "likes": 1, "by_owner": True},
            ]
        items.append(raw)
    for node in find_media_nodes(stories["reels"]["555"]["items"]):
        items.append(RawItem(surface=Surface.IG_STORY, native_id=media_pk(node), raw=node, assets=extract_assets(node)))
    tagged_node = dict(feed["items"][0], pk="777", code="TAG1", user={"username": "brewco", "is_verified": True})
    items.append(
        RawItem(
            surface=Surface.IG_TAGGED,
            native_id="777",
            raw=tagged_node,
            assets=extract_assets(tagged_node),
            extra={"author": "brewco", "author_is_verified": True},
        )
    )
    highlight_node = dict(stories["reels"]["555"]["items"][0], pk="888")
    items.append(
        RawItem(
            surface=Surface.IG_HIGHLIGHT,
            native_id="888",
            raw=highlight_node,
            assets=extract_assets(highlight_node),
            extra={"highlight_title": "Coffee"},
        )
    )
    profile = {
        "id": "555",
        "username": "creator",
        "full_name": "Cozy Creator",
        "biography": "coffee + slow living",
        "external_url": "https://linktr.ee/creator",
        "bio_links": [{"url": "https://shop.example"}],
        "edge_followed_by": {"count": 50000},
        "edge_follow": {"count": 300},
        "edge_owner_to_timeline_media": {"count": 420},
        "is_verified": True,
        "category_name": "Creator",
        "_highlight_titles": ["Coffee"],
    }
    return PlatformScrapeResult(platform="instagram", username="creator", profile_raw=profile, items=items)


def _tt_result() -> PlatformScrapeResult:
    videos = load_fixture("tt_item_list.json")["itemList"]
    reposts = load_fixture("tt_repost_list.json")["itemList"]
    items = [
        RawItem(
            surface=Surface.TT_VIDEO,
            native_id=n["id"],
            raw=n,
            assets=extract_tt_assets(n),
            extra={"url": tt_item_url(n)},
        )
        for n in videos
    ]
    items += [
        RawItem(
            surface=Surface.TT_REPOST,
            native_id=n["id"],
            raw=n,
            assets=extract_tt_assets(n),
            extra={"url": tt_item_url(n), "author": n["author"]["uniqueId"]},
        )
        for n in reposts
    ]
    profile = {
        "user": {
            "id": "1",
            "uniqueId": "creator",
            "nickname": "Cozy",
            "signature": "☕ daily",
            "verified": True,
            "privateAccount": False,
            "bioLink": {"link": "linktr.ee/creator"},
        },
        "stats": {"followerCount": 120000, "followingCount": 80, "heartCount": 3400000, "videoCount": 210},
    }
    return PlatformScrapeResult(platform="tiktok", username="creator", profile_raw=profile, items=items)


def test_parse_tags_handles_unicode_and_dedupes() -> None:
    tags, mentions = parse_tags("Hi #Coffee #coffee #café with @Brew.Co and @brew.co.")
    assert tags == ["coffee", "café"]
    assert mentions == ["brew.co"]


def test_instagram_asset_extraction() -> None:
    feed = load_fixture("ig_feed_page.json")["items"]
    photo, carousel, reel = feed
    assert [a.url for a in extract_assets(photo)] == ["https://cdn.example/p1_big.jpg"]
    car_assets = extract_assets(carousel)
    assert [(a.kind, a.role, a.index) for a in car_assets] == [
        ("image", "slide", 0),
        ("video", "slide", 1),
        ("image", "cover", 1),
        ("image", "slide", 2),
    ]
    reel_assets = extract_assets(reel)
    assert reel_assets[0].url == "https://cdn.example/r_hi.mp4" and reel_assets[0].duration_s == 31.2
    assert classify_feed_surface(reel) == Surface.IG_REEL
    assert classify_feed_surface(photo) == Surface.IG_POST
    assert classify_feed_surface(carousel) == Surface.IG_POST


def test_graphql_payload_media_nodes_do_not_split_carousels() -> None:
    nodes = find_media_nodes(load_fixture("ig_graphql_payload.json"))
    assert sorted(media_pk(n) for n in nodes) == ["9001", "9002"]


def test_tiktok_assets_and_urls() -> None:
    video, photo = load_fixture("tt_item_list.json")["itemList"]
    va = extract_tt_assets(video)
    assert va[0].kind == "video" and va[0].url.endswith("v1.mp4")
    assert va[1].role == "cover" and va[1].url.endswith("origin1.jpg")
    pa = extract_tt_assets(photo)
    assert [a.index for a in pa] == [0, 1] and all(a.kind == "image" for a in pa)
    assert tt_item_url(photo) == "https://www.tiktok.com/@creator/photo/7400000000000000002"


def test_build_manifest_normalizes_and_minimizes(settings) -> None:  # type: ignore[no-untyped-def]
    manifest = build_manifest("creator", [_ig_result(), _tt_result()])
    by_surface = manifest.surface_counts()
    assert by_surface[Surface.IG_POST] == 2 and by_surface[Surface.IG_REEL] == 1
    assert by_surface[Surface.IG_STORY] == 1 and by_surface[Surface.IG_TAGGED] == 1
    assert by_surface[Surface.TT_VIDEO] == 2 and by_surface[Surface.TT_REPOST] == 1

    ig_profile = manifest.profile("instagram")
    assert ig_profile and ig_profile.followers == 50000 and ig_profile.external_links[0] == "https://linktr.ee/creator"
    tt_profile = manifest.profile("tiktok")
    assert tt_profile and tt_profile.total_likes == 3400000

    photo = next(i for i in manifest.items if i.native_id == "3401000000000000001")
    assert photo.media_kind == MediaKind.IMAGE
    assert photo.location and photo.location.city == "Brooklyn, New York"
    assert "lat" not in photo.location.model_dump()  # coordinates are never kept
    assert photo.is_paid_partnership is True
    assert "brewco" in photo.mentions and "slowliving" in photo.hashtags
    assert [c.by_profile_owner for c in photo.comments] == [False, True]
    assert "user" not in photo.comments[0].model_dump()  # no commenter identity

    carousel = next(i for i in manifest.items if i.native_id == "3401000000000000002")
    assert carousel.media_kind == MediaKind.CAROUSEL
    reel = next(i for i in manifest.items if i.surface == Surface.IG_REEL)
    assert reel.audio and reel.audio.artist == "JVKE" and reel.metrics.views == 98000
    assert reel.url == "https://www.instagram.com/reel/C3reelCCC/"

    story = manifest.items_for(Surface.IG_STORY)[0]
    assert any(s.startswith("poll:") for s in story.story_stickers)
    assert "morningroutine" in story.hashtags and "brewco" in story.mentions
    assert story.audio and story.audio.title == "Sunflower"

    tagged = manifest.items_for(Surface.IG_TAGGED)[0]
    assert tagged.author_is_profile_owner is False and tagged.author_handle == "brewco"
    highlight = manifest.items_for(Surface.IG_HIGHLIGHT)[0]
    assert highlight.highlight_title == "Coffee"

    tt_video = next(i for i in manifest.items if i.native_id == "7400000000000000001")
    assert tt_video.metrics.views == 250000 and tt_video.metrics.saves == 900
    assert tt_video.audio and tt_video.audio.is_original is True
    assert "text sticker: day 1 of learning" in tt_video.story_stickers
    tt_photo = next(i for i in manifest.items if i.native_id == "7400000000000000002")
    assert tt_photo.media_kind == MediaKind.CAROUSEL and tt_photo.product_type == "photo"
    repost = manifest.items_for(Surface.TT_REPOST)[0]
    assert repost.author_handle == "coffeescience" and not repost.author_is_profile_owner

    save_manifest(manifest, settings)
    again = load_manifest(settings, "creator")
    assert again is not None and len(again.items) == len(manifest.items)
