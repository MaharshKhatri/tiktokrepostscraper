"""End-to-end scraper tests in a real Chromium, with the platforms simulated.

Playwright request routing answers every instagram.com / tiktok.com request from
local fixtures, so the real scraper code paths (response hooks, in-page fetch,
pagination, scrolling, tab switching) run without touching the live sites.
"""

from __future__ import annotations

import asyncio
import copy
import functools
import http.server
import json
import threading
import time
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

from intelligence.schemas import Surface
from scrapers.base_scraper import PrivateProfileError
from scrapers.instagram_scraper import InstagramScraper
from scrapers.tiktok_scraper import TikTokScraper
from tests.conftest import load_fixture

pytestmark = pytest.mark.browser


def _write_state(settings, platform: str) -> None:  # type: ignore[no-untyped-def]
    domain = ".instagram.com" if platform == "instagram" else ".tiktok.com"
    cookies = [
        {
            "name": "sessionid",
            "value": "fake",
            "domain": domain,
            "path": "/",
            "expires": time.time() + 3600,
            "httpOnly": True,
            "secure": True,
            "sameSite": "Lax",
        }
    ]
    if platform == "instagram":
        cookies.append(
            {
                "name": "ds_user_id",
                "value": "999",
                "domain": domain,
                "path": "/",
                "expires": time.time() + 3600,
                "httpOnly": False,
                "secure": True,
                "sameSite": "Lax",
            }
        )
    settings.session_file(platform).write_text(json.dumps({"cookies": cookies, "origins": []}))


async def _launch_or_skip(scraper) -> None:  # type: ignore[no-untyped-def]
    try:
        await scraper.start()
    except Exception as exc:  # no launchable Chromium in this environment
        pytest.skip(f"Chromium unavailable: {exc}")


def _json(route, payload, status: int = 200):  # type: ignore[no-untyped-def]
    return route.fulfill(status=status, content_type="application/json", body=json.dumps(payload))


# ---------------------------------------------------------------------------
# Instagram
# ---------------------------------------------------------------------------


def _ig_routes(private: bool = False):  # type: ignore[no-untyped-def]
    feed = load_fixture("ig_feed_page.json")["items"]
    story_items = load_fixture("ig_story_reel.json")["reels"]["555"]["items"]
    page1 = {"items": feed[:2], "more_available": True, "next_max_id": "p2"}
    page2 = {"items": feed[2:], "more_available": False, "next_max_id": None}
    extra_reel = copy.deepcopy(feed[2])
    extra_reel.update(pk="3401000000000000099", code="ReelOnly", comment_count=0)
    tagged = copy.deepcopy(feed[0])
    tagged.update(pk="7001", code="TAG1", user={"username": "brewco", "is_verified": True})
    highlight_a = [dict(copy.deepcopy(story_items[0]), pk=f"80{i}") for i in range(3)]
    highlight_b = [dict(copy.deepcopy(story_items[0]), pk="90")]
    profile = {
        "id": "555",
        "username": "creator",
        "full_name": "Cozy Creator",
        "is_private": private,
        "biography": "coffee",
        "edge_followed_by": {"count": 50000},
        "edge_owner_to_timeline_media": {"count": 3},
        "profile_pic_url_hd": "https://cdn.example/avatar.jpg",
    }
    calls: list[str] = []

    async def handler(route, request):  # type: ignore[no-untyped-def]
        url = urlparse(request.url)
        qs = parse_qs(url.query)
        path = url.path
        calls.append(f"{request.method} {path}")
        if path.startswith("/api/"):
            assert request.headers.get("x-ig-app-id"), "IG app id header missing"
        if path == "/creator/":
            return await route.fulfill(
                status=200, content_type="text/html", body="<html><body><main>profile</main></body></html>"
            )
        if path == "/api/v1/users/web_profile_info/":
            return await _json(route, {"data": {"user": profile}})
        if path == "/api/v1/feed/user/555/":
            return await _json(route, page2 if qs.get("max_id") == ["p2"] else page1)
        if path == "/api/v1/clips/user/":
            assert request.method == "POST" and "target_user_id=555" in (request.post_data or "")
            return await _json(
                route, {"items": [{"media": feed[2]}, {"media": extra_reel}], "paging_info": {"more_available": False}}
            )
        if path.startswith("/api/v1/media/") and path.endswith("/comments/"):
            return await _json(
                route,
                {
                    "comments": [
                        {
                            "text": "so cozy",
                            "created_at": 1730000100,
                            "comment_like_count": 4,
                            "user": {"pk": "1", "username": "fan"},
                        },
                        {
                            "text": "thank you!",
                            "created_at": 1730000200,
                            "comment_like_count": 1,
                            "user": {"pk": "555", "username": "creator"},
                        },
                    ]
                },
            )
        if path == "/api/v1/feed/reels_media/":
            ids = qs.get("reel_ids", [])
            reels = {}
            for rid in ids:
                if rid == "555":
                    reels[rid] = {"items": story_items}
                elif rid == "highlight:1":
                    reels[rid] = {"items": highlight_a}
                elif rid == "highlight:2":
                    reels[rid] = {"items": highlight_b}
            return await _json(route, {"reels": reels})
        if path == "/api/v1/highlights/555/highlights_tray/":
            return await _json(
                route, {"tray": [{"id": "highlight:1", "title": "Coffee"}, {"id": "highlight:2", "title": "Travel"}]}
            )
        if path == "/api/v1/usertags/555/feed/":
            return await _json(route, {"items": [tagged], "more_available": False})
        return await route.fulfill(status=404, body="not found")

    return handler, calls


def test_instagram_scraper_collects_every_surface(settings, cache) -> None:  # type: ignore[no-untyped-def]
    _write_state(settings, "instagram")
    settings.scrape.download_media = False

    async def run():  # type: ignore[no-untyped-def]
        scraper = InstagramScraper(settings, cache, "creator")
        await _launch_or_skip(scraper)
        handler, calls = _ig_routes()
        await scraper.context.route("https://www.instagram.com/**", handler)
        try:
            return await scraper.scrape("creator"), calls
        finally:
            await scraper.stop()

    result, calls = asyncio.run(run())
    counts = {s: len(result.by_surface(s)) for s in Surface}
    assert counts[Surface.IG_POST] == 2
    assert counts[Surface.IG_REEL] == 2  # one from the feed + one Reels-tab-only
    assert counts[Surface.IG_STORY] == 1
    assert counts[Surface.IG_HIGHLIGHT] == 4
    assert counts[Surface.IG_TAGGED] == 1
    assert result.profile_raw["_highlight_titles"] == ["Coffee", "Travel"]
    commented = next(i for i in result.items if i.native_id == "3401000000000000001")
    assert commented.extra["comments"] == [
        {"text": "so cozy", "created_at": 1730000100, "likes": 4, "by_owner": False},
        {"text": "thank you!", "created_at": 1730000200, "likes": 1, "by_owner": True},
    ]
    assert "username" not in json.dumps(commented.extra["comments"])
    assert sum(1 for c in calls if c.endswith("/api/v1/feed/user/555/")) == 2  # paginated
    assert (settings.raw_dir("creator", "instagram") / "scrape_result.json").exists()
    assert (settings.raw_dir("creator", "instagram") / "network_capture.jsonl").exists()

    # Resume: a second run reuses the checkpoints instead of re-paginating.
    async def rerun():  # type: ignore[no-untyped-def]
        scraper = InstagramScraper(settings, cache, "creator")
        await _launch_or_skip(scraper)
        handler, calls2 = _ig_routes()
        await scraper.context.route("https://www.instagram.com/**", handler)
        try:
            await scraper.scrape("creator")
        finally:
            await scraper.stop()
        return calls2

    calls2 = asyncio.run(rerun())
    assert not any("/feed/user/" in c or "/usertags/" in c for c in calls2)


def test_instagram_private_profile_is_refused(settings, cache) -> None:  # type: ignore[no-untyped-def]
    _write_state(settings, "instagram")

    async def run():  # type: ignore[no-untyped-def]
        scraper = InstagramScraper(settings, cache, "creator")
        await _launch_or_skip(scraper)
        handler, _ = _ig_routes(private=True)
        await scraper.context.route("https://www.instagram.com/**", handler)
        try:
            await scraper.scrape("creator")
        finally:
            await scraper.stop()

    with pytest.raises(PrivateProfileError):
        asyncio.run(run())


# ---------------------------------------------------------------------------
# TikTok
# ---------------------------------------------------------------------------

TT_PAGE = """<!doctype html><html><head>
<script id="__UNIVERSAL_DATA_FOR_REHYDRATION__" type="application/json">%s</script></head>
<body style="margin:0">
<div style="position:sticky;top:0;background:#fff">
  <p role="tab" data-e2e="videos-tab">Videos</p>
  <p role="tab" data-e2e="repost-tab">Reposts</p>
</div>
<div id="grid" style="height:3000px"></div>
<script>
  let state = {kind: 'post', cursor: 0, hasMore: true, loading: false};
  async function load() {
    if (state.loading || !state.hasMore) return;
    state.loading = true;
    const r = await fetch('/api/' + state.kind + '/item_list/?secUid=x&cursor=' + state.cursor);
    const data = await r.json();
    state.cursor = data.cursor; state.hasMore = data.hasMore; state.loading = false;
    document.getElementById('grid').style.height = (parseInt(document.getElementById('grid').style.height) + 2000) + 'px';
  }
  window.addEventListener('scroll', () => {
    if (window.innerHeight + window.scrollY > document.body.scrollHeight - 1200) load();
  });
  document.querySelector('[data-e2e="repost-tab"]').addEventListener('click', () => {
    state = {kind: 'repost', cursor: 0, hasMore: true, loading: false}; load();
  });
  load();
</script></body></html>"""


def _tt_routes(private: bool = False):  # type: ignore[no-untyped-def]
    items = load_fixture("tt_item_list.json")["itemList"]
    third = dict(copy.deepcopy(items[0]), id="7400000000000000003", desc="third #coffee")
    reposts = load_fixture("tt_repost_list.json")["itemList"]
    rehydration = {
        "__DEFAULT_SCOPE__": {
            "webapp.user-detail": {
                "statusCode": 0,
                "userInfo": {
                    "user": {
                        "id": "1",
                        "uniqueId": "creator",
                        "nickname": "Cozy",
                        "privateAccount": private,
                        "avatarLarger": "https://p16.example/a.jpg",
                    },
                    "stats": {"followerCount": 120000, "videoCount": 3},
                },
            }
        }
    }

    async def handler(route, request):  # type: ignore[no-untyped-def]
        url = urlparse(request.url)
        cursor = parse_qs(url.query).get("cursor", ["0"])[0]
        if url.path == "/@creator":
            return await route.fulfill(status=200, content_type="text/html", body=TT_PAGE % json.dumps(rehydration))
        if url.path == "/api/post/item_list/":
            if cursor == "0":
                return await _json(route, {"itemList": items, "hasMore": True, "cursor": "2"})
            return await _json(route, {"itemList": [third], "hasMore": False, "cursor": "3"})
        if url.path == "/api/repost/item_list/":
            return await _json(route, {"itemList": reposts, "hasMore": False, "cursor": "1"})
        return await route.fulfill(status=404, body="")

    return handler


def test_tiktok_scraper_videos_and_reposts(settings, cache) -> None:  # type: ignore[no-untyped-def]
    _write_state(settings, "tiktok")
    settings.scrape.download_media = False

    async def run():  # type: ignore[no-untyped-def]
        scraper = TikTokScraper(settings, cache, "creator")
        await _launch_or_skip(scraper)
        await scraper.context.route("https://www.tiktok.com/**", _tt_routes())
        try:
            return await scraper.scrape("creator")
        finally:
            await scraper.stop()

    result = asyncio.run(run())
    videos = result.by_surface(Surface.TT_VIDEO)
    reposts = result.by_surface(Surface.TT_REPOST)
    assert sorted(v.native_id for v in videos) == ["7400000000000000001", "7400000000000000002", "7400000000000000003"]
    assert [r.extra["author"] for r in reposts] == ["coffeescience"]
    assert result.profile_raw["stats"]["followerCount"] == 120000
    photo = next(v for v in videos if v.native_id == "7400000000000000002")
    assert photo.extra["url"].endswith("/photo/7400000000000000002")


def test_tiktok_private_profile_is_refused(settings, cache) -> None:  # type: ignore[no-untyped-def]
    _write_state(settings, "tiktok")

    async def run():  # type: ignore[no-untyped-def]
        scraper = TikTokScraper(settings, cache, "creator")
        await _launch_or_skip(scraper)
        await scraper.context.route("https://www.tiktok.com/**", _tt_routes(private=True))
        try:
            await scraper.scrape("creator")
        finally:
            await scraper.stop()

    with pytest.raises(PrivateProfileError):
        asyncio.run(run())


# ---------------------------------------------------------------------------
# Downloads (real local HTTP server)
# ---------------------------------------------------------------------------


def test_download_skips_existing_and_handles_404(settings, cache, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    served = tmp_path / "served"
    served.mkdir()
    (served / "clip.bin").write_bytes(b"x" * 4096)
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(served))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"

    async def run():  # type: ignore[no-untyped-def]
        scraper = InstagramScraper(settings, cache, "creator")
        try:
            await scraper.start(require_session=False)
        except Exception as exc:
            pytest.skip(f"Chromium unavailable: {exc}")
        try:
            dest = tmp_path / "out" / "clip.mp4"
            first = await scraper.download(f"{base}/clip.bin", dest)
            mtime = dest.stat().st_mtime_ns
            second = await scraper.download(f"{base}/clip.bin", dest)
            missing = await scraper.download(f"{base}/missing.bin", tmp_path / "out" / "missing.mp4")
            return first, second, mtime, dest, missing
        finally:
            await scraper.stop()
            server.shutdown()

    first, second, mtime, dest, missing = asyncio.run(run())
    assert first == dest and dest.read_bytes() == b"x" * 4096
    assert second == dest and dest.stat().st_mtime_ns == mtime
    assert missing is None
