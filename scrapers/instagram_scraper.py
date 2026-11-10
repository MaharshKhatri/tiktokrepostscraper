"""Instagram profile ingestion.

Surfaces collected for ``instagram.com/{username}/``:

* feed posts (photos and multi-slide carousels) and Reels, with captions,
  timestamps, location labels, metrics and a sample of comments;
* every active Story slide;
* every Story Highlight tray and every slide inside each Highlight;
* every post in the ``/{username}/tagged/`` tab.

Strategy: the page is opened in the operator's authenticated browser and the
same internal ``/api/v1`` endpoints the web app uses are called from inside
the page (``fetch`` with browser cookies + CSRF). Every JSON response the page
receives is also captured, so if an endpoint changes shape the scraper falls
back to scrolling the relevant tab and harvesting media nodes from whatever
the web app fetched on its own (REST or GraphQL).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlencode

from intelligence.schemas import MediaAsset, PlatformScrapeResult, RawItem, Surface
from scrapers.base_scraper import (
    BaseScraper,
    EndpointError,
    PrivateProfileError,
    ProfileNotFoundError,
    walk_json,
)

log = logging.getLogger(__name__)

IG_SURFACES: tuple[Surface, ...] = (
    Surface.IG_POST,
    Surface.IG_REEL,
    Surface.IG_STORY,
    Surface.IG_HIGHLIGHT,
    Surface.IG_TAGGED,
)


def is_media_node(node: dict) -> bool:
    """Heuristic that recognizes an Instagram media object in any API payload."""

    if "taken_at" not in node or "media_type" not in node:
        return False
    if not ("pk" in node or "id" in node):
        return False
    return any(k in node for k in ("image_versions2", "video_versions", "carousel_media"))


def find_media_nodes(payload: Any) -> list[dict]:
    """Top-level media objects in ``payload`` (carousel children are not split out)."""

    found: list[dict] = []
    stack = [payload]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            if is_media_node(node):
                found.append(node)
                continue
            stack.extend(node.values())
        elif isinstance(node, list):
            stack.extend(reversed(node))
    return found


def media_pk(node: dict) -> str:
    pk = node.get("pk")
    if pk is None:
        pk = str(node.get("id", "")).split("_")[0]
    return str(pk)


def _best(candidates: list[dict] | None) -> dict | None:
    if not candidates:
        return None
    return max(candidates, key=lambda c: (c.get("width") or 0) * (c.get("height") or 0))


def _single_assets(node: dict, role: str, index: int) -> list[MediaAsset]:
    assets: list[MediaAsset] = []
    image = _best((node.get("image_versions2") or {}).get("candidates"))
    video = _best(node.get("video_versions"))
    if video and video.get("url"):
        assets.append(
            MediaAsset(
                kind="video",
                url=video["url"],
                width=video.get("width"),
                height=video.get("height"),
                duration_s=node.get("video_duration"),
                role="slide" if role == "slide" else "primary",
                index=index,
            )
        )
        if image and image.get("url"):
            assets.append(
                MediaAsset(
                    kind="image",
                    url=image["url"],
                    width=image.get("width"),
                    height=image.get("height"),
                    role="cover",
                    index=index,
                )
            )
    elif image and image.get("url"):
        assets.append(
            MediaAsset(
                kind="image",
                url=image["url"],
                width=image.get("width"),
                height=image.get("height"),
                role="slide" if role == "slide" else "primary",
                index=index,
            )
        )
    return assets


def extract_assets(node: dict) -> list[MediaAsset]:
    if node.get("media_type") == 8 and node.get("carousel_media"):
        assets: list[MediaAsset] = []
        for idx, child in enumerate(node["carousel_media"]):
            assets.extend(_single_assets(child, "slide", idx))
        return assets
    return _single_assets(node, "primary", 0)


def classify_feed_surface(node: dict) -> Surface:
    product = (node.get("product_type") or "").lower()
    if product in ("clips", "igtv") or (
        node.get("media_type") == 2 and product != "feed" and not node.get("carousel_media")
    ):
        return Surface.IG_REEL
    return Surface.IG_POST


class InstagramScraper(BaseScraper):
    platform = "instagram"
    origin = "https://www.instagram.com"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.add_capture("api", r"instagram\.com/(api/v1/|graphql/query|api/graphql)")
        self.user_id: str | None = None
        self.username: str = ""

    # ---------------------------------------------------------- API helpers
    @property
    def _headers(self) -> dict[str, str]:
        return {
            "X-IG-App-ID": self.settings.scrape.ig_app_id,
            "X-Requested-With": "XMLHttpRequest",
            "X-ASBD-ID": "129477",
            "Accept": "*/*",
        }

    @staticmethod
    def _classify(status: int, data: Any) -> str | None:
        if isinstance(data, dict):
            message = str(data.get("message", "")).lower()
            if data.get("require_login") or message in ("login_required",):
                return "login"
            if "checkpoint" in message or "challenge" in message or data.get("checkpoint_url"):
                return "login"
            if "wait a few minutes" in message or "rate limit" in message or data.get("spam"):
                return "rate_limited"
        if status == 401:
            return "rate_limited" if isinstance(data, dict) and "wait" in str(data).lower() else "login"
        if status == 404:
            return "not_found"
        return None

    async def api(self, path: str, params: dict[str, Any] | None = None) -> Any:
        url = f"{self.origin}{path}"
        if params:
            url += ("&" if "?" in url else "?") + urlencode(params, doseq=True)
        data = await self.fetch_json(url, headers=self._headers, classify=self._classify)
        await self.human_pause()
        return data

    async def api_post(self, path: str, form: dict[str, Any]) -> Any:
        url = f"{self.origin}{path}"
        headers = dict(self._headers, **{"Content-Type": "application/x-www-form-urlencoded"})
        status, text = await self.page_fetch(url, headers=headers, method="POST", body=urlencode(form))
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise EndpointError(f"{path} -> HTTP {status}: non-JSON response") from exc
        verdict = self._classify(status, data)
        if verdict == "login":
            from auth.session_manager import SessionExpiredError

            raise SessionExpiredError(self.platform, f"HTTP {status} from {path}")
        if status >= 400 or verdict:
            raise EndpointError(f"{path} -> HTTP {status}: {text[:200]}")
        await self.human_pause()
        return data

    def _assert_logged_in(self) -> None:
        assert self.page is not None
        if "/accounts/login" in self.page.url or "/challenge" in self.page.url:
            from auth.session_manager import SessionExpiredError

            raise SessionExpiredError("instagram", f"redirected to {self.page.url}")

    async def _own_account_id(self) -> str | None:
        assert self.context is not None
        for cookie in await self.context.cookies(self.origin):
            if cookie.get("name") == "ds_user_id":
                return str(cookie.get("value"))
        return None

    # ------------------------------------------------------------- scraping
    async def scrape(self, username: str, surfaces: Iterable[Surface] | None = None) -> PlatformScrapeResult:
        wanted = [s for s in (surfaces or IG_SURFACES) if s in IG_SURFACES]
        self.username = username
        result = PlatformScrapeResult(platform="instagram", username=username)

        await self.goto(f"{self.origin}/{quote(username)}/")
        self._assert_logged_in()

        profile = await self._profile(username)
        result.profile_raw = profile
        self.user_id = str(profile.get("id") or profile.get("pk") or "")
        if not self.user_id:
            raise EndpointError("Profile payload did not include a user id")
        if profile.get("is_private"):
            own = await self._own_account_id()
            if own != self.user_id:
                raise PrivateProfileError(
                    f"@{username} is a private Instagram account. This engine only audits public "
                    "creator profiles (or the operator's own account)."
                )
            result.notes.append("Private account processed because it is the authenticated operator's own account.")
        self.progress(
            f"[instagram] @{username}: {profile.get('edge_followed_by', {}).get('count', '?')} followers, "
            f"{profile.get('edge_owner_to_timeline_media', {}).get('count', '?')} posts"
        )

        if Surface.IG_POST in wanted or Surface.IG_REEL in wanted:
            feed_items = await self._surface_cached_or(Surface.IG_POST, self._collect_feed)
            reel_items = await self._surface_cached_or(Surface.IG_REEL, lambda: self._collect_reels(feed_items))
            for item in feed_items + reel_items:
                if item.surface in wanted:
                    result.items.append(item)
            result.surfaces_completed.extend(s for s in (Surface.IG_POST, Surface.IG_REEL) if s in wanted)

        if Surface.IG_STORY in wanted:
            result.items.extend(await self._surface_cached_or(Surface.IG_STORY, self._collect_stories))
            result.surfaces_completed.append(Surface.IG_STORY)
        if Surface.IG_HIGHLIGHT in wanted:
            highlights = await self._surface_cached_or(Surface.IG_HIGHLIGHT, self._collect_highlights)
            result.items.extend(highlights)
            result.surfaces_completed.append(Surface.IG_HIGHLIGHT)
            titles = []
            for item in highlights:
                title = item.extra.get("highlight_title")
                if title and title not in titles:
                    titles.append(title)
            result.profile_raw["_highlight_titles"] = titles
        if Surface.IG_TAGGED in wanted:
            result.items.extend(await self._surface_cached_or(Surface.IG_TAGGED, self._collect_tagged))
            result.surfaces_completed.append(Surface.IG_TAGGED)

        if self.settings.scrape.download_media:
            await self._download_all(result)
        self.save_result(result)
        return result

    async def _surface_cached_or(self, surface: Surface, collector: Any) -> list[RawItem]:
        cached = self.load_checkpoint(surface)
        if cached is not None:
            self.progress(f"[instagram] {surface.label}: resuming with {len(cached)} cached items")
            return cached
        items = await collector()
        self.checkpoint_items(surface, items)
        self.progress(f"[instagram] {surface.label}: {len(items)} items")
        return items

    async def _profile(self, username: str) -> dict:
        try:
            data = await self.api("/api/v1/users/web_profile_info/", {"username": username})
        except ProfileNotFoundError as exc:
            raise ProfileNotFoundError(f"Instagram user @{username} was not found") from exc
        user = ((data or {}).get("data") or {}).get("user")
        if not user:
            raise ProfileNotFoundError(f"Instagram user @{username} was not found")
        return user

    def _cap_reached(self, count: int) -> bool:
        return bool(self.cap()) and count >= self.cap()

    async def _paginate(self, path: str, label: str, extra: dict[str, Any] | None = None) -> list[dict]:
        """Generic ``max_id`` pagination over a v1 feed endpoint."""

        nodes: dict[str, dict] = {}
        max_id: str | None = None
        page_no = 0
        while True:
            params: dict[str, Any] = {"count": self.settings.scrape.ig_page_size}
            if extra:
                params.update(extra)
            if max_id:
                params["max_id"] = max_id
            data = await self.api(path, params)
            page_no += 1
            for node in find_media_nodes(data.get("items", data)):
                nodes.setdefault(media_pk(node), node)
            self.progress(f"[instagram] {label}: page {page_no}, {len(nodes)} items")
            if self._cap_reached(len(nodes)):
                break
            max_id = data.get("next_max_id")
            if not data.get("more_available") or not max_id:
                break
        return list(nodes.values())[: self.cap() or None]

    async def _harvest_by_scrolling(self, url: str, label: str) -> list[dict]:
        """Fallback: open a tab and scroll, harvesting media from captured responses."""

        self.clear_capture("api")
        await self.goto(url)
        self._assert_logged_in()
        nodes: dict[str, dict] = {}

        def harvest() -> int:
            for captured in self.bucket("api").responses:
                for node in find_media_nodes(captured.data):
                    nodes.setdefault(media_pk(node), node)
            return len(nodes)

        await self.natural_scroll(harvest, max_items=self.cap(), label=label)
        harvest()
        return list(nodes.values())[: self.cap() or None]

    def _to_raw(self, node: dict, surface: Surface, **extra: Any) -> RawItem:
        return RawItem(
            surface=surface,
            native_id=media_pk(node),
            raw=node,
            assets=extract_assets(node),
            extra={"shortcode": node.get("code"), **extra},
        )

    async def _collect_feed(self) -> list[RawItem]:
        try:
            nodes = await self._paginate(f"/api/v1/feed/user/{self.user_id}/", "feed")
        except EndpointError as exc:
            log.warning("Feed endpoint failed (%s); falling back to scrolling the grid", exc)
            nodes = await self._harvest_by_scrolling(f"{self.origin}/{quote(self.username)}/", "feed (scroll)")
        items = [self._to_raw(n, classify_feed_surface(n)) for n in nodes]
        await self._attach_comments(items)
        return items

    async def _collect_reels(self, feed_items: list[RawItem]) -> list[RawItem]:
        """Reels tab: catches Reels that were not shared to the main grid."""

        known = {item.native_id for item in feed_items}
        nodes: dict[str, dict] = {}
        max_id = ""
        try:
            while True:
                form = {
                    "target_user_id": self.user_id,
                    "page_size": self.settings.scrape.ig_page_size,
                    "include_feed_video": "true",
                }
                if max_id:
                    form["max_id"] = max_id
                data = await self.api_post("/api/v1/clips/user/", form)
                for node in find_media_nodes(data.get("items", [])):
                    nodes.setdefault(media_pk(node), node)
                paging = data.get("paging_info") or {}
                if self._cap_reached(len(nodes)) or not paging.get("more_available") or not paging.get("max_id"):
                    break
                max_id = str(paging["max_id"])
        except EndpointError as exc:
            log.warning("Clips endpoint failed (%s); falling back to scrolling the Reels tab", exc)
            for node in await self._harvest_by_scrolling(
                f"{self.origin}/{quote(self.username)}/reels/", "reels (scroll)"
            ):
                nodes.setdefault(media_pk(node), node)
        new_nodes = [n for pk, n in nodes.items() if pk not in known]
        items = [self._to_raw(n, Surface.IG_REEL) for n in new_nodes]
        await self._attach_comments(items)
        return items

    async def _attach_comments(self, items: list[RawItem]) -> None:
        limit = self.settings.scrape.ig_comments_per_post
        if limit <= 0:
            return
        for idx, item in enumerate(items, 1):
            if not (item.raw.get("comment_count") or 0):
                continue
            ckey = item.native_id
            cached = self.cache.get(f"{self.ns}:comments", ckey)
            if cached is None:
                cached = []
                try:
                    data = await self.api(
                        f"/api/v1/media/{item.native_id}/comments/",
                        {"can_support_threading": "true", "permalink_enabled": "false"},
                    )
                    for c in (data.get("comments") or [])[:limit]:
                        user = c.get("user") or {}
                        cached.append(
                            {
                                "text": c.get("text", ""),
                                "created_at": c.get("created_at"),
                                "likes": c.get("comment_like_count"),
                                # Only whether the creator replied - commenter identities are not kept.
                                "by_owner": str(user.get("pk") or user.get("id")) == self.user_id,
                            }
                        )
                except EndpointError as exc:
                    log.debug("Comments unavailable for %s: %s", item.native_id, exc)
                self.cache.set(f"{self.ns}:comments", ckey, cached)
            item.extra["comments"] = cached
            if idx % 20 == 0:
                self.progress(f"[instagram] comments: {idx}/{len(items)} posts")

    async def _collect_stories(self) -> list[RawItem]:
        try:
            data = await self.api("/api/v1/feed/reels_media/", {"reel_ids": self.user_id})
            nodes = self._reel_items(data, self.user_id or "")
        except EndpointError as exc:
            log.warning("Stories endpoint failed (%s); opening the story viewer instead", exc)
            nodes = await self._harvest_viewer(f"{self.origin}/stories/{quote(self.username)}/")
        return [self._to_raw(n, Surface.IG_STORY, expiring_at=n.get("expiring_at")) for n in nodes]

    @staticmethod
    def _reel_items(data: dict, reel_id: str) -> list[dict]:
        reels = data.get("reels") or {}
        if reel_id in reels:
            return find_media_nodes(reels[reel_id].get("items", []))
        for reel in data.get("reels_media") or []:
            if str(reel.get("id")) == reel_id:
                return find_media_nodes(reel.get("items", []))
        return []

    async def _harvest_viewer(self, url: str) -> list[dict]:
        """Open the story viewer and advance through every slide, capturing payloads."""

        assert self.page is not None
        self.clear_capture("api")
        await self.goto(url)
        self._assert_logged_in()
        for label in ("View story", "View Story"):
            button = self.page.get_by_role("button", name=label)
            if await button.count():
                await button.first.click()
                break
        seen = 0
        for _ in range(400):
            await self.human_pause(0.6, 1.4)
            if "/stories/" not in self.page.url:
                break
            await self.page.keyboard.press("ArrowRight")
            seen += 1
        nodes: dict[str, dict] = {}
        for captured in self.bucket("api").responses:
            for node in find_media_nodes(captured.data):
                nodes.setdefault(media_pk(node), node)
        log.debug("Story viewer advanced %s times, captured %s slides", seen, len(nodes))
        return list(nodes.values())

    async def _collect_highlights(self) -> list[RawItem]:
        try:
            tray_data = await self.api(f"/api/v1/highlights/{self.user_id}/highlights_tray/")
            tray = tray_data.get("tray") or []
        except EndpointError as exc:
            log.warning("Highlights tray endpoint failed: %s", exc)
            tray = []
        if not tray:
            # Fallback: highlight ids are present in whatever the profile page loaded.
            for captured in self.bucket("api").responses:
                for node in walk_json(captured.data):
                    if str(node.get("id", "")).startswith("highlight:") and node.get("title") is not None:
                        tray.append(node)
        seen_ids: set[str] = set()
        tray = [t for t in tray if not (str(t.get("id")) in seen_ids or seen_ids.add(str(t.get("id"))))]
        self.progress(f"[instagram] {len(tray)} highlight trays found")

        items: list[RawItem] = []
        batch = max(1, self.settings.scrape.ig_highlight_batch_size)
        for start in range(0, len(tray), batch):
            chunk = tray[start : start + batch]
            reel_ids = [str(t["id"]) for t in chunk]
            try:
                data = await self.api("/api/v1/feed/reels_media/", {"reel_ids": reel_ids})
            except EndpointError as exc:
                log.warning("Highlight batch failed (%s); using the viewer", exc)
                data = None
            for tray_entry in chunk:
                hid = str(tray_entry["id"])
                nodes = self._reel_items(data, hid) if data else []
                if not nodes:
                    nodes = await self._harvest_viewer(f"{self.origin}/stories/highlights/{hid.split(':')[-1]}/")
                title = tray_entry.get("title") or ""
                for pos, node in enumerate(nodes):
                    items.append(
                        self._to_raw(
                            node,
                            Surface.IG_HIGHLIGHT,
                            highlight_id=hid,
                            highlight_title=title,
                            highlight_position=pos,
                            highlight_size=len(nodes),
                        )
                    )
                self.progress(f"[instagram] highlight '{title}': {len(nodes)} slides")
            if self._cap_reached(len(items)):
                break
        return items[: self.cap() or None]

    async def _collect_tagged(self) -> list[RawItem]:
        try:
            nodes = await self._paginate(f"/api/v1/usertags/{self.user_id}/feed/", "tagged")
        except EndpointError as exc:
            log.warning("Tagged endpoint failed (%s); scrolling the Tagged tab", exc)
            nodes = await self._harvest_by_scrolling(f"{self.origin}/{quote(self.username)}/tagged/", "tagged (scroll)")
        items = []
        for node in nodes:
            owner = node.get("user") or node.get("owner") or {}
            items.append(
                self._to_raw(
                    node, Surface.IG_TAGGED, author=owner.get("username"), author_is_verified=owner.get("is_verified")
                )
            )
        return items

    # ------------------------------------------------------------ downloads
    async def _download_all(self, result: PlatformScrapeResult) -> None:
        avatar = result.profile_raw.get("profile_pic_url_hd") or result.profile_raw.get("profile_pic_url")
        if avatar:
            path = await self.download(avatar, self.media_dir / "avatar.jpg")
            if path:
                result.profile_raw["_avatar_path"] = str(path)

        async def resolve(item: RawItem, asset: MediaAsset) -> Path | None:
            ext = "mp4" if asset.kind == "video" else "jpg"
            dest = self.media_dir / item.surface.value / item.native_id / f"{asset.role}_{asset.index:02d}.{ext}"
            return await self.download(asset.url or "", dest)

        for surface in IG_SURFACES:
            items = result.by_surface(surface)
            if items:
                await self.download_assets(items, resolve, surface.label)
