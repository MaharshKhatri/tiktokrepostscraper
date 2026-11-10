"""TikTok profile ingestion.

Surfaces collected for ``tiktok.com/@{username}``:

* profile bio + metrics (from the page's rehydration JSON);
* every original post on the Videos tab (video, cover, caption, audio, stats);
* every item on the profile's **Reposts** tab.

TikTok's item-list endpoints require request signatures that only the web app
can compute, so this scraper never calls them directly. Instead it lets the
real page do the work: it scrolls each tab with human pacing and intercepts the
``/api/post/item_list/`` and ``/api/repost/item_list/`` XHR responses. Media is
fetched with ``yt-dlp`` (using cookies exported from the saved session), with a
direct CDN download as fallback and for photo-mode posts.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
from collections.abc import Iterable
from pathlib import Path
from typing import Any
from urllib.parse import quote

from auth.session_manager import SessionExpiredError, export_netscape_cookies
from intelligence.schemas import MediaAsset, PlatformScrapeResult, RawItem, Surface
from scrapers.base_scraper import (
    BaseScraper,
    PrivateProfileError,
    ProfileNotFoundError,
    ScrapeError,
    walk_json,
)

log = logging.getLogger(__name__)

TT_SURFACES: tuple[Surface, ...] = (Surface.TT_VIDEO, Surface.TT_REPOST)

CAPTCHA_SELECTORS = (
    "#captcha-verify-container-main-page",
    ".captcha-verify-container",
    "#captcha_container",
    "div[class*='captcha_verify']",
    "iframe[src*='captcha']",
)


def is_tt_item(node: dict) -> bool:
    return bool(node.get("id")) and "createTime" in node and ("video" in node or "imagePost" in node)


def extract_tt_assets(item: dict) -> list[MediaAsset]:
    assets: list[MediaAsset] = []
    image_post = item.get("imagePost") or {}
    images = image_post.get("images") or []
    if images:
        for idx, img in enumerate(images):
            urls = (img.get("imageURL") or {}).get("urlList") or []
            if urls:
                assets.append(
                    MediaAsset(
                        kind="image",
                        url=urls[0],
                        width=img.get("imageWidth"),
                        height=img.get("imageHeight"),
                        role="slide",
                        index=idx,
                    )
                )
        return assets
    video = item.get("video") or {}
    play = video.get("playAddr") or video.get("downloadAddr")
    if isinstance(play, list):
        play = play[0] if play else None
    assets.append(
        MediaAsset(
            kind="video",
            url=play or None,
            width=video.get("width"),
            height=video.get("height"),
            duration_s=video.get("duration"),
            role="primary",
            index=0,
        )
    )
    cover = video.get("originCover") or video.get("cover")
    if cover:
        assets.append(MediaAsset(kind="image", url=cover, role="cover", index=0))
    return assets


def tt_item_url(item: dict) -> str:
    author = (item.get("author") or {}).get("uniqueId") or "i"
    kind = "photo" if (item.get("imagePost") or {}).get("images") else "video"
    return f"https://www.tiktok.com/@{author}/{kind}/{item.get('id')}"


class _QuietLogger:
    def debug(self, msg: str) -> None:
        pass

    def info(self, msg: str) -> None:
        pass

    def warning(self, msg: str) -> None:
        log.debug("yt-dlp: %s", msg)

    def error(self, msg: str) -> None:
        log.debug("yt-dlp: %s", msg)


class TikTokScraper(BaseScraper):
    platform = "tiktok"
    origin = "https://www.tiktok.com"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.add_capture("posts", r"tiktok\.com/api/(post|creator)/item_list")
        self.add_capture("reposts", r"tiktok\.com/api/repost/item_list")
        self.add_capture("user", r"tiktok\.com/api/user/detail")
        self.username = ""
        self.repost_tab_status = "not attempted"

    # ------------------------------------------------------------- scraping
    async def scrape(self, username: str, surfaces: Iterable[Surface] | None = None) -> PlatformScrapeResult:
        wanted = [s for s in (surfaces or TT_SURFACES) if s in TT_SURFACES]
        self.username = username
        result = PlatformScrapeResult(platform="tiktok", username=username)

        await self.goto(f"{self.origin}/@{quote(username)}")
        await self._handle_captcha()
        assert self.page is not None
        if "/login" in self.page.url:
            raise SessionExpiredError("tiktok", f"redirected to {self.page.url}")

        user_info = await self._profile()
        result.profile_raw = user_info
        user = user_info.get("user") or {}
        if user.get("privateAccount"):
            raise PrivateProfileError(
                f"@{username} is a private TikTok account. This engine only audits public creator profiles."
            )
        stats = user_info.get("stats") or {}
        self.progress(
            f"[tiktok] @{username}: {stats.get('followerCount', '?')} followers, {stats.get('videoCount', '?')} videos"
        )

        if Surface.TT_VIDEO in wanted:
            result.items.extend(await self._cached_or(Surface.TT_VIDEO, self._collect_videos))
            result.surfaces_completed.append(Surface.TT_VIDEO)
        if Surface.TT_REPOST in wanted:
            reposts = await self._cached_or(Surface.TT_REPOST, self._collect_reposts)
            if self.repost_tab_status == "hidden":
                result.notes.append("Reposts tab was not visible (hidden by the account or unavailable).")
            elif self.repost_tab_status == "empty":
                result.notes.append("Reposts tab was visible but no reposts loaded.")
            result.items.extend(reposts)
            result.surfaces_completed.append(Surface.TT_REPOST)

        if self.settings.scrape.download_media:
            await self._download_all(result)
        self.save_result(result)
        return result

    async def _cached_or(self, surface: Surface, collector: Any) -> list[RawItem]:
        cached = self.load_checkpoint(surface)
        if cached is not None:
            self.progress(f"[tiktok] {surface.label}: resuming with {len(cached)} cached items")
            return cached
        items = await collector()
        self.checkpoint_items(surface, items)
        self.progress(f"[tiktok] {surface.label}: {len(items)} items")
        return items

    async def _handle_captcha(self) -> None:
        """If TikTok shows a verification puzzle, hand it to the operator (headed mode)."""

        assert self.page is not None
        if not await self._captcha_visible():
            return
        if self.settings.browser.headless:
            raise ScrapeError(
                "TikTok is showing a verification puzzle. Re-run with --headed and solve it in the "
                "browser window, or refresh the session with `audit auth tiktok`."
            )
        wait_s = self.settings.scrape.tiktok_captcha_wait_s
        self.progress(f"[tiktok] Verification puzzle shown - please solve it in the browser (waiting {wait_s}s)")
        for _ in range(wait_s // 2):
            await asyncio.sleep(2)
            if not await self._captcha_visible():
                await self.human_pause(1.5, 3.0)
                return
        raise ScrapeError("TikTok verification puzzle was not solved in time.")

    async def _captcha_visible(self) -> bool:
        assert self.page is not None
        for selector in CAPTCHA_SELECTORS:
            try:
                loc = self.page.locator(selector)
                if await loc.count() and await loc.first.is_visible():
                    return True
            except Exception:
                continue
        return False

    async def _profile(self) -> dict:
        assert self.page is not None
        info: dict | None = None
        for script_id in ("__UNIVERSAL_DATA_FOR_REHYDRATION__", "SIGI_STATE"):
            loc = self.page.locator(f"script#{script_id}")
            if not await loc.count():
                continue
            try:
                data = json.loads(await loc.first.text_content() or "{}")
            except json.JSONDecodeError:
                continue
            scope = data.get("__DEFAULT_SCOPE__") or {}
            detail = scope.get("webapp.user-detail") or {}
            if detail.get("userInfo"):
                info = detail["userInfo"]
                break
            if detail.get("statusCode") in (10221, 10202):
                raise ProfileNotFoundError(f"TikTok user @{self.username} was not found")
            # Legacy SIGI_STATE layout
            users = (data.get("UserModule") or {}).get("users") or {}
            if users:
                user = next(iter(users.values()))
                stats = ((data.get("UserModule") or {}).get("stats") or {}).get(user.get("uniqueId"), {})
                info = {"user": user, "stats": stats}
                break
        if info is None:
            await self.wait_for_capture("user", timeout_s=8)
            for captured in self.bucket("user").responses:
                if isinstance(captured.data, dict) and captured.data.get("userInfo"):
                    info = captured.data["userInfo"]
                    break
        if info is None:
            body = (await self.page.content()).lower()
            if "couldn't find this account" in body or "user not found" in body:
                raise ProfileNotFoundError(f"TikTok user @{self.username} was not found")
            raise ScrapeError("Could not locate TikTok profile data on the page (layout change or block).")
        return info

    def _harvest(self, bucket: str, store: dict[str, dict]) -> int:
        for captured in self.bucket(bucket).responses:
            data = captured.data
            items = data.get("itemList") if isinstance(data, dict) else None
            if items is None:
                items = [n for n in walk_json(data) if is_tt_item(n)]
            for node in items or []:
                if is_tt_item(node):
                    store.setdefault(str(node["id"]), node)
        return len(store)

    def _exhausted(self, bucket: str) -> bool:
        responses = self.bucket(bucket).responses
        if not responses:
            return False
        last = responses[-1].data
        return isinstance(last, dict) and last.get("hasMore") is False

    async def _collect_videos(self) -> list[RawItem]:
        store: dict[str, dict] = {}
        if not self.bucket("posts").responses:
            await self.wait_for_capture("posts", timeout_s=15)
        self._harvest("posts", store)
        await self.natural_scroll(
            lambda: self._harvest("posts", store),
            is_exhausted=lambda: self._exhausted("posts"),
            max_items=self.cap(),
            label="videos",
        )
        self._harvest("posts", store)
        nodes = list(store.values())[: self.cap() or None]
        # Only the profile owner's own posts belong on this surface.
        own = [
            n
            for n in nodes
            if ((n.get("author") or {}).get("uniqueId") or self.username).lower() == self.username.lower()
        ]
        return [self._to_raw(n, Surface.TT_VIDEO) for n in own]

    async def _open_tab(self, selectors: list[str]) -> bool:
        assert self.page is not None
        await self.page.keyboard.press("Home")
        await self.human_pause(0.8, 1.6)
        for selector in selectors:
            loc = self.page.locator(selector)
            try:
                if await loc.count() and await loc.first.is_visible():
                    await loc.first.scroll_into_view_if_needed()
                    await self.human_pause(0.4, 1.0)
                    await loc.first.click()
                    return True
            except Exception as exc:  # selector syntax or detached node
                log.debug("Tab selector %s failed: %s", selector, exc)
        return False

    async def _collect_reposts(self) -> list[RawItem]:
        self.clear_capture("reposts")
        if not await self._open_tab(self.settings.scrape.tiktok_repost_tab_selectors):
            self.progress("[tiktok] Reposts tab not visible on this profile - skipping")
            self.repost_tab_status = "hidden"
            return []
        await self._handle_captcha()
        store: dict[str, dict] = {}
        if not await self.wait_for_capture("reposts", timeout_s=15):
            self.progress("[tiktok] Reposts tab opened but no repost list loaded (empty or hidden)")
            self.repost_tab_status = "empty"
            return []
        self.repost_tab_status = "ok"
        self._harvest("reposts", store)
        await self.natural_scroll(
            lambda: self._harvest("reposts", store),
            is_exhausted=lambda: self._exhausted("reposts"),
            max_items=self.cap(),
            label="reposts",
        )
        self._harvest("reposts", store)
        nodes = list(store.values())[: self.cap() or None]
        items = []
        for node in nodes:
            author = node.get("author") or {}
            items.append(
                self._to_raw(
                    node, Surface.TT_REPOST, author=author.get("uniqueId"), author_is_verified=author.get("verified")
                )
            )
        return items

    def _to_raw(self, node: dict, surface: Surface, **extra: Any) -> RawItem:
        return RawItem(
            surface=surface,
            native_id=str(node["id"]),
            raw=node,
            assets=extract_tt_assets(node),
            extra={"url": tt_item_url(node), **extra},
        )

    # ------------------------------------------------------------ downloads
    def _cookie_file(self) -> Path | None:
        state = self.session_path
        out = self.settings.tiktok_cookie_file()
        if not state.exists():
            return None
        if not out.exists() or out.stat().st_mtime < state.stat().st_mtime:
            export_netscape_cookies(state, out)
        return out

    def _ytdlp(self, url: str, out_dir: Path) -> Path | None:
        try:
            import yt_dlp  # imported lazily: heavy module
        except ImportError:
            return None
        out_dir.mkdir(parents=True, exist_ok=True)
        opts: dict[str, Any] = {
            "outtmpl": str(out_dir / "video.%(ext)s"),
            "format": "b[ext=mp4]/bv*+ba/b",
            "merge_output_format": "mp4",
            "quiet": True,
            "no_warnings": True,
            "noprogress": True,
            "noplaylist": True,
            "retries": 3,
            "fragment_retries": 3,
            "logger": _QuietLogger(),
            "http_headers": {"Referer": "https://www.tiktok.com/"},
            "ffmpeg_location": self.settings.media.ffmpeg_bin if "/" in self.settings.media.ffmpeg_bin else None,
        }
        cookie_file = self._cookie_file()
        if cookie_file:
            opts["cookiefile"] = str(cookie_file)
        opts = {k: v for k, v in opts.items() if v is not None}
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                ydl.extract_info(url, download=True)
        except Exception as exc:  # yt-dlp raises many types; any failure -> fallback
            log.debug("yt-dlp failed for %s: %s", url, exc)
            return None
        candidates = sorted(p for p in out_dir.glob("video.*") if p.suffix not in (".part", ".ytdl"))
        return candidates[0] if candidates else None

    async def _download_all(self, result: PlatformScrapeResult) -> None:
        avatar = (result.profile_raw.get("user") or {}).get("avatarLarger")
        if avatar:
            path = await self.download(avatar, self.media_dir / "avatar.jpg")
            if path:
                result.profile_raw["_avatar_path"] = str(path)
        ytdlp_sem = asyncio.Semaphore(max(1, self.settings.rate_limits.download_concurrency - 1))

        async def resolve(item: RawItem, asset: MediaAsset) -> Path | None:
            base = self.media_dir / item.surface.value / item.native_id
            if asset.kind == "video":
                existing = [p for p in base.glob("video.*") if p.suffix not in (".part", ".ytdl")]
                if existing and existing[0].stat().st_size > 1024:
                    return existing[0]
                async with ytdlp_sem:
                    path = await asyncio.to_thread(self._ytdlp, item.extra.get("url") or tt_item_url(item.raw), base)
                    await asyncio.sleep(random.uniform(1.0, 2.5))
                if path:
                    return path
                cdn_url = item.extra.get("cdn_video_url")
                if cdn_url:
                    return await self.download(cdn_url, base / "video.mp4", referer=f"{self.origin}/", min_bytes=20_000)
                return None
            ext = "jpg"
            return await self.download(
                asset.url or "", base / f"{asset.role}_{asset.index:02d}.{ext}", referer=f"{self.origin}/"
            )

        for surface in TT_SURFACES:
            items = result.by_surface(surface)
            # yt-dlp works from the post URL; the CDN URL is kept only as a fallback.
            for item in items:
                for asset in item.assets:
                    if asset.kind == "video" and asset.url != item.extra.get("url"):
                        if asset.url:
                            item.extra["cdn_video_url"] = asset.url
                        asset.url = item.extra.get("url") or tt_item_url(item.raw)
            if items:
                await self.download_assets(items, resolve, surface.label)
