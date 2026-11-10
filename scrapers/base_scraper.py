"""Async Playwright controller shared by the platform scrapers.

Responsibilities:

* launch Chromium with the operator's saved ``storage_state``;
* capture JSON network responses into named buckets (``page.on("response")``)
  so scrapers can read exactly what the web app itself received;
* pace navigation, scrolling and requests with randomized human-like delays;
* same-origin ``fetch`` from inside the page (cookies + CSRF handled by the
  browser) with retry/backoff on rate limits;
* download media files with resume-safe skipping.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import re
import time
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from playwright.async_api import (
    Browser,
    BrowserContext,
    Page,
    Playwright,
    Response,
    async_playwright,
)
from playwright.async_api import Error as PlaywrightError
from playwright.async_api import TimeoutError as PlaywrightTimeoutError

from auth.session_manager import SessionMissingError, context_kwargs
from config.settings import Platform, Settings
from intelligence.schemas import MediaAsset, PlatformScrapeResult, RawItem, Surface
from storage.cache_manager import CacheManager

log = logging.getLogger(__name__)


class ScrapeError(RuntimeError):
    """Generic, non-recoverable scraping failure."""


class PrivateProfileError(ScrapeError):
    """Raised for private accounts - the engine only audits public creator profiles."""


class ProfileNotFoundError(ScrapeError):
    pass


class RateLimitedError(ScrapeError):
    pass


class EndpointError(ScrapeError):
    """An internal endpoint answered with something unusable (schema/route change)."""


@dataclass
class CapturedResponse:
    url: str
    status: int
    data: Any
    captured_at: float = field(default_factory=time.monotonic)


@dataclass
class CaptureBucket:
    name: str
    pattern: re.Pattern[str]
    responses: list[CapturedResponse] = field(default_factory=list)
    event: asyncio.Event = field(default_factory=asyncio.Event)


ProgressCallback = Callable[[str], None]


class BaseScraper(ABC):
    """Lifecycle + utilities. Subclasses implement :meth:`scrape`."""

    platform: Platform
    origin: str

    def __init__(
        self,
        settings: Settings,
        cache: CacheManager,
        run_key: str,
        progress: ProgressCallback | None = None,
    ) -> None:
        self.settings = settings
        self.cache = cache
        self.run_key = run_key
        self.progress = progress or (lambda msg: log.info(msg))
        self._pw: Playwright | None = None
        self._browser: Browser | None = None
        self.context: BrowserContext | None = None
        self.page: Page | None = None
        self._buckets: dict[str, CaptureBucket] = {}
        self._raw_log_path: Path | None = None
        self._download_sem = asyncio.Semaphore(settings.rate_limits.download_concurrency)

    # ------------------------------------------------------------- lifecycle
    @property
    def session_path(self) -> Path:
        return self.settings.session_file(self.platform)

    @property
    def raw_dir(self) -> Path:
        return self.settings.raw_dir(self.run_key, self.platform)

    @property
    def media_dir(self) -> Path:
        return self.settings.media_dir(self.run_key, self.platform)

    @property
    def ns(self) -> str:
        """Cache namespace prefix for this run + platform."""

        return f"{self.run_key}:{self.platform}"

    async def start(self, require_session: bool = True) -> None:
        if require_session and not self.session_path.exists():
            raise SessionMissingError(self.platform, self.session_path)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.media_dir.mkdir(parents=True, exist_ok=True)
        if self.settings.scrape.save_raw_responses:
            self._raw_log_path = self.raw_dir / "network_capture.jsonl"
        b = self.settings.browser
        launch: dict[str, Any] = {"headless": b.headless, "slow_mo": b.slow_mo_ms}
        if b.executable_path:
            launch["executable_path"] = str(b.executable_path)
        if b.extra_args:
            launch["args"] = list(b.extra_args)
        self._pw = await async_playwright().start()
        self._browser = await self._pw.chromium.launch(**launch)
        ctx_kwargs = context_kwargs(self.settings)
        if self.session_path.exists():
            ctx_kwargs["storage_state"] = str(self.session_path)
        self.context = await self._browser.new_context(**ctx_kwargs)
        self.context.set_default_navigation_timeout(b.navigation_timeout_ms)
        self.page = await self.context.new_page()
        self.page.on("response", self._on_response)

    async def stop(self) -> None:
        try:
            if self.context and self.session_path.exists():
                # Persist refreshed cookies so long-lived sessions stay valid.
                state = await self.context.storage_state()
                if any(c.get("name") in ("sessionid", "sid_tt") for c in state.get("cookies", [])):
                    self.session_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        except PlaywrightError:
            pass
        finally:
            if self._browser:
                await self._browser.close()
            if self._pw:
                await self._pw.stop()
            self._browser = self._pw = self.context = self.page = None

    async def __aenter__(self) -> BaseScraper:
        await self.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.stop()

    @abstractmethod
    async def scrape(self, username: str, surfaces: Iterable[Surface] | None = None) -> PlatformScrapeResult:
        """Collect every requested surface for ``username``."""

    # ------------------------------------------------------ network capture
    def add_capture(self, name: str, pattern: str) -> CaptureBucket:
        bucket = CaptureBucket(name=name, pattern=re.compile(pattern))
        self._buckets[name] = bucket
        return bucket

    def bucket(self, name: str) -> CaptureBucket:
        return self._buckets[name]

    def clear_capture(self, name: str) -> None:
        bucket = self._buckets[name]
        bucket.responses.clear()
        bucket.event.clear()

    async def _on_response(self, response: Response) -> None:
        url = response.url
        matched = [b for b in self._buckets.values() if b.pattern.search(url)]
        if not matched:
            return
        try:
            ctype = (await response.header_value("content-type")) or ""
            if "json" not in ctype and "javascript" not in ctype and "text/plain" not in ctype:
                return
            text = await response.text()
            data = json.loads(text)
        except (PlaywrightError, json.JSONDecodeError, UnicodeDecodeError):
            return
        captured = CapturedResponse(url=url, status=response.status, data=data)
        for bucket in matched:
            bucket.responses.append(captured)
            bucket.event.set()
        if self._raw_log_path:
            try:
                with self._raw_log_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps({"url": url, "status": response.status, "data": data}) + "\n")
            except OSError:
                pass

    async def wait_for_capture(self, name: str, timeout_s: float = 15.0, after: int = 0) -> bool:
        """Wait until bucket ``name`` holds more than ``after`` responses."""

        bucket = self._buckets[name]
        deadline = time.monotonic() + timeout_s
        while len(bucket.responses) <= after:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            bucket.event.clear()
            try:
                await asyncio.wait_for(bucket.event.wait(), timeout=min(remaining, 1.0))
            except TimeoutError:
                continue
        return True

    # --------------------------------------------------------------- pacing
    async def human_pause(self, low: float | None = None, high: float | None = None) -> None:
        rl = self.settings.rate_limits
        low = rl.request_delay_min_s if low is None else low
        high = rl.request_delay_max_s if high is None else high
        await asyncio.sleep(random.uniform(low, max(low, high)))

    async def goto(self, url: str, wait_until: str = "domcontentloaded") -> Response | None:
        assert self.page is not None
        rl = self.settings.rate_limits
        for attempt in range(1, rl.max_retries + 1):
            try:
                resp = await self.page.goto(url, wait_until=wait_until)  # type: ignore[arg-type]
                try:
                    await self.page.wait_for_load_state("networkidle", timeout=8_000)
                except PlaywrightTimeoutError:
                    pass
                await self.human_pause(0.8, 2.0)
                return resp
            except (PlaywrightTimeoutError, PlaywrightError) as exc:
                if attempt == rl.max_retries:
                    raise ScrapeError(f"Navigation to {url} failed: {exc}") from exc
                await self._backoff(attempt, f"navigation error: {exc}")
        return None

    async def _backoff(self, attempt: int, reason: str) -> None:
        rl = self.settings.rate_limits
        delay = min(rl.backoff_max_s, rl.backoff_base_s * (2 ** (attempt - 1)))
        delay *= random.uniform(0.8, 1.3)
        self.progress(f"[{self.platform}] {reason} - backing off {delay:.0f}s (attempt {attempt})")
        await asyncio.sleep(delay)

    async def natural_scroll(
        self,
        count_items: Callable[[], int],
        is_exhausted: Callable[[], bool] | None = None,
        max_items: int = 0,
        label: str = "",
    ) -> int:
        """Scroll like a person until no new items appear.

        ``count_items`` reports how many unique items have been harvested so far;
        scrolling stops after ``scroll_idle_rounds`` rounds without growth, when
        ``is_exhausted`` says the feed has no more pages, or at ``max_items``.
        """

        assert self.page is not None
        rl = self.settings.rate_limits
        idle = 0
        last = count_items()
        for round_no in range(1, rl.scroll_max_rounds + 1):
            if is_exhausted and is_exhausted():
                break
            if max_items and last >= max_items:
                break
            # A few wheel steps of varying size, sometimes a small scroll back up.
            for _ in range(random.randint(2, 4)):
                await self.page.mouse.wheel(0, random.randint(450, 1100))
                await asyncio.sleep(random.uniform(0.15, 0.45))
            if random.random() < 0.15:
                await self.page.mouse.wheel(0, -random.randint(150, 400))
            await asyncio.sleep(random.uniform(rl.scroll_pause_min_s, rl.scroll_pause_max_s))
            if random.random() < 0.08:  # occasional longer "reading" pause
                await asyncio.sleep(random.uniform(2.5, 6.0))
            current = count_items()
            if current > last:
                idle = 0
                last = current
                if label:
                    self.progress(f"[{self.platform}] {label}: {current} items")
            else:
                idle += 1
                if idle >= rl.scroll_idle_rounds:
                    break
                # Nudge lazy loaders: jump to the bottom once in a while.
                if idle % 2 == 0:
                    await self.page.keyboard.press("End")
            if round_no % 25 == 0:
                await self.human_pause(3.0, 8.0)
        return last

    # ---------------------------------------------------------- in-page API
    async def page_fetch(
        self,
        url: str,
        headers: dict[str, str] | None = None,
        method: str = "GET",
        body: str | None = None,
    ) -> tuple[int, str]:
        """Same-origin fetch executed by the page (browser cookies apply)."""

        assert self.page is not None
        result = await self.page.evaluate(
            """async ({url, headers, method, body}) => {
                const csrf = (document.cookie.match(/(?:^|; )csrftoken=([^;]+)/) || [])[1];
                const h = Object.assign({}, headers || {});
                if (csrf && !h['X-CSRFToken']) h['X-CSRFToken'] = decodeURIComponent(csrf);
                const init = {method, headers: h, credentials: 'include'};
                if (body) init.body = body;
                try {
                    const r = await fetch(url, init);
                    return {status: r.status, text: await r.text()};
                } catch (e) {
                    return {status: 0, text: String(e)};
                }
            }""",
            {"url": url, "headers": headers or {}, "method": method, "body": body},
        )
        return int(result["status"]), str(result["text"])

    async def fetch_json(
        self,
        url: str,
        headers: dict[str, str] | None = None,
        classify: Callable[[int, Any], str | None] | None = None,
    ) -> Any:
        """``page_fetch`` + JSON decode + retry/backoff.

        ``classify(status, data)`` may return ``"rate_limited"``, ``"login"``,
        ``"not_found"`` or ``None`` (ok) to map platform-specific error shapes.
        """

        rl = self.settings.rate_limits
        last_error = ""
        for attempt in range(1, rl.max_retries + 1):
            status, text = await self.page_fetch(url, headers=headers)
            try:
                data = json.loads(text) if text else None
            except json.JSONDecodeError:
                data = None
            verdict = classify(status, data) if classify else None
            if verdict is None and status == 429:
                verdict = "rate_limited"
            if verdict == "login":
                from auth.session_manager import SessionExpiredError

                raise SessionExpiredError(self.platform, f"HTTP {status} from {url.split('?')[0]}")
            if verdict == "not_found":
                raise ProfileNotFoundError(f"{url} returned not found")
            if verdict == "rate_limited" or status >= 500 or status == 0:
                last_error = f"HTTP {status}"
                if attempt < rl.max_retries:
                    await self._backoff(attempt, f"rate limited/unavailable ({last_error})")
                    continue
                raise RateLimitedError(f"{url} kept failing: {last_error}")
            if status >= 400 or data is None:
                raise EndpointError(f"{url.split('?')[0]} -> HTTP {status}: {text[:200]}")
            return data
        raise RateLimitedError(f"{url} failed: {last_error}")

    # ------------------------------------------------------------ downloads
    async def download(
        self,
        url: str,
        dest: Path,
        referer: str | None = None,
        min_bytes: int = 512,
    ) -> Path | None:
        """Download ``url`` to ``dest`` through the browser context (cookies + UA)."""

        assert self.context is not None
        if dest.exists() and dest.stat().st_size >= min_bytes:
            return dest
        dest.parent.mkdir(parents=True, exist_ok=True)
        rl = self.settings.rate_limits
        headers = {"Referer": referer or f"{self.origin}/"}
        async with self._download_sem:
            for attempt in range(1, rl.max_retries + 1):
                try:
                    resp = await self.context.request.get(url, headers=headers, timeout=rl.download_timeout_s * 1000)
                    if resp.status == 200:
                        body = await resp.body()
                        if len(body) < min_bytes:
                            raise ScrapeError(f"suspiciously small body ({len(body)} bytes)")
                        tmp = dest.with_suffix(dest.suffix + ".part")
                        tmp.write_bytes(body)
                        tmp.replace(dest)
                        await asyncio.sleep(random.uniform(0.2, 0.8))
                        return dest
                    if resp.status in (403, 404, 410):
                        log.warning("Download %s -> HTTP %s (URL likely expired)", url[:80], resp.status)
                        return None
                    raise ScrapeError(f"HTTP {resp.status}")
                except (PlaywrightError, ScrapeError) as exc:
                    if attempt == rl.max_retries:
                        log.warning("Giving up on %s: %s", url[:80], exc)
                        return None
                    await asyncio.sleep(min(30, 2**attempt) * random.uniform(0.8, 1.2))
        return None

    async def download_assets(
        self,
        items: list[RawItem],
        resolver: Callable[[RawItem, MediaAsset], Awaitable[Path | None]],
        label: str,
    ) -> None:
        """Download every asset of ``items`` with bounded concurrency."""

        pending = [(item, asset) for item in items for asset in item.assets if asset.url and not asset.local_path]
        done = 0

        async def _one(item: RawItem, asset: MediaAsset) -> None:
            nonlocal done
            path = await resolver(item, asset)
            if path is not None:
                asset.local_path = str(path)
            done += 1
            if done % 10 == 0 or done == len(pending):
                self.progress(f"[{self.platform}] {label}: downloaded {done}/{len(pending)} files")

        await asyncio.gather(*(_one(item, asset) for item, asset in pending))

    # -------------------------------------------------------------- results
    def save_result(self, result: PlatformScrapeResult) -> Path:
        path = self.raw_dir / "scrape_result.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(result.model_dump_json(indent=1), encoding="utf-8")
        tmp.replace(path)
        return path

    def checkpoint_items(self, surface: Surface, items: list[RawItem]) -> None:
        self.cache.set(f"{self.ns}:surface_items", surface.value, [i.model_dump(mode="json") for i in items])

    def load_checkpoint(self, surface: Surface) -> list[RawItem] | None:
        data = self.cache.get(f"{self.ns}:surface_items", surface.value)
        if data is None:
            return None
        return [RawItem.model_validate(d) for d in data]

    def cap(self) -> int:
        return self.settings.scrape.max_items_per_surface


def load_scrape_result(settings: Settings, run_key: str, platform: Platform) -> PlatformScrapeResult | None:
    path = settings.raw_dir(run_key, platform) / "scrape_result.json"
    if not path.exists():
        return None
    return PlatformScrapeResult.model_validate_json(path.read_text(encoding="utf-8"))


def walk_json(node: Any) -> Iterable[dict]:
    """Yield every dict nested anywhere inside ``node`` (iterative, cycle-free)."""

    stack = [node]
    while stack:
        current = stack.pop()
        if isinstance(current, dict):
            yield current
            stack.extend(current.values())
        elif isinstance(current, list):
            stack.extend(current)
