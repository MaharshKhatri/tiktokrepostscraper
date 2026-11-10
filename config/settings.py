"""Central, typed configuration for the audit engine.

Every value can be overridden with an ``AUDIT_``-prefixed environment variable
or a line in ``.env`` (see ``.env.example``). Nested groups use a double
underscore, e.g. ``AUDIT_BROWSER__HEADLESS=false``.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent

Platform = Literal["instagram", "tiktok"]


class BrowserSettings(BaseModel):
    """Playwright browser/context options."""

    headless: bool = True
    executable_path: Path | None = Field(
        default=None,
        description="Use a specific Chromium binary instead of Playwright's bundled one.",
    )
    slow_mo_ms: int = 0
    locale: str = "en-US"
    timezone_id: str = "America/New_York"
    viewport_width: int = 1366
    viewport_height: int = 900
    user_agent: str | None = Field(
        default=None,
        description="Leave unset to use the browser's real user agent (recommended).",
    )
    navigation_timeout_ms: int = 45_000
    extra_args: list[str] = Field(default_factory=list)


class RateLimitSettings(BaseModel):
    """Pacing so the scraper behaves like a person browsing, not a crawler."""

    request_delay_min_s: float = 1.2
    request_delay_max_s: float = 3.5
    scroll_pause_min_s: float = 1.0
    scroll_pause_max_s: float = 2.6
    scroll_idle_rounds: int = Field(default=5, description="Stop scrolling after this many rounds with no new items.")
    scroll_max_rounds: int = 400
    max_retries: int = 4
    backoff_base_s: float = 4.0
    backoff_max_s: float = 120.0
    download_concurrency: int = 3
    download_timeout_s: float = 180.0


class ScrapeSettings(BaseModel):
    """What to collect and how much of it."""

    max_items_per_surface: int = Field(default=0, description="0 means no cap: collect every item on the surface.")
    ig_comments_per_post: int = 25
    ig_page_size: int = 12
    ig_app_id: str = "936619743392459"
    ig_highlight_batch_size: int = 3
    tiktok_captcha_wait_s: int = 180
    tiktok_repost_tab_selectors: list[str] = Field(
        default_factory=lambda: [
            '[data-e2e="repost-tab"]',
            '[data-e2e="reposts-tab"]',
            'p[role="tab"]:has-text("Reposts")',
            'div[role="tab"]:has-text("Reposts")',
            'button[role="tab"]:has-text("Reposts")',
        ]
    )
    tiktok_videos_tab_selectors: list[str] = Field(
        default_factory=lambda: [
            '[data-e2e="videos-tab"]',
            'p[role="tab"]:has-text("Videos")',
            'div[role="tab"]:has-text("Videos")',
        ]
    )
    download_media: bool = True
    save_raw_responses: bool = True


class MediaSettings(BaseModel):
    """ffmpeg keyframe extraction and contact-sheet rendering."""

    ffmpeg_bin: str = "ffmpeg"
    ffprobe_bin: str = "ffprobe"
    keyframes_min: int = 4
    keyframes_max: int = 6
    scene_threshold: float = 0.28
    scene_analysis_fps: float = 4.0
    min_keyframe_gap_s: float = 0.75
    contact_sheet_columns: int = 3
    contact_sheet_max_edge: int = Field(
        default=2400,
        description="Long edge of a contact sheet. Current vision models read up to ~2576px.",
    )
    image_max_edge: int = 2000
    carousel_sheet_max_slides: int = 12
    jpeg_quality: int = 88
    max_image_bytes: int = 3_600_000
    workers: int = 2


class WhisperSettings(BaseModel):
    """Local faster-whisper transcription."""

    enabled: bool = True
    model_size: str = "small"
    device: str = "auto"
    compute_type: str = "default"
    language: str | None = None
    beam_size: int = 5
    vad_filter: bool = True
    cpu_threads: int = 0
    download_root: Path | None = None


class AnthropicSettings(BaseModel):
    """Claude models and request shaping for the map-reduce synthesis."""

    api_key: SecretStr | None = Field(
        default=None,
        description="Optional. The SDK also reads ANTHROPIC_API_KEY or an `ant auth login` profile.",
    )
    vision_model: str = "claude-sonnet-5-5"
    synthesis_model: str = "claude-sonnet-5-5"
    vision_effort: Literal["low", "medium", "high", "xhigh", "max"] = "medium"
    synthesis_effort: Literal["low", "medium", "high", "xhigh", "max"] = "high"
    vision_max_tokens: int = 16_000
    digest_max_tokens: int = 16_000
    section_max_tokens: int = 32_000
    vision_items_per_batch: int = 8
    vision_max_images_per_request: int = 20
    vision_max_request_bytes: int = 24_000_000
    transcript_char_limit: int = 12_000
    digest_chunk_items: int = 120
    evidence_char_budget: int = 600_000
    concurrency: int = 3
    max_retries: int = 4
    request_timeout_s: float = 600.0
    refusal_fallbacks: bool = Field(
        default=True,
        description="Opt into server-side refusal fallbacks (fallbacks='default').",
    )
    fallback_beta: str = "server-side-fallback-2026-07-01"


class Settings(BaseSettings):
    """Top-level settings object. Load with :func:`get_settings`."""

    model_config = SettingsConfigDict(
        env_prefix="AUDIT_",
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )

    base_dir: Path = PROJECT_ROOT
    data_dir: Path = PROJECT_ROOT / "data"
    reports_dir: Path = PROJECT_ROOT / "reports"
    sessions_dir: Path = PROJECT_ROOT / "auth" / "sessions"
    cache_db: Path = PROJECT_ROOT / "data" / "cache.sqlite3"
    log_level: str = "INFO"

    browser: BrowserSettings = Field(default_factory=BrowserSettings)
    rate_limits: RateLimitSettings = Field(default_factory=RateLimitSettings)
    scrape: ScrapeSettings = Field(default_factory=ScrapeSettings)
    media: MediaSettings = Field(default_factory=MediaSettings)
    whisper: WhisperSettings = Field(default_factory=WhisperSettings)
    anthropic: AnthropicSettings = Field(default_factory=AnthropicSettings)

    @field_validator("data_dir", "reports_dir", "sessions_dir", "cache_db", mode="after")
    @classmethod
    def _absolute(cls, value: Path) -> Path:
        return value if value.is_absolute() else (PROJECT_ROOT / value).resolve()

    # ------------------------------------------------------------------ paths
    def session_file(self, platform: Platform) -> Path:
        name = "ig_state.json" if platform == "instagram" else "tiktok_state.json"
        return self.sessions_dir / name

    def tiktok_cookie_file(self) -> Path:
        return self.sessions_dir / "tiktok_cookies.txt"

    def run_dir(self, run_key: str) -> Path:
        return self.data_dir / safe_slug(run_key)

    def raw_dir(self, run_key: str, platform: Platform) -> Path:
        return self.run_dir(run_key) / "raw" / platform

    def media_dir(self, run_key: str, platform: Platform) -> Path:
        return self.run_dir(run_key) / "media" / platform

    def derived_dir(self, run_key: str) -> Path:
        return self.run_dir(run_key) / "derived"

    def manifest_path(self, run_key: str) -> Path:
        return self.run_dir(run_key) / "manifest.json"

    def report_path(self, run_key: str) -> Path:
        return self.reports_dir / f"{safe_slug(run_key)}_full_report.md"

    def ensure_dirs(self) -> None:
        for path in (self.data_dir, self.reports_dir, self.sessions_dir, self.cache_db.parent):
            path.mkdir(parents=True, exist_ok=True)


_SLUG_RE = re.compile(r"[^A-Za-z0-9._-]+")


def safe_slug(value: str) -> str:
    """Filesystem-safe version of a handle (``@Foo.Bar`` -> ``foo.bar``)."""

    cleaned = _SLUG_RE.sub("_", value.strip().lstrip("@")).strip("._-").lower()
    return cleaned or "unnamed"


def normalize_handle(value: str) -> str:
    """Strip URL/at-sign noise from a username passed on the command line."""

    value = value.strip()
    value = re.sub(r"^https?://(www\.)?(instagram|tiktok)\.com/", "", value, flags=re.I)
    value = value.strip("/").split("/")[0].split("?")[0]
    return value.lstrip("@")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_dirs()
    return settings
