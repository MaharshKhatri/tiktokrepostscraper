"""Pydantic models shared by every stage of the engine.

Three families live here:

* **Ingestion / canonical records** - ``MediaAsset``, ``RawItem``,
  ``PlatformScrapeResult`` (what scrapers persist) and ``MediaItem``,
  ``PlatformProfile``, ``ProfileManifest`` (the normalized, cross-platform view).
* **Map-stage outputs** - ``ItemObservation`` / ``BatchObservation``: what the
  vision model returns for each batch of media. These are sent to the API as
  structured-output schemas, so they avoid free-form dict fields.
* **Reduce-stage outputs** - ``SurfaceDigest`` (structured) and
  ``ReportSection`` / ``PersonaReport`` (rendered Markdown).
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class Surface(str, Enum):
    """Where on a profile an item was found."""

    IG_POST = "ig_post"
    IG_REEL = "ig_reel"
    IG_STORY = "ig_story"
    IG_HIGHLIGHT = "ig_highlight"
    IG_TAGGED = "ig_tagged"
    TT_VIDEO = "tt_video"
    TT_REPOST = "tt_repost"

    @property
    def label(self) -> str:
        return SURFACE_LABELS[self]

    @property
    def is_first_party(self) -> bool:
        """True when the profile owner authored the item."""

        return self not in (Surface.IG_TAGGED, Surface.TT_REPOST)


SURFACE_LABELS: dict[Surface, str] = {
    Surface.IG_POST: "Instagram feed posts",
    Surface.IG_REEL: "Instagram Reels",
    Surface.IG_STORY: "Instagram active Stories",
    Surface.IG_HIGHLIGHT: "Instagram Story Highlights",
    Surface.IG_TAGGED: "Instagram Tagged tab (posted by others)",
    Surface.TT_VIDEO: "TikTok original posts",
    Surface.TT_REPOST: "TikTok Reposts tab (amplified from others)",
}


class MediaKind(str, Enum):
    IMAGE = "image"
    VIDEO = "video"
    CAROUSEL = "carousel"


# ---------------------------------------------------------------------------
# Ingestion records (scrapers -> disk)
# ---------------------------------------------------------------------------


class MediaAsset(BaseModel):
    """A single downloadable file belonging to an item (one image or one video)."""

    kind: Literal["image", "video"]
    url: str | None = None
    local_path: str | None = None
    width: int | None = None
    height: int | None = None
    duration_s: float | None = None
    role: Literal["primary", "slide", "cover"] = "primary"
    index: int = 0


class RawItem(BaseModel):
    """An item exactly as the platform returned it, plus its asset references."""

    surface: Surface
    native_id: str
    raw: dict
    assets: list[MediaAsset] = Field(default_factory=list)
    extra: dict = Field(default_factory=dict)


class PlatformScrapeResult(BaseModel):
    """Everything collected from one platform for one handle."""

    platform: Literal["instagram", "tiktok"]
    username: str
    scraped_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    profile_raw: dict = Field(default_factory=dict)
    items: list[RawItem] = Field(default_factory=list)
    surfaces_completed: list[Surface] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    def by_surface(self, surface: Surface) -> list[RawItem]:
        return [item for item in self.items if item.surface == surface]


# ---------------------------------------------------------------------------
# Canonical, cross-platform records
# ---------------------------------------------------------------------------


class Metrics(BaseModel):
    likes: int | None = None
    comments: int | None = None
    views: int | None = None
    shares: int | None = None
    saves: int | None = None


class AudioInfo(BaseModel):
    title: str | None = None
    artist: str | None = None
    is_original: bool | None = None


class LocationTag(BaseModel):
    """Venue/city label only. Coordinates are intentionally never stored."""

    name: str | None = None
    city: str | None = None


class CommentSample(BaseModel):
    """Comment text without the commenter's identity (data minimization)."""

    text: str
    likes: int | None = None
    created_at: datetime | None = None
    by_profile_owner: bool = False


class Transcript(BaseModel):
    text: str = ""
    language: str | None = None
    language_probability: float | None = None
    duration_s: float | None = None
    segments: list[tuple[float, float, str]] = Field(default_factory=list)


class DerivedMedia(BaseModel):
    """Artifacts produced by the local distillation pipeline."""

    contact_sheet: str | None = None
    images: list[str] = Field(default_factory=list)
    keyframe_times: list[float] = Field(default_factory=list)
    audio_path: str | None = None
    transcript: Transcript | None = None
    duration_s: float | None = None
    has_audio: bool | None = None
    errors: list[str] = Field(default_factory=list)

    @property
    def vision_images(self) -> list[str]:
        if self.contact_sheet:
            return [self.contact_sheet]
        return self.images


class MediaItem(BaseModel):
    """One canonical post / story / highlight slide / repost."""

    item_id: str = Field(description="Stable id: '<platform>:<surface>:<native id>'")
    platform: Literal["instagram", "tiktok"]
    surface: Surface
    native_id: str
    url: str | None = None
    shortcode: str | None = None
    author_handle: str | None = None
    author_is_profile_owner: bool = True
    author_is_verified: bool | None = None
    taken_at: datetime | None = None
    media_kind: MediaKind
    caption: str = ""
    hashtags: list[str] = Field(default_factory=list)
    mentions: list[str] = Field(default_factory=list)
    location: LocationTag | None = None
    audio: AudioInfo | None = None
    metrics: Metrics = Field(default_factory=Metrics)
    comments: list[CommentSample] = Field(default_factory=list)
    highlight_title: str | None = None
    story_stickers: list[str] = Field(default_factory=list)
    is_paid_partnership: bool | None = None
    product_type: str | None = None
    duration_s: float | None = None
    assets: list[MediaAsset] = Field(default_factory=list)
    derived: DerivedMedia = Field(default_factory=DerivedMedia)

    @property
    def has_video(self) -> bool:
        return any(asset.kind == "video" and asset.local_path for asset in self.assets)


class PlatformProfile(BaseModel):
    platform: Literal["instagram", "tiktok"]
    username: str
    user_id: str | None = None
    display_name: str | None = None
    biography: str = ""
    external_links: list[str] = Field(default_factory=list)
    category: str | None = None
    is_verified: bool | None = None
    is_private: bool | None = None
    is_business: bool | None = None
    followers: int | None = None
    following: int | None = None
    posts_count: int | None = None
    total_likes: int | None = None
    avatar_url: str | None = None
    avatar_path: str | None = None
    highlight_titles: list[str] = Field(default_factory=list)
    scraped_at: datetime | None = None
    notes: list[str] = Field(default_factory=list)


class ProfileManifest(BaseModel):
    """The unified dataset every downstream stage consumes."""

    run_key: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    profiles: list[PlatformProfile] = Field(default_factory=list)
    items: list[MediaItem] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    def profile(self, platform: str) -> PlatformProfile | None:
        return next((p for p in self.profiles if p.platform == platform), None)

    def items_for(self, surface: Surface) -> list[MediaItem]:
        return [item for item in self.items if item.surface == surface]

    def surface_counts(self) -> dict[Surface, int]:
        counts: dict[Surface, int] = {}
        for item in self.items:
            counts[item.surface] = counts.get(item.surface, 0) + 1
        return counts


# ---------------------------------------------------------------------------
# Map stage: per-item and per-batch observations (structured outputs)
# ---------------------------------------------------------------------------


class VisualAesthetic(BaseModel):
    color_palette: list[str] = Field(description="Dominant colors / grading, e.g. 'warm beige', 'teal-orange grade'.")
    lighting: str = Field(description="e.g. 'soft natural window light', 'ring light', 'harsh flash'.")
    composition: str = Field(description="Framing and camera work, e.g. 'centered selfie, handheld'.")
    editing_style: str = Field(description="Cuts, transitions, text treatment, filters, pacing.")
    mood: str = Field(description="Overall feel, e.g. 'cozy minimal', 'high-energy chaotic'.")


class BrandMention(BaseModel):
    name: str
    context: Literal["visible", "spoken", "caption", "tagged", "sponsored", "other"]
    evidence: str = Field(description="Short note on where/how the brand appears.")


class ItemObservation(BaseModel):
    item_id: str = Field(description="Copy the item id exactly as given in the input.")
    summary: str = Field(description="2-4 factual sentences on what is shown and said.")
    content_format: str = Field(
        description="e.g. talking-head, tutorial, GRWM, vlog, skit, photo dump, product review, meme, lip-sync."
    )
    content_pillars: list[str] = Field(description="Topic themes this item belongs to.")
    setting: str = Field(description="Generic setting type only, e.g. 'home interior', 'gym', 'beach', 'studio'.")
    visual_aesthetic: VisualAesthetic
    on_screen_text: list[str] = Field(description="Verbatim text overlays / captions burned into the media.")
    voice_and_tone: str | None = Field(default=None, description="How the creator speaks or writes in this item.")
    persona_signals: list[str] = Field(description="How the creator presents themselves (humor, expertise, warmth...).")
    interest_signals: list[str] = Field(description="Publicly presented hobbies, activities and interests.")
    brands_and_products: list[BrandMention]
    commercial_signals: list[str] = Field(description="#ad, paid-partnership labels, discount codes, affiliate CTAs...")
    production_quality: Literal["lo-fi", "casual", "polished", "professional"]
    hook_technique: str | None = Field(default=None, description="For video: how the first seconds grab attention.")
    audience_response: str | None = Field(
        default=None, description="What the metrics/comments suggest about reception."
    )
    brand_safety_flags: list[str] = Field(description="Potential brand-safety concerns; empty if none.")
    notable_quotes: list[str] = Field(description="Short verbatim quotes from transcript/caption worth citing.")
    confidence: Literal["low", "medium", "high"]


class BatchObservation(BaseModel):
    items: list[ItemObservation]
    batch_patterns: list[str] = Field(description="Patterns that recur across the items in this batch.")


# ---------------------------------------------------------------------------
# Reduce stage
# ---------------------------------------------------------------------------


class StandoutItem(BaseModel):
    item_id: str
    why: str


class SurfaceDigest(BaseModel):
    surface: str
    item_count: int
    overview: str = Field(description="One dense paragraph characterizing this surface.")
    recurring_themes: list[str]
    format_patterns: list[str]
    aesthetic_patterns: list[str]
    voice_and_persona_patterns: list[str]
    interest_and_lifestyle_patterns: list[str]
    brand_and_commercial_patterns: list[str]
    audience_response_patterns: list[str]
    brand_safety_notes: list[str]
    standout_items: list[StandoutItem]
    notable_quotes: list[str]


class ReportSection(BaseModel):
    section_id: str
    number: int
    title: str
    markdown: str
    model: str | None = None
    stop_reason: str | None = None
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class PersonaReport(BaseModel):
    run_key: str
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    sections: list[ReportSection]
    appendix_markdown: str = ""
    models: dict[str, str] = Field(default_factory=dict)
