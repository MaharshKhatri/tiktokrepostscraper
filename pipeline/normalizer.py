"""Map raw Instagram + TikTok payloads onto the canonical ``ProfileManifest``.

Normalization is also where data minimization happens: GPS coordinates,
commenter identities and other incidental personal data present in the raw
platform payloads are dropped here and never reach the analysis stages.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config.settings import Settings
from intelligence.schemas import (
    AudioInfo,
    CommentSample,
    LocationTag,
    MediaItem,
    MediaKind,
    Metrics,
    PlatformProfile,
    PlatformScrapeResult,
    ProfileManifest,
    RawItem,
    Surface,
)

HASHTAG_RE = re.compile(r"#([\wÀ-￿]+)", re.UNICODE)
MENTION_RE = re.compile(r"@([A-Za-z0-9._]{2,30})")


def _ts(value: Any) -> datetime | None:
    try:
        if value in (None, "", 0):
            return None
        return datetime.fromtimestamp(int(value), tz=timezone.utc)
    except (TypeError, ValueError, OSError):
        return None


def _int(value: Any) -> int | None:
    try:
        return int(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _dedupe(values: Iterable[str]) -> list[str]:
    seen: dict[str, None] = {}
    for v in values:
        if v:
            seen.setdefault(v, None)
    return list(seen)


def parse_tags(text: str) -> tuple[list[str], list[str]]:
    hashtags = _dedupe(h.lower() for h in HASHTAG_RE.findall(text or ""))
    mentions = _dedupe(m.rstrip(".").lower() for m in MENTION_RE.findall(text or ""))
    return hashtags, mentions


def _kind_from_assets(raw_item: RawItem) -> MediaKind:
    indices = {a.index for a in raw_item.assets if a.role in ("primary", "slide")}
    if len(indices) > 1:
        return MediaKind.CAROUSEL
    if any(a.kind == "video" for a in raw_item.assets):
        return MediaKind.VIDEO
    return MediaKind.IMAGE


# ---------------------------------------------------------------------------
# Instagram
# ---------------------------------------------------------------------------


def _ig_audio(node: dict) -> AudioInfo | None:
    clips = node.get("clips_metadata") or {}
    music = (clips.get("music_info") or {}).get("music_asset_info") or {}
    if music:
        return AudioInfo(title=music.get("title"), artist=music.get("display_artist"), is_original=False)
    original = clips.get("original_sound_info") or {}
    if original:
        artist = (original.get("ig_artist") or {}).get("username")
        return AudioInfo(
            title=original.get("original_audio_title") or "Original audio", artist=artist, is_original=True
        )
    for sticker in node.get("story_music_stickers") or []:
        info = sticker.get("music_asset_info") or {}
        if info:
            return AudioInfo(title=info.get("title"), artist=info.get("display_artist"), is_original=False)
    return None


def _ig_stickers(node: dict) -> list[str]:
    out: list[str] = []
    for h in node.get("story_hashtags") or []:
        tag = (h.get("hashtag") or {}).get("name")
        if tag:
            out.append(f"hashtag sticker #{tag}")
    for loc in node.get("story_locations") or []:
        name = (loc.get("location") or {}).get("name")
        if name:
            out.append(f"location sticker: {name}")
    for m in node.get("reel_mentions") or []:
        user = (m.get("user") or {}).get("username")
        if user:
            out.append(f"mention sticker @{user}")
    for link in node.get("story_link_stickers") or []:
        url = (link.get("story_link") or {}).get("url")
        if url:
            out.append(f"link sticker: {url}")
    for poll in node.get("story_polls") or []:
        question = (poll.get("poll_sticker") or {}).get("question")
        if question:
            out.append(f"poll: {question}")
    for q in node.get("story_questions") or []:
        question = (q.get("question_sticker") or {}).get("question")
        if question:
            out.append(f"question box: {question}")
    for feed in node.get("story_feed_media") or []:
        out.append(f"reshared post {feed.get('media_code') or feed.get('media_id') or ''}".strip())
    for bloks in node.get("story_bloks_stickers") or []:
        data = (bloks.get("bloks_sticker") or {}).get("sticker_data") or {}
        ig_mention = data.get("ig_mention") or {}
        if ig_mention.get("username"):
            out.append(f"mention sticker @{ig_mention['username']}")
    return _dedupe(out)


def normalize_instagram_item(raw_item: RawItem, owner: str) -> MediaItem:
    node = raw_item.raw
    caption = (
        ((node.get("caption") or {}).get("text") or "")
        if isinstance(node.get("caption"), dict)
        else str(node.get("caption") or "")
    )
    hashtags, mentions = parse_tags(caption)
    for tag in (node.get("usertags") or {}).get("in") or []:
        user = (tag.get("user") or {}).get("username")
        if user:
            mentions.append(user.lower())
    stickers = _ig_stickers(node)
    for s in stickers:
        if s.startswith("mention sticker @"):
            mentions.append(s.removeprefix("mention sticker @").lower())
        if s.startswith("hashtag sticker #"):
            hashtags.append(s.removeprefix("hashtag sticker #").lower())
    location = None
    if node.get("location"):
        loc = node["location"]
        location = LocationTag(name=loc.get("name"), city=loc.get("city") or None)
    owner_user = node.get("user") or node.get("owner") or {}
    author = owner_user.get("username") or raw_item.extra.get("author") or owner
    shortcode = node.get("code") or raw_item.extra.get("shortcode")
    if raw_item.surface in (Surface.IG_STORY, Surface.IG_HIGHLIGHT):
        url = f"https://www.instagram.com/stories/{owner}/{raw_item.native_id}/"
    elif raw_item.surface == Surface.IG_REEL and shortcode:
        url = f"https://www.instagram.com/reel/{shortcode}/"
    else:
        url = f"https://www.instagram.com/p/{shortcode}/" if shortcode else None
    comments = [
        CommentSample(
            text=c.get("text", ""),
            likes=_int(c.get("likes")),
            created_at=_ts(c.get("created_at")),
            by_profile_owner=bool(c.get("by_owner")),
        )
        for c in raw_item.extra.get("comments") or []
        if c.get("text")
    ]
    views = _int(
        node.get("play_count") or node.get("ig_play_count") or node.get("view_count") or node.get("video_view_count")
    )
    return MediaItem(
        item_id=f"instagram:{raw_item.surface.value}:{raw_item.native_id}",
        platform="instagram",
        surface=raw_item.surface,
        native_id=raw_item.native_id,
        url=url,
        shortcode=shortcode,
        author_handle=author,
        author_is_profile_owner=(author or "").lower() == owner.lower(),
        author_is_verified=owner_user.get("is_verified", raw_item.extra.get("author_is_verified")),
        taken_at=_ts(node.get("taken_at")),
        media_kind=_kind_from_assets(raw_item),
        caption=caption,
        hashtags=_dedupe(hashtags),
        mentions=_dedupe(m for m in mentions if m != owner.lower()),
        location=location,
        audio=_ig_audio(node),
        metrics=Metrics(
            likes=_int(node.get("like_count")),
            comments=_int(node.get("comment_count")),
            views=views,
            shares=_int(node.get("reshare_count")),
        ),
        comments=comments,
        highlight_title=raw_item.extra.get("highlight_title"),
        story_stickers=stickers,
        is_paid_partnership=bool(node.get("is_paid_partnership") or node.get("sponsor_tags")) or None,
        product_type=node.get("product_type"),
        duration_s=node.get("video_duration"),
        assets=raw_item.assets,
    )


def normalize_instagram_profile(result: PlatformScrapeResult) -> PlatformProfile:
    p = result.profile_raw
    links = [link.get("url") for link in p.get("bio_links") or [] if link.get("url")]
    if p.get("external_url") and p["external_url"] not in links:
        links.insert(0, p["external_url"])
    return PlatformProfile(
        platform="instagram",
        username=p.get("username") or result.username,
        user_id=str(p.get("id") or p.get("pk") or "") or None,
        display_name=p.get("full_name"),
        biography=p.get("biography") or "",
        external_links=links,
        category=p.get("category_name") or p.get("business_category_name"),
        is_verified=p.get("is_verified"),
        is_private=p.get("is_private"),
        is_business=p.get("is_business_account") or p.get("is_professional_account"),
        followers=_int((p.get("edge_followed_by") or {}).get("count") or p.get("follower_count")),
        following=_int((p.get("edge_follow") or {}).get("count") or p.get("following_count")),
        posts_count=_int((p.get("edge_owner_to_timeline_media") or {}).get("count") or p.get("media_count")),
        avatar_url=p.get("profile_pic_url_hd") or p.get("profile_pic_url"),
        avatar_path=p.get("_avatar_path"),
        highlight_titles=p.get("_highlight_titles") or [],
        scraped_at=result.scraped_at,
        notes=result.notes,
    )


# ---------------------------------------------------------------------------
# TikTok
# ---------------------------------------------------------------------------


def normalize_tiktok_item(raw_item: RawItem, owner: str) -> MediaItem:
    node = raw_item.raw
    caption = node.get("desc") or ""
    hashtags, mentions = parse_tags(caption)
    for extra in node.get("textExtra") or []:
        if extra.get("hashtagName"):
            hashtags.append(extra["hashtagName"].lower())
        if extra.get("userUniqueId"):
            mentions.append(extra["userUniqueId"].lower())
    for ch in node.get("challenges") or []:
        if ch.get("title"):
            hashtags.append(ch["title"].lower())
    stickers = []
    for sticker in node.get("stickersOnItem") or []:
        for text in sticker.get("stickerText") or []:
            if text:
                stickers.append(f"text sticker: {text}")
    music = node.get("music") or {}
    stats = node.get("stats") or {}
    stats_v2 = node.get("statsV2") or {}
    author = node.get("author") or {}
    author_handle = author.get("uniqueId") or raw_item.extra.get("author") or owner
    video = node.get("video") or {}
    location = None
    if node.get("poi"):
        poi = node["poi"]
        location = LocationTag(name=poi.get("name"), city=poi.get("city") or poi.get("cityName"))
    return MediaItem(
        item_id=f"tiktok:{raw_item.surface.value}:{raw_item.native_id}",
        platform="tiktok",
        surface=raw_item.surface,
        native_id=raw_item.native_id,
        url=raw_item.extra.get("url"),
        author_handle=author_handle,
        author_is_profile_owner=(author_handle or "").lower() == owner.lower(),
        author_is_verified=author.get("verified", raw_item.extra.get("author_is_verified")),
        taken_at=_ts(node.get("createTime")),
        media_kind=_kind_from_assets(raw_item),
        caption=caption,
        hashtags=_dedupe(hashtags),
        mentions=_dedupe(m for m in mentions if m != owner.lower()),
        location=location,
        audio=AudioInfo(title=music.get("title"), artist=music.get("authorName"), is_original=music.get("original"))
        if music
        else None,
        metrics=Metrics(
            likes=_int(stats.get("diggCount") or stats_v2.get("diggCount")),
            comments=_int(stats.get("commentCount") or stats_v2.get("commentCount")),
            views=_int(stats.get("playCount") or stats_v2.get("playCount")),
            shares=_int(stats.get("shareCount") or stats_v2.get("shareCount")),
            saves=_int(stats.get("collectCount") or stats_v2.get("collectCount")),
        ),
        story_stickers=stickers,
        is_paid_partnership=bool(node.get("isAd") or (node.get("adAuthorization"))) or None,
        product_type="photo" if (node.get("imagePost") or {}).get("images") else "video",
        duration_s=_int(video.get("duration")),
        assets=raw_item.assets,
    )


def normalize_tiktok_profile(result: PlatformScrapeResult) -> PlatformProfile:
    info = result.profile_raw
    user = info.get("user") or {}
    stats = info.get("stats") or {}
    links = []
    bio_link = (user.get("bioLink") or {}).get("link")
    if bio_link:
        links.append(bio_link)
    return PlatformProfile(
        platform="tiktok",
        username=user.get("uniqueId") or result.username,
        user_id=str(user.get("id") or "") or None,
        display_name=user.get("nickname"),
        biography=user.get("signature") or "",
        external_links=links,
        category=(user.get("commerceUserInfo") or {}).get("category"),
        is_verified=user.get("verified"),
        is_private=user.get("privateAccount"),
        is_business=bool((user.get("commerceUserInfo") or {}).get("commerceUser")) or None,
        followers=_int(stats.get("followerCount")),
        following=_int(stats.get("followingCount")),
        posts_count=_int(stats.get("videoCount")),
        total_likes=_int(stats.get("heartCount") or stats.get("heart")),
        avatar_url=user.get("avatarLarger"),
        avatar_path=info.get("_avatar_path"),
        scraped_at=result.scraped_at,
        notes=result.notes,
    )


# ---------------------------------------------------------------------------
# Manifest assembly
# ---------------------------------------------------------------------------


def build_manifest(
    run_key: str,
    results: Iterable[PlatformScrapeResult],
) -> ProfileManifest:
    manifest = ProfileManifest(run_key=run_key)
    seen: set[str] = set()
    for result in results:
        if result.platform == "instagram":
            manifest.profiles.append(normalize_instagram_profile(result))
            normalize = normalize_instagram_item
        else:
            manifest.profiles.append(normalize_tiktok_profile(result))
            normalize = normalize_tiktok_item
        for raw_item in result.items:
            item = normalize(raw_item, result.username)
            if item.item_id in seen:
                continue
            seen.add(item.item_id)
            manifest.items.append(item)
        manifest.notes.extend(f"[{result.platform}] {n}" for n in result.notes)
    manifest.items.sort(key=lambda i: (i.platform, i.surface.value, -(i.taken_at.timestamp() if i.taken_at else 0)))
    return manifest


def save_manifest(manifest: ProfileManifest, settings: Settings) -> Path:
    path = settings.manifest_path(manifest.run_key)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(manifest.model_dump_json(indent=1), encoding="utf-8")
    tmp.replace(path)
    return path


def load_manifest(settings: Settings, run_key: str) -> ProfileManifest | None:
    path = settings.manifest_path(run_key)
    if not path.exists():
        return None
    return ProfileManifest.model_validate_json(path.read_text(encoding="utf-8"))
