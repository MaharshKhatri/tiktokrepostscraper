"""Map stage: batched Claude vision analysis of every media item.

Items are grouped by surface and packed into batches bounded by item count,
image count and request size. Each batch is one Messages API call containing,
per item, a text header (caption, metrics, comments, transcript...) followed by
that item's contact sheet or photo. The model returns a ``BatchObservation``
via structured outputs, so results are validated Pydantic objects.

This module also hosts :class:`ClaudeGateway`, the small wrapper both stages
use for concurrency limits, refusal handling, fallbacks and usage accounting.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

import anthropic
from pydantic import BaseModel

from config.prompts import (
    PROMPT_VERSION,
    VISION_BATCH_INSTRUCTIONS,
    VISION_SYSTEM_PROMPT,
)
from config.settings import Settings
from intelligence.schemas import (
    SURFACE_LABELS,
    BatchObservation,
    ItemObservation,
    MediaItem,
    ProfileManifest,
    Surface,
)
from storage.cache_manager import CacheManager, file_fingerprint, stable_hash

log = logging.getLogger(__name__)

ModelT = TypeVar("ModelT", bound=BaseModel)

# ---------------------------------------------------------------------------
# Gateway
# ---------------------------------------------------------------------------


@dataclass
class UsageTotals:
    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    refusals: int = 0
    fallbacks_used: int = 0

    def add(self, usage: Any) -> None:
        self.requests += 1
        if usage is None:
            return
        self.input_tokens += getattr(usage, "input_tokens", 0) or 0
        self.output_tokens += getattr(usage, "output_tokens", 0) or 0
        self.cache_read_input_tokens += getattr(usage, "cache_read_input_tokens", 0) or 0
        self.cache_creation_input_tokens += getattr(usage, "cache_creation_input_tokens", 0) or 0
        iterations = getattr(usage, "iterations", None) or []
        if any(getattr(entry, "type", None) == "fallback_message" for entry in iterations):
            self.fallbacks_used += 1


@dataclass
class ClaudeResult:
    text: str = ""
    parsed: Any = None
    stop_reason: str | None = None
    model: str | None = None
    refusal_category: str | None = None
    refusal_explanation: str | None = None

    @property
    def refused(self) -> bool:
        return self.stop_reason == "refusal"


class ClaudeGateway:
    """Thin async wrapper around ``anthropic.AsyncAnthropic``."""

    def __init__(self, settings: Settings, client: anthropic.AsyncAnthropic | None = None) -> None:
        self.settings = settings
        cfg = settings.anthropic
        if client is None:
            kwargs: dict[str, Any] = {"max_retries": cfg.max_retries, "timeout": cfg.request_timeout_s}
            if cfg.api_key is not None:
                kwargs["api_key"] = cfg.api_key.get_secret_value()
            client = anthropic.AsyncAnthropic(**kwargs)
        self.client = client
        self.sem = asyncio.Semaphore(max(1, cfg.concurrency))
        self.usage = UsageTotals()
        self.fallbacks_enabled = cfg.refusal_fallbacks

    def _fallback_kwargs(self) -> dict[str, Any]:
        if not self.fallbacks_enabled:
            return {}
        return {"betas": [self.settings.anthropic.fallback_beta], "fallbacks": "default"}

    @staticmethod
    def _refusal(result: ClaudeResult, message: Any) -> None:
        details = getattr(message, "stop_details", None)
        if details is not None:
            result.refusal_category = getattr(details, "category", None)
            result.refusal_explanation = getattr(details, "explanation", None)

    async def _with_fallback_retry(self, call: Callable[[dict[str, Any]], Any]) -> Any:
        try:
            return await call(self._fallback_kwargs())
        except anthropic.BadRequestError as exc:
            if self.fallbacks_enabled and "fallback" in str(exc).lower():
                log.warning("Server-side refusal fallbacks unavailable here; continuing without them")
                self.fallbacks_enabled = False
                return await call({})
            raise

    async def parse(
        self,
        *,
        model: str,
        system: str,
        content: list[dict[str, Any]],
        output_format: type[ModelT],
        max_tokens: int,
        effort: str,
    ) -> ClaudeResult:
        """Single structured-output request (returns ``parsed`` as ``output_format``)."""

        async def call(extra: dict[str, Any]) -> Any:
            return await self.client.beta.messages.parse(
                model=model,
                max_tokens=max_tokens,
                system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": content}],
                output_format=output_format,
                output_config={"effort": effort},
                **extra,
            )

        async with self.sem:
            message = await self._with_fallback_retry(call)
        self.usage.add(getattr(message, "usage", None))
        result = ClaudeResult(stop_reason=message.stop_reason, model=getattr(message, "model", model))
        if result.refused:
            self.usage.refusals += 1
            self._refusal(result, message)
            return result
        result.parsed = getattr(message, "parsed_output", None)
        result.text = "".join(getattr(b, "text", "") for b in message.content if getattr(b, "type", "") == "text")
        return result

    async def stream_text(
        self,
        *,
        model: str,
        system: str,
        content: list[dict[str, Any]],
        max_tokens: int,
        effort: str,
    ) -> ClaudeResult:
        """Streaming free-text request; returns the final assembled message."""

        async def call(extra: dict[str, Any]) -> Any:
            async with self.client.beta.messages.stream(
                model=model,
                max_tokens=max_tokens,
                system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": content}],
                output_config={"effort": effort},
                **extra,
            ) as stream:
                return await stream.get_final_message()

        async with self.sem:
            message = await self._with_fallback_retry(call)
        self.usage.add(getattr(message, "usage", None))
        result = ClaudeResult(stop_reason=message.stop_reason, model=getattr(message, "model", model))
        if result.refused:
            self.usage.refusals += 1
            self._refusal(result, message)
        result.text = "".join(getattr(b, "text", "") for b in message.content if getattr(b, "type", "") == "text")
        return result


# ---------------------------------------------------------------------------
# Batch planning
# ---------------------------------------------------------------------------


@dataclass
class VisionBatch:
    surface: Surface
    items: list[MediaItem]
    batch_no: int = 0
    batch_total: int = 0
    key: str = ""
    image_bytes: int = 0

    @property
    def image_count(self) -> int:
        return sum(len(i.derived.vision_images) for i in self.items)


def _image_size(path: str) -> int:
    try:
        return Path(path).stat().st_size
    except OSError:
        return 0


def plan_batches(items: list[MediaItem], settings: Settings) -> list[VisionBatch]:
    """Pack items (grouped by surface, newest first) into request-sized batches."""

    cfg = settings.anthropic
    batches: list[VisionBatch] = []
    for surface in Surface:
        surface_items = sorted(
            (i for i in items if i.surface == surface),
            key=lambda i: i.taken_at.timestamp() if i.taken_at else 0,
            reverse=True,
        )
        current: list[MediaItem] = []
        images = 0
        size = 0
        surface_batches: list[VisionBatch] = []
        for item in surface_items:
            n_img = len(item.derived.vision_images)
            n_bytes = int(sum(_image_size(p) for p in item.derived.vision_images) * 1.37)
            if current and (
                len(current) >= cfg.vision_items_per_batch
                or images + n_img > cfg.vision_max_images_per_request
                or size + n_bytes > cfg.vision_max_request_bytes
            ):
                surface_batches.append(VisionBatch(surface, current, image_bytes=size))
                current, images, size = [], 0, 0
            current.append(item)
            images += n_img
            size += n_bytes
        if current:
            surface_batches.append(VisionBatch(surface, current, image_bytes=size))
        for idx, batch in enumerate(surface_batches, 1):
            batch.batch_no, batch.batch_total = idx, len(surface_batches)
            batch.key = batch_cache_key(batch, settings)
        batches.extend(surface_batches)
    return batches


def batch_cache_key(batch: VisionBatch, settings: Settings) -> str:
    cfg = settings.anthropic
    parts = [
        (
            i.item_id,
            [file_fingerprint(p) for p in i.derived.vision_images],
            stable_hash(i.derived.transcript.text if i.derived.transcript else "", i.caption, i.metrics.model_dump()),
        )
        for i in batch.items
    ]
    return stable_hash(PROMPT_VERSION, cfg.vision_model, cfg.vision_effort, batch.surface.value, parts)


# ---------------------------------------------------------------------------
# Request building
# ---------------------------------------------------------------------------


def _fmt_num(value: int | None) -> str | None:
    return f"{value:,}" if value is not None else None


def render_item_header(item: MediaItem, settings: Settings) -> str:
    lines = [f"ITEM {item.item_id}"]
    who = f"@{item.author_handle}" if item.author_handle else "unknown"
    who += " (profile owner)" if item.author_is_profile_owner else " (another account)"
    kind = item.media_kind.value
    if item.duration_s:
        kind += f", {item.duration_s:.0f}s"
    when = item.taken_at.strftime("%Y-%m-%d %a") if item.taken_at else "unknown date"
    lines.append(f"surface: {item.surface.label} | author: {who} | posted: {when} | media: {kind}")
    if item.url:
        lines.append(f"url: {item.url}")
    if item.highlight_title:
        lines.append(f'highlight tray: "{item.highlight_title}"')
    if item.location and (item.location.city or item.location.name):
        lines.append(f"location label (city-level use only): {item.location.city or item.location.name}")
    if item.caption:
        lines.append(f'caption: """{item.caption.strip()}"""')
    if item.hashtags:
        lines.append("hashtags: " + " ".join(f"#{h}" for h in item.hashtags[:40]))
    if item.mentions:
        lines.append("mentions/tags: " + " ".join(f"@{m}" for m in item.mentions[:30]))
    if item.audio and (item.audio.title or item.audio.artist):
        origin = "original audio" if item.audio.is_original else "licensed/trending audio"
        lines.append(f'audio: "{item.audio.title or "?"}" by {item.audio.artist or "?"} ({origin})')
    metrics = {k: _fmt_num(v) for k, v in item.metrics.model_dump().items() if v is not None}
    if metrics:
        lines.append("metrics: " + " · ".join(f"{k} {v}" for k, v in metrics.items()))
    if item.is_paid_partnership:
        lines.append("platform paid-partnership / ad label: yes")
    if item.story_stickers:
        lines.append("stickers: " + "; ".join(item.story_stickers[:15]))
    if item.comments:
        lines.append("comment sample (text only):")
        for c in item.comments[:12]:
            tag = " [creator reply]" if c.by_profile_owner else ""
            likes = f" ({c.likes} likes)" if c.likes else ""
            lines.append(f'  - "{c.text.strip()[:280]}"{likes}{tag}')
    tr = item.derived.transcript
    if tr and tr.text:
        text = tr.text.strip()
        limit = settings.anthropic.transcript_char_limit
        if len(text) > limit:
            log.info("Transcript for %s truncated from %s to %s chars", item.item_id, len(text), limit)
            text = text[:limit] + " [transcript truncated]"
        lang = f", language {tr.language}" if tr.language else ""
        lines.append(f'speech transcript (automatic{lang}): """{text}"""')
    elif item.has_video:
        lines.append("speech transcript: none detected (music-only, silent, or not transcribed)")
    if item.derived.contact_sheet:
        n = len(item.derived.keyframe_times)
        desc = f"contact sheet of {n} keyframes" if n else "slide sheet"
        lines.append(f"attached image: {desc}")
    elif item.derived.images:
        lines.append("attached image: photo")
    else:
        lines.append("attached image: none (media unavailable - rely on text only, lower confidence)")
    if item.derived.errors:
        lines.append("processing notes: " + "; ".join(item.derived.errors[:3]))
    return "\n".join(lines)


def image_block(path: str) -> dict[str, Any]:
    data = base64.standard_b64encode(Path(path).read_bytes()).decode("ascii")
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": data}}


def build_batch_content(batch: VisionBatch, settings: Settings, handles: str) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    for item in batch.items:
        content.append({"type": "text", "text": render_item_header(item, settings)})
        for path in item.derived.vision_images:
            if Path(path).exists():
                content.append(image_block(path))
    content.append(
        {
            "type": "text",
            "text": VISION_BATCH_INSTRUCTIONS.format(
                n_items=len(batch.items),
                surface_label=SURFACE_LABELS[batch.surface],
                batch_no=batch.batch_no,
                batch_total=batch.batch_total,
                handles=handles,
            ),
        }
    )
    return content


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


@dataclass
class VisionOutcome:
    observations: dict[str, ItemObservation] = field(default_factory=dict)
    batch_patterns: dict[str, list[str]] = field(default_factory=dict)
    failed_items: list[str] = field(default_factory=list)
    refused_items: list[str] = field(default_factory=list)


def _reconcile_ids(batch: VisionBatch, parsed: BatchObservation) -> list[ItemObservation]:
    """Repair item ids the model altered (same count -> map by order)."""

    expected = [i.item_id for i in batch.items]
    got = [o.item_id for o in parsed.items]
    if set(got) <= set(expected):
        return parsed.items
    if len(got) == len(expected):
        fixed = []
        for obs, item_id in zip(parsed.items, expected, strict=True):
            fixed.append(obs.model_copy(update={"item_id": item_id}))
        return fixed
    return [o for o in parsed.items if o.item_id in set(expected)]


class VisionAnalyzer:
    def __init__(self, settings: Settings, cache: CacheManager, gateway: ClaudeGateway) -> None:
        self.settings = settings
        self.cache = cache
        self.gateway = gateway
        self._run_key = ""

    async def _analyze(self, batch: VisionBatch, handles: str, outcome: VisionOutcome, depth: int = 0) -> None:
        cache_ns = f"{self._run_key}:observations"
        cached = self.cache.get_model(cache_ns, batch.key, BatchObservation)
        if cached is not None:
            self._absorb(batch, cached, outcome)
            return
        if len(batch.items) > 1 and self.cache.has(f"{self._run_key}:batch_splits", batch.key):
            await self._split(batch, handles, outcome, depth)  # decided on an earlier run
            return
        cfg = self.settings.anthropic
        try:
            result = await self.gateway.parse(
                model=cfg.vision_model,
                system=VISION_SYSTEM_PROMPT,
                content=build_batch_content(batch, self.settings, handles),
                output_format=BatchObservation,
                max_tokens=cfg.vision_max_tokens,
                effort=cfg.vision_effort,
            )
        except anthropic.BadRequestError as exc:
            if len(batch.items) > 1:
                log.warning("Batch %s/%s rejected (%s); splitting", batch.surface.value, batch.batch_no, exc)
                await self._split(batch, handles, outcome, depth)
                return
            log.error("Item %s rejected by the API: %s", batch.items[0].item_id, exc)
            outcome.failed_items.append(batch.items[0].item_id)
            return
        except anthropic.APIError as exc:
            log.error("Batch %s/%s failed after retries: %s", batch.surface.value, batch.batch_no, exc)
            outcome.failed_items.extend(i.item_id for i in batch.items)
            return

        if result.refused:
            if len(batch.items) > 1:
                log.warning(
                    "Batch %s/%s declined (%s); isolating items",
                    batch.surface.value,
                    batch.batch_no,
                    result.refusal_category,
                )
                await self._split(batch, handles, outcome, depth)
            else:
                log.warning("Item %s declined by the model (%s)", batch.items[0].item_id, result.refusal_category)
                outcome.refused_items.append(batch.items[0].item_id)
            return
        if result.stop_reason == "max_tokens" or result.parsed is None:
            if len(batch.items) > 1:
                await self._split(batch, handles, outcome, depth)
            else:
                outcome.failed_items.append(batch.items[0].item_id)
            return

        parsed: BatchObservation = result.parsed
        parsed = parsed.model_copy(update={"items": _reconcile_ids(batch, parsed)})
        missing = [i for i in batch.items if i.item_id not in {o.item_id for o in parsed.items}]
        if missing and depth == 0:
            log.info(
                "Re-requesting %s item(s) omitted from batch %s/%s", len(missing), batch.surface.value, batch.batch_no
            )
            retry = VisionBatch(batch.surface, missing, batch.batch_no, batch.batch_total)
            retry.key = batch_cache_key(retry, self.settings)
            await self._analyze(retry, handles, outcome, depth + 1)
        elif missing:
            outcome.failed_items.extend(i.item_id for i in missing)
        self.cache.set(cache_ns, batch.key, parsed)
        self._absorb(batch, parsed, outcome)

    async def _split(self, batch: VisionBatch, handles: str, outcome: VisionOutcome, depth: int) -> None:
        self.cache.set(f"{self._run_key}:batch_splits", batch.key, True)
        mid = len(batch.items) // 2
        for part in (batch.items[:mid], batch.items[mid:]):
            sub = VisionBatch(batch.surface, part, batch.batch_no, batch.batch_total)
            sub.key = batch_cache_key(sub, self.settings)
            await self._analyze(sub, handles, outcome, depth + 1)

    @staticmethod
    def _absorb(batch: VisionBatch, parsed: BatchObservation, outcome: VisionOutcome) -> None:
        for obs in parsed.items:
            outcome.observations[obs.item_id] = obs
        if parsed.batch_patterns:
            outcome.batch_patterns.setdefault(batch.surface.value, []).extend(parsed.batch_patterns)

    async def analyze(
        self,
        manifest: ProfileManifest,
        progress: Callable[[int, int, str], None] | None = None,
    ) -> VisionOutcome:
        self._run_key = manifest.run_key
        batches = plan_batches(manifest.items, self.settings)
        handles = ", ".join(f"{p.platform} @{p.username}" for p in manifest.profiles) or manifest.run_key
        outcome = VisionOutcome()
        done = 0

        async def run(batch: VisionBatch) -> None:
            nonlocal done
            await self._analyze(batch, handles, outcome)
            done += 1
            if progress:
                progress(done, len(batches), f"{batch.surface.value} batch {batch.batch_no}/{batch.batch_total}")

        await asyncio.gather(*(run(b) for b in batches))
        self._persist(manifest, outcome)
        return outcome

    def _persist(self, manifest: ProfileManifest, outcome: VisionOutcome) -> None:
        path = self.settings.run_dir(manifest.run_key) / "observations.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        order = {item.item_id: idx for idx, item in enumerate(manifest.items)}
        with path.open("w", encoding="utf-8") as fh:
            for obs in sorted(outcome.observations.values(), key=lambda o: order.get(o.item_id, 1 << 30)):
                fh.write(json.dumps(obs.model_dump(mode="json"), ensure_ascii=False) + "\n")


def load_observations(path: Path) -> dict[str, ItemObservation]:
    out: dict[str, ItemObservation] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            obs = ItemObservation.model_validate_json(line)
            out[obs.item_id] = obs
    return out
