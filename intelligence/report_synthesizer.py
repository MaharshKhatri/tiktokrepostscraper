"""Reduce stage: per-surface digests -> 10-section Markdown report.

Pipeline:

1. **Statistics** are computed locally from the manifest (cadence, engagement,
   hashtags, audio, coverage) - the model is given numbers, never asked to count.
2. **Surface digests**: each surface's item observations are condensed into a
   structured ``SurfaceDigest`` (chunked + merged for very large surfaces).
3. **Evidence pack**: profiles + statistics + digests + item observations,
   sent once as a cached prompt prefix.
4. **Sections 2-10** are written one request each against that cached prefix
   (the first request warms the cache, the rest run concurrently), then the
   **Executive Summary** is written from the finished sections.
5. The report is rendered with appendices (coverage, statistics, methodology)
   and an item index CSV for citation lookups.
"""

from __future__ import annotations

import asyncio
import csv
import json
import logging
import statistics
from collections import Counter
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config.prompts import (
    ALL_SECTIONS,
    DIGEST_INSTRUCTIONS,
    DIGEST_MERGE_INSTRUCTIONS,
    DIGEST_SYSTEM_PROMPT,
    EVIDENCE_PREAMBLE,
    EXECUTIVE_SUMMARY,
    EXECUTIVE_SUMMARY_REQUEST,
    PROMPT_VERSION,
    SECTION_SPECS,
    SECTION_SYSTEM_PROMPT,
    SectionSpec,
)
from config.settings import Settings
from intelligence.schemas import (
    SURFACE_LABELS,
    ItemObservation,
    MediaItem,
    PersonaReport,
    ProfileManifest,
    ReportSection,
    Surface,
    SurfaceDigest,
)
from intelligence.vision_analyzer import ClaudeGateway, VisionOutcome
from storage.cache_manager import CacheManager, stable_hash

log = logging.getLogger(__name__)

ProgressFn = Callable[[str], None]

# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

_CADENCE_SURFACES = (Surface.IG_POST, Surface.IG_REEL, Surface.TT_VIDEO)


def _median(values: list[int | float]) -> float | None:
    clean = [v for v in values if v is not None]
    return round(float(statistics.median(clean)), 2) if clean else None


def _mean(values: list[int | float]) -> float | None:
    clean = [v for v in values if v is not None]
    return round(float(statistics.fmean(clean)), 2) if clean else None


def surface_stats(items: list[MediaItem], followers: int | None) -> dict[str, Any]:
    dated = sorted(i.taken_at for i in items if i.taken_at)
    likes = [i.metrics.likes for i in items if i.metrics.likes is not None]
    comments = [i.metrics.comments for i in items if i.metrics.comments is not None]
    views = [i.metrics.views for i in items if i.metrics.views is not None]
    shares = [i.metrics.shares for i in items if i.metrics.shares is not None]
    saves = [i.metrics.saves for i in items if i.metrics.saves is not None]
    stats: dict[str, Any] = {
        "items": len(items),
        "media_kinds": dict(Counter(i.media_kind.value for i in items)),
        "first_date": dated[0].date().isoformat() if dated else None,
        "last_date": dated[-1].date().isoformat() if dated else None,
        "with_transcript": sum(1 for i in items if i.derived.transcript and i.derived.transcript.text),
        "with_visuals": sum(1 for i in items if i.derived.vision_images),
        "paid_partnership_labeled": sum(1 for i in items if i.is_paid_partnership),
        "median_likes": _median(likes),
        "median_comments": _median(comments),
        "median_views": _median(views),
        "median_shares": _median(shares),
        "median_saves": _median(saves),
        "mean_likes": _mean(likes),
        "mean_views": _mean(views),
    }
    if followers and items and items[0].author_is_profile_owner:
        rates = [
            ((i.metrics.likes or 0) + (i.metrics.comments or 0)) / followers * 100
            for i in items
            if i.metrics.likes is not None
        ]
        stats["median_engagement_rate_pct"] = _median(rates)
        if views:
            stats["median_views_to_followers_pct"] = round(_median(views) / followers * 100, 2)  # type: ignore[operator]
    if len(dated) >= 2:
        span_days = max(1.0, (dated[-1] - dated[0]).total_seconds() / 86400)
        stats["posts_per_week"] = round(len(dated) / (span_days / 7), 2)
        stats["active_weeks"] = len({d.isocalendar()[:2] for d in dated})
        gaps = [(b - a).total_seconds() / 86400 for a, b in zip(dated, dated[1:], strict=False)]
        stats["median_days_between_posts"] = _median(gaps)
        stats["longest_gap_days"] = round(max(gaps), 1)
    if dated:
        stats["weekday_distribution_utc"] = dict(Counter(d.strftime("%a") for d in dated).most_common())
    tags = Counter(h for i in items for h in i.hashtags)
    stats["top_hashtags"] = [f"#{t} ({n})" for t, n in tags.most_common(15)]
    mentions = Counter(m for i in items for m in i.mentions)
    stats["top_mentions"] = [f"@{m} ({n})" for m, n in mentions.most_common(15)]
    audio = Counter(
        f"{i.audio.title} - {i.audio.artist}" for i in items if i.audio and i.audio.title and not i.audio.is_original
    )
    stats["top_licensed_audio"] = [f"{a} ({n})" for a, n in audio.most_common(8)]
    stats["original_audio_share"] = (
        round(sum(1 for i in items if i.audio and i.audio.is_original) / len(items), 2) if items else None
    )
    key = (lambda i: i.metrics.views or 0) if views else (lambda i: i.metrics.likes or 0)
    metric = "views" if views else "likes"
    ranked = sorted((i for i in items if (i.metrics.views or i.metrics.likes)), key=key, reverse=True)
    stats[f"top_by_{metric}"] = [
        {
            "item_id": i.item_id,
            "url": i.url,
            metric: key(i),
            "date": i.taken_at.date().isoformat() if i.taken_at else None,
        }
        for i in ranked[:5]
    ]
    if len(ranked) > 8:
        stats[f"bottom_by_{metric}"] = [{"item_id": i.item_id, metric: key(i)} for i in ranked[-3:]]
    if items and items[0].surface == Surface.IG_HIGHLIGHT:
        trays = Counter(i.highlight_title or "(untitled)" for i in items)
        stats["highlight_trays"] = [f"{t} ({n} slides)" for t, n in trays.most_common()]
    if items and not items[0].author_is_profile_owner:
        authors = Counter(i.author_handle for i in items if i.author_handle)
        verified = {i.author_handle for i in items if i.author_is_verified}
        stats["top_source_accounts"] = [
            f"@{a} ({n}{', verified' if a in verified else ''})" for a, n in authors.most_common(12)
        ]
    return stats


def compute_stats(manifest: ProfileManifest) -> dict[str, Any]:
    followers = {p.platform: p.followers for p in manifest.profiles}
    out: dict[str, Any] = {"surfaces": {}, "platforms": {}}
    for surface in Surface:
        items = manifest.items_for(surface)
        if items:
            platform = "instagram" if surface.value.startswith("ig_") else "tiktok"
            out["surfaces"][surface.value] = surface_stats(items, followers.get(platform))
    for profile in manifest.profiles:
        own = [i for i in manifest.items if i.platform == profile.platform and i.surface in _CADENCE_SURFACES]
        out["platforms"][profile.platform] = {
            "username": profile.username,
            "followers": profile.followers,
            "following": profile.following,
            "posts_count_reported": profile.posts_count,
            "total_likes": profile.total_likes,
            "original_items_collected": len(own),
            "cadence_and_engagement": surface_stats(own, profile.followers) if own else {},
        }
    return out


# ---------------------------------------------------------------------------
# Compact observation rendering
# ---------------------------------------------------------------------------


def compact_observation(obs: ItemObservation, item: MediaItem | None) -> dict[str, Any]:
    data = obs.model_dump(mode="json", exclude_none=True)
    data = {k: v for k, v in data.items() if v not in ([], "", None)}
    if item is not None:
        data["surface"] = item.surface.value
        if item.taken_at:
            data["date"] = item.taken_at.date().isoformat()
        if item.url:
            data["url"] = item.url
        metrics = {k: v for k, v in item.metrics.model_dump().items() if v is not None}
        if metrics:
            data["metrics"] = metrics
        if not item.author_is_profile_owner and item.author_handle:
            data["author"] = item.author_handle + (" (verified)" if item.author_is_verified else "")
        if item.highlight_title:
            data["highlight"] = item.highlight_title
    return data


def _jsonl(rows: list[dict[str, Any]]) -> str:
    return "\n".join(json.dumps(r, ensure_ascii=False, separators=(",", ":")) for r in rows)


# ---------------------------------------------------------------------------
# Synthesizer
# ---------------------------------------------------------------------------


class ReportSynthesizer:
    def __init__(self, settings: Settings, cache: CacheManager, gateway: ClaudeGateway) -> None:
        self.settings = settings
        self.cache = cache
        self.gateway = gateway

    # ------------------------------------------------------------- digests
    async def build_digests(
        self,
        manifest: ProfileManifest,
        outcome: VisionOutcome,
        stats: dict[str, Any],
        progress: ProgressFn,
    ) -> dict[str, SurfaceDigest]:
        by_id = {i.item_id: i for i in manifest.items}
        tasks = []
        for surface in Surface:
            rows = [
                compact_observation(obs, by_id.get(obs.item_id))
                for obs in outcome.observations.values()
                if by_id.get(obs.item_id) and by_id[obs.item_id].surface == surface
            ]
            if rows:
                tasks.append(
                    self._digest_surface(
                        manifest.run_key,
                        surface,
                        rows,
                        stats["surfaces"].get(surface.value, {}),
                        outcome.batch_patterns.get(surface.value, []),
                        progress,
                    )
                )
        results = await asyncio.gather(*tasks)
        return {d.surface: d for d in results if d is not None}

    async def _digest_surface(
        self,
        run_key: str,
        surface: Surface,
        rows: list[dict[str, Any]],
        stats: dict[str, Any],
        batch_patterns: list[str],
        progress: ProgressFn,
    ) -> SurfaceDigest | None:
        cfg = self.settings.anthropic
        chunk = max(10, cfg.digest_chunk_items)
        chunks = [rows[i : i + chunk] for i in range(0, len(rows), chunk)]
        stats_payload = dict(stats)
        if batch_patterns:
            stats_payload["patterns_noted_during_batch_analysis"] = batch_patterns[:40]
        partials: list[SurfaceDigest] = []
        for idx, part in enumerate(chunks, 1):
            note = f" (chunk {idx} of {len(chunks)})" if len(chunks) > 1 else ""
            prompt = DIGEST_INSTRUCTIONS.format(
                surface_label=SURFACE_LABELS[surface],
                n_items=len(part),
                chunk_note=note,
                stats_json=json.dumps(stats_payload, ensure_ascii=False, indent=1),
                observations=_jsonl(part),
            )
            digest = await self._cached_parse(f"{run_key}:digests", prompt, DIGEST_SYSTEM_PROMPT)
            if digest is not None:
                partials.append(digest)
        if not partials:
            return None
        if len(partials) == 1:
            digest = partials[0]
        else:
            prompt = DIGEST_MERGE_INSTRUCTIONS.format(
                surface_label=SURFACE_LABELS[surface],
                n_parts=len(partials),
                n_items=len(rows),
                digests=json.dumps([p.model_dump(mode="json") for p in partials], ensure_ascii=False),
            )
            digest = await self._cached_parse(f"{run_key}:digests", prompt, DIGEST_SYSTEM_PROMPT) or partials[0]
        digest = digest.model_copy(update={"surface": surface.value, "item_count": len(rows)})
        progress(f"digest ready: {SURFACE_LABELS[surface]} ({len(rows)} items)")
        return digest

    async def _cached_parse(self, namespace: str, prompt: str, system: str) -> SurfaceDigest | None:
        cfg = self.settings.anthropic
        key = stable_hash(PROMPT_VERSION, cfg.synthesis_model, cfg.synthesis_effort, system, prompt)
        cached = self.cache.get_model(namespace, key, SurfaceDigest)
        if cached is not None:
            return cached
        result = await self.gateway.parse(
            model=cfg.synthesis_model,
            system=system,
            content=[{"type": "text", "text": prompt}],
            output_format=SurfaceDigest,
            max_tokens=cfg.digest_max_tokens,
            effort=cfg.synthesis_effort,
        )
        if result.refused or result.parsed is None:
            log.warning(
                "Digest request failed (stop_reason=%s, category=%s)", result.stop_reason, result.refusal_category
            )
            return None
        self.cache.set(namespace, key, result.parsed)
        return result.parsed

    # ------------------------------------------------------------ evidence
    def build_evidence(
        self,
        manifest: ProfileManifest,
        stats: dict[str, Any],
        digests: dict[str, SurfaceDigest],
        outcome: VisionOutcome,
    ) -> str:
        by_id = {i.item_id: i for i in manifest.items}
        order = {item.item_id: idx for idx, item in enumerate(manifest.items)}
        rows = [
            compact_observation(obs, by_id.get(obs.item_id))
            for obs in sorted(outcome.observations.values(), key=lambda o: order.get(o.item_id, 1 << 30))
        ]
        profiles = [
            p.model_dump(mode="json", exclude={"avatar_url", "avatar_path", "user_id"}, exclude_none=True)
            for p in manifest.profiles
        ]
        base = EVIDENCE_PREAMBLE.format(
            run_key=manifest.run_key,
            generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            platforms=", ".join(f"{p.platform} @{p.username}" for p in manifest.profiles),
            profiles_json=json.dumps(profiles, ensure_ascii=False, indent=1),
            stats_json=json.dumps(stats, ensure_ascii=False, indent=1),
            digests_json=json.dumps(
                [d.model_dump(mode="json") for d in digests.values()], ensure_ascii=False, indent=1
            ),
            n_observations=len(rows),
            sample_note="{sample_note}",
            observations="{observations}",
        )
        budget = self.settings.anthropic.evidence_char_budget - len(base)
        body = _jsonl(rows)
        sample_note = ""
        if len(body) > budget:
            rows = self._sample_rows(rows, budget)
            body = _jsonl(rows)
            sample_note = f' sampled="{len(rows)} most informative items; digests cover the rest"'
            log.info("Evidence pack sampled to %s observations to fit the character budget", len(rows))
        return base.replace("{sample_note}", sample_note).replace("{observations}", body)

    @staticmethod
    def _sample_rows(rows: list[dict[str, Any]], budget: int) -> list[dict[str, Any]]:
        """Keep top performers per surface, then fill round-robin across surfaces."""

        def perf(r: dict[str, Any]) -> int:
            m = r.get("metrics") or {}
            return int(m.get("views") or m.get("likes") or 0)

        by_surface: dict[str, list[dict[str, Any]]] = {}
        for r in rows:
            by_surface.setdefault(r.get("surface", "?"), []).append(r)
        queues = {s: sorted(rs, key=perf, reverse=True) for s, rs in by_surface.items()}
        picked: list[dict[str, Any]] = []
        used = 0
        while any(queues.values()):
            for surface in list(queues):
                if not queues[surface]:
                    continue
                row = queues[surface].pop(0)
                size = len(json.dumps(row, ensure_ascii=False)) + 1
                if used + size > budget:
                    queues[surface] = []
                    continue
                picked.append(row)
                used += size
        return picked

    # ------------------------------------------------------------ sections
    async def _write_section(self, spec: SectionSpec, evidence: str, request: str, run_key: str) -> ReportSection:
        cfg = self.settings.anthropic
        ns = f"{run_key}:sections"
        key = stable_hash(
            PROMPT_VERSION,
            cfg.synthesis_model,
            cfg.synthesis_effort,
            spec.section_id,
            stable_hash(evidence),
            stable_hash(request),
        )
        cached = self.cache.get_model(ns, key, ReportSection)
        if cached is not None:
            return cached
        result = await self.gateway.stream_text(
            model=cfg.synthesis_model,
            system=SECTION_SYSTEM_PROMPT,
            content=[
                {"type": "text", "text": evidence, "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": request},
            ],
            max_tokens=cfg.section_max_tokens,
            effort=cfg.synthesis_effort,
        )
        if result.refused or not result.text.strip():
            reason = result.refusal_category or result.stop_reason or "empty response"
            log.warning("Section %s not generated (%s)", spec.title, reason)
            return ReportSection(
                section_id=spec.section_id,
                number=spec.number,
                title=spec.title,
                markdown=f"_This section could not be generated ({reason}). Re-run `audit report` to retry._",
                model=result.model,
                stop_reason=result.stop_reason,
            )
        text = result.text.strip()
        if result.stop_reason == "max_tokens":
            text += "\n\n_(Section truncated at the output limit.)_"
        section = ReportSection(
            section_id=spec.section_id,
            number=spec.number,
            title=spec.title,
            markdown=text,
            model=result.model,
            stop_reason=result.stop_reason,
        )
        if result.stop_reason in ("end_turn", "stop_sequence"):
            self.cache.set(ns, key, section)
        return section

    async def write_sections(self, run_key: str, evidence: str, progress: ProgressFn) -> list[ReportSection]:
        specs = list(SECTION_SPECS)
        first = await self._write_section(specs[0], evidence, specs[0].render_request(), run_key)
        progress(f"section {first.number}/10 written: {first.title}")

        async def run(spec: SectionSpec) -> ReportSection:
            section = await self._write_section(spec, evidence, spec.render_request(), run_key)
            progress(f"section {section.number}/10 written: {section.title}")
            return section

        rest = await asyncio.gather(*(run(s) for s in specs[1:]))
        body_sections = [first, *rest]
        joined = "\n\n".join(f"## {s.number}. {s.title}\n\n{s.markdown}" for s in body_sections)
        summary = await self._write_section(
            EXECUTIVE_SUMMARY, evidence, EXECUTIVE_SUMMARY_REQUEST.format(sections=joined), run_key
        )
        progress("section 1/10 written: Executive Summary")
        return sorted([summary, *body_sections], key=lambda s: s.number)

    # ----------------------------------------------------------- top level
    async def synthesize(
        self,
        manifest: ProfileManifest,
        outcome: VisionOutcome,
        progress: ProgressFn | None = None,
    ) -> tuple[PersonaReport, Path]:
        notify = progress or (lambda msg: log.info(msg))
        stats = compute_stats(manifest)
        notify("computing per-surface digests")
        digests = await self.build_digests(manifest, outcome, stats, notify)
        evidence = self.build_evidence(manifest, stats, digests, outcome)
        (self.settings.run_dir(manifest.run_key) / "evidence_pack.txt").write_text(evidence, encoding="utf-8")
        notify(f"evidence pack ready ({len(evidence):,} chars); writing sections")
        sections = await self.write_sections(manifest.run_key, evidence, notify)
        cfg = self.settings.anthropic
        report = PersonaReport(
            run_key=manifest.run_key,
            sections=sections,
            appendix_markdown=render_appendix(manifest, stats, outcome, self.settings),
            models={"vision": cfg.vision_model, "synthesis": cfg.synthesis_model},
        )
        path = self.settings.report_path(manifest.run_key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(render_report(manifest, report), encoding="utf-8")
        write_item_index(manifest, outcome, path.with_name(path.stem.replace("_full_report", "") + "_item_index.csv"))
        return report, path


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _anchor(title: str, number: int) -> str:
    slug = "".join(ch if ch.isalnum() or ch in " -" else "" for ch in f"{number}. {title}".lower())
    return slug.replace(" ", "-")


def render_report(manifest: ProfileManifest, report: PersonaReport) -> str:
    handles = " · ".join(
        f"{'Instagram' if p.platform == 'instagram' else 'TikTok'} [@{p.username}]"
        f"(https://www.{'instagram.com/' if p.platform == 'instagram' else 'tiktok.com/@'}{p.username})"
        for p in manifest.profiles
    )
    title_handle = manifest.profiles[0].username if manifest.profiles else manifest.run_key
    lines = [
        f"# Persona, Aesthetic, Lifestyle & Content Intelligence Report — @{title_handle}",
        "",
        f"> **Profiles:** {handles}  ",
        f"> **Generated:** {report.generated_at.strftime('%Y-%m-%d %H:%M UTC')}  ",
        f"> **Items analyzed:** {len(manifest.items):,}  ",
        f"> **Models:** vision `{report.models.get('vision')}`, synthesis `{report.models.get('synthesis')}`",
        "",
        "> _Scope: this report analyzes publicly published creator content as a body of media work "
        "(themes, style, voice, audience response, commercial fit). It deliberately excludes "
        "inferences about sensitive personal characteristics, private life, whereabouts, and third "
        "parties. Treat it as an analytical aid, and verify key claims against the cited items._",
        "",
        "## Contents",
        "",
    ]
    for spec in ALL_SECTIONS:
        lines.append(f"{spec.number}. [{spec.title}](#{_anchor(spec.title, spec.number)})")
    lines.append(f"{len(ALL_SECTIONS) + 1}. [Appendices](#appendices)")
    lines.append("")
    for section in report.sections:
        lines += [f"## {section.number}. {section.title}", "", section.markdown.strip(), ""]
    lines += ["## Appendices", "", report.appendix_markdown.strip(), ""]
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:,.2f}".rstrip("0").rstrip(".")
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def render_appendix(
    manifest: ProfileManifest,
    stats: dict[str, Any],
    outcome: VisionOutcome,
    settings: Settings,
) -> str:
    out = [
        "### A. Data coverage",
        "",
        "| Surface | Items | Date range | With visuals | With transcript | Analyzed by vision |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for surface in Surface:
        s = stats["surfaces"].get(surface.value)
        if not s:
            continue
        analyzed = sum(1 for i in manifest.items_for(surface) if i.item_id in outcome.observations)
        span = f"{s['first_date'] or '?'} → {s['last_date'] or '?'}"
        out.append(
            f"| {SURFACE_LABELS[surface]} | {s['items']:,} | {span} | {s['with_visuals']:,} | "
            f"{s['with_transcript']:,} | {analyzed:,} |"
        )
    collected = {s.value for s in Surface if stats["surfaces"].get(s.value)}
    missing = [SURFACE_LABELS[s] for s in Surface if s.value not in collected]
    if missing:
        out += ["", f"_Not collected or empty:_ {', '.join(missing)}."]
    if outcome.failed_items or outcome.refused_items:
        out += [
            "",
            f"_Vision analysis gaps:_ {len(outcome.failed_items)} item(s) failed, "
            f"{len(outcome.refused_items)} declined by the model; they are excluded from the analysis.",
        ]
    if manifest.notes:
        out += ["", "_Collection notes:_"] + [f"- {n}" for n in manifest.notes]

    out += [
        "",
        "### B. Engagement & cadence statistics",
        "",
        "| Surface | Median likes | Median comments | Median views | Median eng. rate % | Posts / week |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for surface in Surface:
        s = stats["surfaces"].get(surface.value)
        if s:
            out.append(
                f"| {SURFACE_LABELS[surface]} | {_fmt(s.get('median_likes'))} | {_fmt(s.get('median_comments'))} | "
                f"{_fmt(s.get('median_views'))} | {_fmt(s.get('median_engagement_rate_pct'))} | "
                f"{_fmt(s.get('posts_per_week')) if surface in _CADENCE_SURFACES else '—'} |"
            )
    out += ["", "| Platform | Followers | Following | Reported posts | Total likes |", "|---|---:|---:|---:|---:|"]
    for platform, p in stats["platforms"].items():
        out.append(
            f"| {platform} (@{p['username']}) | {_fmt(p['followers'])} | {_fmt(p['following'])} | "
            f"{_fmt(p['posts_count_reported'])} | {_fmt(p['total_likes'])} |"
        )

    out += ["", "### C. Hashtags, mentions & audio", ""]
    for surface in Surface:
        s = stats["surfaces"].get(surface.value)
        if not s or not (s.get("top_hashtags") or s.get("top_licensed_audio")):
            continue
        out.append(f"**{SURFACE_LABELS[surface]}**")
        if s.get("top_hashtags"):
            out.append(f"- Hashtags: {', '.join(s['top_hashtags'][:12])}")
        if s.get("top_mentions"):
            out.append(f"- Mentions/tags: {', '.join(s['top_mentions'][:10])}")
        if s.get("top_licensed_audio"):
            out.append(f"- Licensed audio: {', '.join(s['top_licensed_audio'][:6])}")
        out.append("")

    cfg = settings.anthropic
    out += [
        "### D. Methodology & limitations",
        "",
        "- **Collection:** authenticated browser session (Playwright `storage_state`); the web app's own "
        "JSON responses were captured or requested in-page, with human-paced scrolling. Private accounts "
        "are not processed (except the operator's own).",
        f"- **Distillation:** videos reduced to {settings.media.keyframes_min}-{settings.media.keyframes_max} "
        "scene-change keyframes per contact sheet (ffmpeg); speech transcribed locally with faster-whisper "
        f"(`{settings.whisper.model_size}`); photos normalized with Pillow.",
        f"- **Analysis:** map stage `{cfg.vision_model}` (structured per-item observations, batches of up to "
        f"{cfg.vision_items_per_batch} items); reduce stage `{cfg.synthesis_model}` (per-surface digests, then "
        "one request per section over a cached evidence pack). Statistics are computed locally, not by the model.",
        "- **Data minimization:** GPS coordinates and commenter identities are discarded during normalization; "
        "comment text is kept only as an audience-sentiment sample.",
        "- **Limitations:** keyframes sample, not watch, each video; ASR can mishear speech and lyrics; Stories "
        "reflect only what was live at collection time; engagement counts are snapshots; model interpretations "
        "can be wrong - follow the citations.",
        "",
        "### E. Item index",
        "",
        "Every cited item id is listed with its date, surface and URL in the companion "
        "`*_item_index.csv` file next to this report.",
    ]
    return "\n".join(out)


def write_item_index(manifest: ProfileManifest, outcome: VisionOutcome, path: Path) -> Path:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "item_id",
                "platform",
                "surface",
                "date",
                "author",
                "url",
                "likes",
                "comments",
                "views",
                "analyzed",
                "summary",
            ]
        )
        for item in manifest.items:
            obs = outcome.observations.get(item.item_id)
            writer.writerow(
                [
                    item.item_id,
                    item.platform,
                    item.surface.value,
                    item.taken_at.isoformat() if item.taken_at else "",
                    item.author_handle or "",
                    item.url or "",
                    item.metrics.likes,
                    item.metrics.comments,
                    item.metrics.views,
                    "yes" if obs else "no",
                    (obs.summary if obs else "")[:300],
                ]
            )
    return path


__all__ = [
    "ReportSynthesizer",
    "compute_stats",
    "render_report",
    "surface_stats",
]
