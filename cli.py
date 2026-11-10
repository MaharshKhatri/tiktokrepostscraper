"""Command-line interface: ``audit auth | scrape | process | report | run | status | purge | doctor``."""

from __future__ import annotations

import asyncio
import logging
import shutil
import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from auth.session_manager import (
    SessionError,
    inspect_session,
    interactive_login,
    validate_session,
)
from config.settings import Settings, get_settings, normalize_handle
from intelligence.schemas import ProfileManifest, Surface
from pipeline.normalizer import build_manifest, load_manifest, save_manifest
from scrapers.base_scraper import (
    PrivateProfileError,
    ProfileNotFoundError,
    ScrapeError,
    load_scrape_result,
)
from storage.cache_manager import CacheManager

app = typer.Typer(
    name="audit",
    help="Cross-Platform Creator Profile & Deep Persona Audit Engine.",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_show_locals=False,
)
console = Console()
log = logging.getLogger("audit")

IG_SURFACE_NAMES = [s.value for s in Surface if s.value.startswith("ig_")]
TT_SURFACE_NAMES = [s.value for s in Surface if s.value.startswith("tt_")]

IgOpt = Annotated[str | None, typer.Option("--ig", help="Instagram username (or profile URL).")]
TtOpt = Annotated[str | None, typer.Option("--tiktok", help="TikTok username (or profile URL).")]
NameOpt = Annotated[
    str | None, typer.Option("--name", help="Report/run name. Defaults to the IG handle, else the TikTok handle.")
]
VerboseOpt = Annotated[bool, typer.Option("--verbose", "-v", help="Debug logging.")]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _setup(verbose: bool) -> Settings:
    settings = get_settings().model_copy(deep=True)
    level = logging.DEBUG if verbose else getattr(logging, settings.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=verbose)],
        force=True,
    )
    for noisy in ("httpx", "httpx2", "httpcore", "anthropic", "faster_whisper"):
        logging.getLogger(noisy).setLevel(logging.WARNING if not verbose else logging.INFO)
    return settings


def _run_key(name: str | None, ig: str | None, tiktok: str | None) -> str:
    key = name or (normalize_handle(ig) if ig else None) or (normalize_handle(tiktok) if tiktok else None)
    if not key:
        console.print("[red]Pass --ig and/or --tiktok (or --name for an existing run).[/red]")
        raise typer.Exit(2)
    return normalize_handle(key)


def _parse_surfaces(raw: str | None) -> list[Surface] | None:
    if not raw:
        return None
    out = []
    for part in raw.split(","):
        part = part.strip().lower()
        if not part:
            continue
        try:
            out.append(Surface(part))
        except ValueError:
            console.print(f"[red]Unknown surface '{part}'. Valid: {', '.join(s.value for s in Surface)}[/red]")
            raise typer.Exit(2) from None
    return out


def _progress() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("[dim]{task.fields[detail]}"),
        TimeElapsedColumn(),
        console=console,
    )


def _status(msg: str) -> None:
    console.log(msg)


# ---------------------------------------------------------------------------
# Stage implementations
# ---------------------------------------------------------------------------


async def _scrape(
    settings: Settings,
    cache: CacheManager,
    run_key: str,
    ig: str | None,
    tiktok: str | None,
    surfaces: list[Surface] | None,
) -> ProfileManifest | None:
    from scrapers.instagram_scraper import InstagramScraper
    from scrapers.tiktok_scraper import TikTokScraper

    jobs = []
    if ig:
        jobs.append(("instagram", InstagramScraper, normalize_handle(ig)))
    if tiktok:
        jobs.append(("tiktok", TikTokScraper, normalize_handle(tiktok)))
    errors: list[str] = []
    for platform, cls, username in jobs:
        wanted = (
            [s for s in surfaces if s.value.startswith("ig_" if platform == "instagram" else "tt_")]
            if surfaces
            else None
        )
        if surfaces and not wanted:
            continue
        console.rule(f"[bold]{platform}[/bold] @{username}")
        scraper = cls(settings, cache, run_key, progress=_status)
        try:
            await scraper.start()
            result = await scraper.scrape(username, wanted)
            cache.mark_stage(run_key, f"scrape:{platform}", f"{len(result.items)} items")
            counts = {s.label: len(result.by_surface(s)) for s in result.surfaces_completed}
            _status(f"[green]{platform} done[/green]: " + ", ".join(f"{k}: {v}" for k, v in counts.items()))
        except (SessionError, PrivateProfileError, ProfileNotFoundError, ScrapeError) as exc:
            errors.append(f"{platform}: {exc}")
            console.print(f"[red]{platform}: {exc}[/red]")
        finally:
            await scraper.stop()

    results = [r for p in ("instagram", "tiktok") if (r := load_scrape_result(settings, run_key, p))]  # type: ignore[arg-type]
    if not results:
        console.print("[red]Nothing was collected.[/red]")
        return None
    manifest = build_manifest(run_key, results)
    path = save_manifest(manifest, settings)
    cache.mark_stage(run_key, "manifest", f"{len(manifest.items)} items")
    _status(f"Manifest: {len(manifest.items)} items -> {path}")
    if errors:
        console.print(Panel("\n".join(errors), title="Some platforms failed", border_style="yellow"))
    return manifest


async def _process(settings: Settings, cache: CacheManager, manifest: ProfileManifest) -> ProfileManifest:
    from pipeline.media_processor import check_binaries, distill_manifest

    missing = [name for name, ok in check_binaries(settings.media).items() if not ok]
    if missing:
        console.print(f"[red]Missing required binaries: {', '.join(missing)}. Install ffmpeg first.[/red]")
        raise typer.Exit(1)
    with _progress() as progress:
        task = progress.add_task("Distilling media", total=len(manifest.items), detail="")
        await distill_manifest(
            manifest,
            settings,
            cache,
            progress=lambda done, total, item_id: progress.update(task, completed=done, detail=item_id),
        )
    save_manifest(manifest, settings)
    sheets = sum(1 for i in manifest.items if i.derived.contact_sheet)
    transcripts = sum(1 for i in manifest.items if i.derived.transcript and i.derived.transcript.text)
    errors = sum(1 for i in manifest.items if i.derived.errors)
    cache.mark_stage(manifest.run_key, "process", f"{sheets} sheets, {transcripts} transcripts")
    _status(f"Distilled: {sheets} contact sheets, {transcripts} transcripts, {errors} items with warnings")
    return manifest


async def _report(settings: Settings, cache: CacheManager, manifest: ProfileManifest) -> Path:
    from intelligence.report_synthesizer import ReportSynthesizer
    from intelligence.vision_analyzer import ClaudeGateway, VisionAnalyzer

    gateway = ClaudeGateway(settings)
    analyzer = VisionAnalyzer(settings, cache, gateway)
    with _progress() as progress:
        task = progress.add_task("Vision analysis", total=None, detail="")

        def update(done: int, total: int, detail: str) -> None:
            progress.update(task, completed=done, total=total, detail=detail)

        outcome = await analyzer.analyze(manifest, progress=update)
    cache.mark_stage(manifest.run_key, "vision", f"{len(outcome.observations)} observations")
    _status(
        f"Vision: {len(outcome.observations)} observations, {len(outcome.failed_items)} failed, "
        f"{len(outcome.refused_items)} declined"
    )
    if not outcome.observations:
        console.print("[red]No observations were produced; cannot write a report.[/red]")
        raise typer.Exit(1)
    synthesizer = ReportSynthesizer(settings, cache, gateway)
    _, path = await synthesizer.synthesize(manifest, outcome, progress=_status)
    cache.mark_stage(manifest.run_key, "report", str(path))
    u = gateway.usage
    table = Table(title="Claude API usage", show_header=False)
    for label, value in (
        ("requests", u.requests),
        ("input tokens", u.input_tokens),
        ("cache-read tokens", u.cache_read_input_tokens),
        ("cache-write tokens", u.cache_creation_input_tokens),
        ("output tokens", u.output_tokens),
        ("refusals", u.refusals),
        ("served by fallback", u.fallbacks_used),
    ):
        table.add_row(label, f"{value:,}")
    console.print(table)
    return path


def _load_manifest_or_exit(settings: Settings, run_key: str) -> ProfileManifest:
    manifest = load_manifest(settings, run_key)
    if manifest is None:
        console.print(f"[red]No manifest for '{run_key}'. Run `audit scrape` first.[/red]")
        raise typer.Exit(1)
    return manifest


def _apply_overrides(
    settings: Settings, headed: bool, max_items: int | None, no_download: bool = False, whisper_model: str | None = None
) -> None:
    if headed:
        settings.browser.headless = False
    if max_items is not None:
        settings.scrape.max_items_per_surface = max_items
    if no_download:
        settings.scrape.download_media = False
    if whisper_model:
        settings.whisper.model_size = whisper_model


def _run(coro):  # type: ignore[no-untyped-def]
    try:
        return asyncio.run(coro)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted - progress is checkpointed; re-run the same command to resume.[/yellow]")
        raise typer.Exit(130) from None


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@app.command()
def auth(
    platform: Annotated[str, typer.Argument(help="instagram | tiktok")],
    check: Annotated[bool, typer.Option("--check", help="Validate the saved session instead of logging in.")] = False,
    timeout: Annotated[int, typer.Option(help="Seconds to wait for the manual login.")] = 600,
    verbose: VerboseOpt = False,
) -> None:
    """Sign in once in a headed browser and save the session (storage_state)."""

    settings = _setup(verbose)
    platform = platform.lower().strip()
    if platform in ("ig", "insta"):
        platform = "instagram"
    if platform in ("tt",):
        platform = "tiktok"
    if platform not in ("instagram", "tiktok"):
        console.print("[red]platform must be 'instagram' or 'tiktok'[/red]")
        raise typer.Exit(2)
    if check:
        status = _run(validate_session(platform, settings))  # type: ignore[arg-type]
        color = "green" if status.valid else "red"
        days = f", cookie expires in {status.expires_in_days:.0f} days" if status.expires_in_days else ""
        console.print(
            f"[{color}]{platform}: {'valid' if status.valid else 'invalid'}[/{color}] - {status.detail}{days}"
        )
        raise typer.Exit(0 if status.valid else 1)
    try:
        path = _run(interactive_login(platform, settings, timeout_s=timeout, on_status=_status))  # type: ignore[arg-type]
    except SessionError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from None
    console.print(f"[green]Session saved:[/green] {path} (keep this file private; it is git-ignored)")


@app.command()
def scrape(
    ig: IgOpt = None,
    tiktok: TtOpt = None,
    name: NameOpt = None,
    surfaces: Annotated[
        str | None,
        typer.Option(
            help=f"Comma list of surfaces. IG: {','.join(IG_SURFACE_NAMES)}; TikTok: {','.join(TT_SURFACE_NAMES)}"
        ),
    ] = None,
    max_items: Annotated[int | None, typer.Option(help="Cap items per surface (0 = all).")] = None,
    headed: Annotated[
        bool, typer.Option("--headed", help="Show the browser (needed to solve TikTok puzzles).")
    ] = False,
    refresh: Annotated[bool, typer.Option("--refresh", help="Ignore checkpoints and re-collect everything.")] = False,
    no_download: Annotated[bool, typer.Option("--no-download", help="Collect metadata only.")] = False,
    verbose: VerboseOpt = False,
) -> None:
    """Collect profiles, posts, Reels, Stories, Highlights, Tagged and TikTok videos/reposts."""

    settings = _setup(verbose)
    if not ig and not tiktok:
        console.print("[red]Pass --ig and/or --tiktok.[/red]")
        raise typer.Exit(2)
    _apply_overrides(settings, headed, max_items, no_download)
    run_key = _run_key(name, ig, tiktok)
    with CacheManager(settings.cache_db) as cache:
        if refresh:
            for platform in ("instagram", "tiktok"):
                cache.delete(f"{run_key}:{platform}:surface_items")
        manifest = _run(_scrape(settings, cache, run_key, ig, tiktok, _parse_surfaces(surfaces)))
    if manifest is None:
        raise typer.Exit(1)


@app.command()
def process(
    ig: IgOpt = None,
    tiktok: TtOpt = None,
    name: NameOpt = None,
    whisper_model: Annotated[
        str | None, typer.Option(help="faster-whisper model size (tiny/base/small/medium/large-v3).")
    ] = None,
    verbose: VerboseOpt = False,
) -> None:
    """Extract keyframes + contact sheets and transcribe audio locally."""

    settings = _setup(verbose)
    _apply_overrides(settings, False, None, whisper_model=whisper_model)
    run_key = _run_key(name, ig, tiktok)
    with CacheManager(settings.cache_db) as cache:
        manifest = _load_manifest_or_exit(settings, run_key)
        _run(_process(settings, cache, manifest))


@app.command()
def report(
    ig: IgOpt = None,
    tiktok: TtOpt = None,
    name: NameOpt = None,
    skip_process: Annotated[bool, typer.Option("--skip-process", help="Use existing derived media as-is.")] = False,
    whisper_model: Annotated[str | None, typer.Option(help="faster-whisper model size.")] = None,
    verbose: VerboseOpt = False,
) -> None:
    """Run vision map-reduce over the collected media and write the 10-section report."""

    settings = _setup(verbose)
    _apply_overrides(settings, False, None, whisper_model=whisper_model)
    run_key = _run_key(name, ig, tiktok)
    with CacheManager(settings.cache_db) as cache:
        manifest = _load_manifest_or_exit(settings, run_key)
        if not skip_process:
            manifest = _run(_process(settings, cache, manifest))
        path = _run(_report(settings, cache, manifest))
    console.print(Panel(f"[bold green]Report written:[/bold green] {path}", border_style="green"))


@app.command("run")
def run_all(
    ig: IgOpt = None,
    tiktok: TtOpt = None,
    name: NameOpt = None,
    surfaces: Annotated[str | None, typer.Option(help="Comma list of surfaces to collect.")] = None,
    max_items: Annotated[int | None, typer.Option(help="Cap items per surface (0 = all).")] = None,
    headed: Annotated[bool, typer.Option("--headed", help="Show the browser.")] = False,
    refresh: Annotated[bool, typer.Option("--refresh", help="Re-collect instead of resuming.")] = False,
    whisper_model: Annotated[str | None, typer.Option(help="faster-whisper model size.")] = None,
    verbose: VerboseOpt = False,
) -> None:
    """scrape -> process -> report in one go (resumable)."""

    settings = _setup(verbose)
    if not ig and not tiktok:
        console.print("[red]Pass --ig and/or --tiktok.[/red]")
        raise typer.Exit(2)
    _apply_overrides(settings, headed, max_items, whisper_model=whisper_model)
    run_key = _run_key(name, ig, tiktok)
    with CacheManager(settings.cache_db) as cache:
        if refresh:
            for platform in ("instagram", "tiktok"):
                cache.delete(f"{run_key}:{platform}:surface_items")
        manifest = _run(_scrape(settings, cache, run_key, ig, tiktok, _parse_surfaces(surfaces)))
        if manifest is None:
            raise typer.Exit(1)
        manifest = _run(_process(settings, cache, manifest))
        path = _run(_report(settings, cache, manifest))
    console.print(Panel(f"[bold green]Report written:[/bold green] {path}", border_style="green"))


@app.command()
def status(
    name: Annotated[str, typer.Argument(help="Run name (usually the handle).")], verbose: VerboseOpt = False
) -> None:
    """Show checkpointed stages and collected counts for a run."""

    settings = _setup(verbose)
    run_key = normalize_handle(name)
    with CacheManager(settings.cache_db) as cache:
        stages = cache.stages(run_key)
    table = Table(title=f"Run '{run_key}'")
    table.add_column("Stage")
    table.add_column("Detail")
    for stage, detail in stages.items():
        table.add_row(stage, detail or "")
    console.print(table if stages else f"[yellow]No checkpoints for '{run_key}'.[/yellow]")
    manifest = load_manifest(settings, run_key)
    if manifest:
        counts = Table(title="Collected items")
        counts.add_column("Surface")
        counts.add_column("Items", justify="right")
        for surface, n in manifest.surface_counts().items():
            counts.add_row(surface.label, str(n))
        console.print(counts)


@app.command()
def purge(
    name: Annotated[str, typer.Argument(help="Run name to delete.")],
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip confirmation.")] = False,
    keep_report: Annotated[bool, typer.Option(help="Keep the Markdown report.")] = True,
) -> None:
    """Delete all collected media, derived files and checkpoints for a run (data retention)."""

    settings = _setup(False)
    run_key = normalize_handle(name)
    run_dir = settings.run_dir(run_key)
    if not yes and not typer.confirm(f"Delete {run_dir} and all checkpoints for '{run_key}'?"):
        raise typer.Exit(1)
    if run_dir.exists():
        shutil.rmtree(run_dir)
    with CacheManager(settings.cache_db) as cache:
        cache.reset_run(run_key)
    if not keep_report:
        for path in settings.reports_dir.glob(f"{run_dir.name}_*"):
            path.unlink()
    console.print(f"[green]Purged '{run_key}'.[/green]")


@app.command()
def doctor(verbose: VerboseOpt = False) -> None:
    """Check binaries, sessions, browser and API configuration."""

    settings = _setup(verbose)
    from pipeline.media_processor import check_binaries

    table = Table(title="Environment check")
    table.add_column("Check")
    table.add_column("Status")
    table.add_column("Detail")
    table.add_row("python", "ok" if sys.version_info >= (3, 11) else "[red]too old[/red]", sys.version.split()[0])
    for binary, ok in check_binaries(settings.media).items():
        table.add_row(binary, "ok" if ok else "[red]missing[/red]", shutil.which(binary) or "")
    table.add_row("yt-dlp", "ok" if shutil.which("yt-dlp") or _importable("yt_dlp") else "[red]missing[/red]", "")
    table.add_row(
        "faster-whisper", "ok" if _importable("faster_whisper") else "[red]missing[/red]", settings.whisper.model_size
    )
    for platform in ("instagram", "tiktok"):
        st = inspect_session(platform, settings)  # type: ignore[arg-type]
        ok = st.exists and st.has_auth_cookie and st.detail == "ok"
        days = f" (expires in {st.expires_in_days:.0f}d)" if st.expires_in_days else ""
        table.add_row(f"{platform} session", "ok" if ok else "[yellow]not ready[/yellow]", f"{st.detail}{days}")
    import os

    has_key = bool(
        settings.anthropic.api_key or os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")
    )
    table.add_row(
        "Anthropic credentials",
        "ok" if has_key else "[yellow]env var not set[/yellow]",
        "an `ant auth login` profile also works" if not has_key else "",
    )
    table.add_row(
        "models", "", f"vision={settings.anthropic.vision_model}, synthesis={settings.anthropic.synthesis_model}"
    )
    console.print(table)


def _importable(module: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(module) is not None


if __name__ == "__main__":
    app()
