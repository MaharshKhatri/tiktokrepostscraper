from __future__ import annotations

from typer.testing import CliRunner

import cli
from config import settings as settings_module

runner = CliRunner()


def _patch_settings(monkeypatch, settings) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    monkeypatch.setattr(settings_module, "get_settings", lambda: settings)


def test_help_lists_commands() -> None:
    result = runner.invoke(cli.app, ["--help"])
    assert result.exit_code == 0
    for command in ("auth", "scrape", "process", "report", "run", "status", "purge", "doctor"):
        assert command in result.output


def test_scrape_requires_a_handle(monkeypatch, settings) -> None:  # type: ignore[no-untyped-def]
    _patch_settings(monkeypatch, settings)
    result = runner.invoke(cli.app, ["scrape"])
    assert result.exit_code == 2


def test_scrape_without_session_reports_error(monkeypatch, settings) -> None:  # type: ignore[no-untyped-def]
    _patch_settings(monkeypatch, settings)
    result = runner.invoke(cli.app, ["scrape", "--ig", "https://www.instagram.com/creator/", "--no-download"])
    assert result.exit_code == 1
    assert "audit auth instagram" in result.output


def test_bad_surface_name(monkeypatch, settings) -> None:  # type: ignore[no-untyped-def]
    _patch_settings(monkeypatch, settings)
    result = runner.invoke(cli.app, ["scrape", "--ig", "creator", "--surfaces", "ig_post,bogus"])
    assert result.exit_code == 2 and "Unknown surface" in result.output


def test_report_without_manifest(monkeypatch, settings) -> None:  # type: ignore[no-untyped-def]
    _patch_settings(monkeypatch, settings)
    result = runner.invoke(cli.app, ["report", "--name", "nobody"])
    assert result.exit_code == 1 and "audit scrape" in result.output


def test_status_and_doctor(monkeypatch, settings) -> None:  # type: ignore[no-untyped-def]
    _patch_settings(monkeypatch, settings)
    assert runner.invoke(cli.app, ["status", "creator"]).exit_code == 0
    result = runner.invoke(cli.app, ["doctor"])
    assert result.exit_code == 0 and "ffmpeg" in result.output


def test_purge(monkeypatch, settings) -> None:  # type: ignore[no-untyped-def]
    _patch_settings(monkeypatch, settings)
    run_dir = settings.run_dir("creator")
    (run_dir / "media").mkdir(parents=True)
    (run_dir / "media" / "x.jpg").write_bytes(b"1")
    result = runner.invoke(cli.app, ["purge", "creator", "--yes"])
    assert result.exit_code == 0 and not run_dir.exists()


def test_report_command_end_to_end(monkeypatch, settings, tmp_path) -> None:  # type: ignore[no-untyped-def]
    """scrape output (fixtures) -> `audit report` with real ffmpeg and a fake Claude client."""

    import shutil

    import pytest
    from PIL import Image

    from intelligence import vision_analyzer
    from pipeline.normalizer import build_manifest, save_manifest
    from tests.conftest import fake_client, make_test_video
    from tests.test_intelligence import _echo_parse
    from tests.test_normalizer import _ig_result, _tt_result

    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not installed")
    _patch_settings(monkeypatch, settings)
    video = make_test_video(tmp_path / "v.mp4", seconds=6)
    photo = tmp_path / "p.jpg"
    Image.new("RGB", (1080, 1350), "beige").save(photo)
    manifest = build_manifest("creator", [_ig_result(), _tt_result()])
    for item in manifest.items:
        for asset in item.assets:
            asset.local_path = str(video if asset.kind == "video" else photo)
    save_manifest(manifest, settings)

    client = fake_client(_echo_parse)
    monkeypatch.setattr(vision_analyzer.anthropic, "AsyncAnthropic", lambda **kw: client)
    result = runner.invoke(cli.app, ["report", "--name", "creator"])
    assert result.exit_code == 0, result.output
    report_path = settings.report_path("creator")
    text = report_path.read_text()
    assert text.count("\n## ") >= 11  # contents + 10 sections + appendices
    assert "Instagram [@creator]" in text and "TikTok [@creator]" in text
    assert len(client.beta.messages.stream_calls) == 10
    batch_calls = [c for c in client.beta.messages.parse_calls if "ITEM " in str(c["messages"])]
    images = sum(1 for c in batch_calls for b in c["messages"][0]["content"] if b["type"] == "image")
    assert images == len(manifest.items)  # one sheet/photo per item
