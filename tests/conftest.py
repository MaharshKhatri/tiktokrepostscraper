from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from config.settings import Settings
from storage.cache_manager import CacheManager

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    s = Settings(
        data_dir=tmp_path / "data",
        reports_dir=tmp_path / "reports",
        sessions_dir=tmp_path / "sessions",
        cache_db=tmp_path / "data" / "cache.sqlite3",
    )
    s.whisper.enabled = False
    s.ensure_dirs()
    exe = os.environ.get("AUDIT_BROWSER__EXECUTABLE_PATH") or (
        "/opt/pw-browsers/chromium" if Path("/opt/pw-browsers/chromium").exists() else None
    )
    if exe:
        s.browser.executable_path = Path(exe)
        s.browser.extra_args = ["--no-sandbox"]
    s.rate_limits.request_delay_min_s = 0.0
    s.rate_limits.request_delay_max_s = 0.01
    s.rate_limits.scroll_pause_min_s = 0.05
    s.rate_limits.scroll_pause_max_s = 0.1
    s.rate_limits.backoff_base_s = 0.01
    return s


@pytest.fixture
def cache(settings: Settings):
    with CacheManager(settings.cache_db) as c:
        yield c


def load_fixture(name: str) -> Any:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def make_test_video(path: Path, seconds: int = 6, with_audio: bool = True, portrait: bool = True) -> Path:
    """Synthesize a video with hard scene cuts (color blocks) and a tone."""

    size = "540x960" if portrait else "960x540"
    colors = ["red", "green", "blue", "yellow", "purple", "white"]
    per = max(1, seconds // len(colors))
    inputs: list[str] = []
    for c in colors[: max(2, seconds // per)]:
        inputs += ["-f", "lavfi", "-t", str(per), "-i", f"color=c={c}:s={size}:r=25"]
    n = len(inputs) // 6
    filt = "".join(f"[{i}:v]drawtext=text='scene {i}':fontcolor=black:fontsize=64:x=40:y=80[v{i}];" for i in range(n))
    filt += "".join(f"[v{i}]" for i in range(n)) + f"concat=n={n}:v=1:a=0[outv]"
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *inputs]
    if with_audio:
        cmd += ["-f", "lavfi", "-t", str(per * n), "-i", "sine=frequency=440:sample_rate=16000"]
    cmd += ["-filter_complex", filt, "-map", "[outv]"]
    if with_audio:
        cmd += ["-map", f"{n}:a", "-c:a", "aac"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError:
        # drawtext needs fontconfig; retry without text overlays.
        filt = "".join(f"[{i}:v]" for i in range(n)) + f"concat=n={n}:v=1:a=0[outv]"
        idx = cmd.index("-filter_complex")
        cmd[idx + 1] = filt
        subprocess.run(cmd, check=True, capture_output=True)
    return path


# ---------------------------------------------------------------------------
# Fake Anthropic client
# ---------------------------------------------------------------------------


class _FakeStream:
    def __init__(self, message: Any) -> None:
        self.message = message

    async def __aenter__(self) -> _FakeStream:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    async def get_final_message(self) -> Any:
        return self.message


class FakeMessages:
    """Mimics ``client.beta.messages`` (parse + stream) and records calls."""

    def __init__(self, parse_handler, stream_handler) -> None:  # type: ignore[no-untyped-def]
        self.parse_handler = parse_handler
        self.stream_handler = stream_handler
        self.parse_calls: list[dict] = []
        self.stream_calls: list[dict] = []

    async def parse(self, **kwargs: Any) -> Any:
        self.parse_calls.append(kwargs)
        return self.parse_handler(kwargs)

    def stream(self, **kwargs: Any) -> _FakeStream:
        self.stream_calls.append(kwargs)
        return _FakeStream(self.stream_handler(kwargs))


def fake_message(
    *, parsed: Any = None, text: str = "", stop_reason: str = "end_turn", model: str = "claude-sonnet-5-5"
) -> Any:
    content = [SimpleNamespace(type="text", text=text or (parsed.model_dump_json() if parsed is not None else ""))]
    usage = SimpleNamespace(
        input_tokens=100, output_tokens=50, cache_read_input_tokens=0, cache_creation_input_tokens=0, iterations=None
    )
    return SimpleNamespace(
        parsed_output=parsed, content=content, stop_reason=stop_reason, model=model, usage=usage, stop_details=None
    )


def fake_client(parse_handler, stream_handler=None) -> Any:  # type: ignore[no-untyped-def]
    messages = FakeMessages(parse_handler, stream_handler or (lambda kw: fake_message(text="Section body.")))
    return SimpleNamespace(beta=SimpleNamespace(messages=messages))
