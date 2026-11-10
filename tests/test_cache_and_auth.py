from __future__ import annotations

import json
import time

from auth.session_manager import export_netscape_cookies, inspect_session
from config.settings import normalize_handle, safe_slug
from intelligence.schemas import Transcript
from storage.cache_manager import CacheManager, stable_hash


def test_kv_roundtrip_and_prefix_escaping(cache: CacheManager) -> None:
    cache.set("run_a:obs", "k_1", {"x": 1})
    cache.set("run_a:obs", "kx1", {"x": 2})
    cache.set("run_a:obs", "k%1", {"x": 3})
    assert cache.get("run_a:obs", "k_1") == {"x": 1}
    # "_" and "%" must be literal, not LIKE wildcards
    assert cache.keys("run_a:obs", "k_") == ["k_1"]
    assert cache.keys("run_a:obs", "k%") == ["k%1"]
    cache.set("transcripts", "abc", Transcript(text="hello", language="en"))
    assert cache.get_model("transcripts", "abc", Transcript).text == "hello"  # type: ignore[union-attr]
    assert cache.get_model("transcripts", "missing", Transcript) is None


def test_stages_and_reset_run(cache: CacheManager) -> None:
    cache.mark_stage("foo_bar", "scrape:instagram", "10 items")
    cache.set("foo_bar:observations", "b1", {"ok": True})
    cache.set("fooxbar:observations", "b1", {"ok": True})
    assert cache.stage_done("foo_bar", "scrape:instagram")
    cache.reset_run("foo_bar")
    assert not cache.stage_done("foo_bar", "scrape:instagram")
    assert cache.get("foo_bar:observations", "b1") is None
    assert cache.get("fooxbar:observations", "b1") == {"ok": True}  # untouched


def test_stable_hash_is_order_insensitive_for_dicts() -> None:
    assert stable_hash({"a": 1, "b": 2}) == stable_hash({"b": 2, "a": 1})
    assert stable_hash("a") != stable_hash("b")


def test_handle_normalization() -> None:
    assert normalize_handle("https://www.instagram.com/Some.User/?hl=en") == "Some.User"
    assert normalize_handle("@creator") == "creator"
    assert normalize_handle("https://www.tiktok.com/@creator/video/123") == "@creator".lstrip("@")
    assert safe_slug("@Some User!") == "some_user"


def test_session_inspection_and_cookie_export(settings) -> None:  # type: ignore[no-untyped-def]
    state = {
        "cookies": [
            {
                "name": "sessionid",
                "value": "abc",
                "domain": ".tiktok.com",
                "path": "/",
                "expires": time.time() + 86400 * 30,
                "httpOnly": True,
                "secure": True,
            },
            {
                "name": "tt_csrf_token",
                "value": "xyz",
                "domain": "www.tiktok.com",
                "path": "/",
                "expires": -1,
                "httpOnly": False,
                "secure": True,
            },
        ],
        "origins": [],
    }
    path = settings.session_file("tiktok")
    path.write_text(json.dumps(state))
    status = inspect_session("tiktok", settings)
    assert status.exists and status.has_auth_cookie and status.detail == "ok"
    assert status.expires_in_days and 29 < status.expires_in_days <= 30

    out = export_netscape_cookies(path, settings.tiktok_cookie_file())
    lines = [line for line in out.read_text().splitlines() if line and not line.startswith("# ")]
    assert lines[0].startswith("#HttpOnly_.tiktok.com\tTRUE\t/\tTRUE\t")
    assert lines[1].split("\t")[4] == "0"  # session cookie -> expiry 0
    assert oct(out.stat().st_mode & 0o777) == "0o600"

    assert inspect_session("instagram", settings).detail == "missing"
