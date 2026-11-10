"""Operator login flow and ``storage_state`` persistence.

``audit auth instagram`` / ``audit auth tiktok`` opens a *headed* browser on the
platform's login page. The operator signs in by hand (including any 2FA or
captcha); this module only watches for the session cookie to appear and then
writes Playwright's ``storage_state`` JSON to ``auth/sessions/``. No
credentials are ever typed, read or stored by the engine itself.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path

from playwright.async_api import BrowserContext, async_playwright

from config.settings import Platform, Settings

log = logging.getLogger(__name__)

LOGIN_URLS: dict[str, str] = {
    "instagram": "https://www.instagram.com/accounts/login/",
    "tiktok": "https://www.tiktok.com/login",
}
HOME_URLS: dict[str, str] = {
    "instagram": "https://www.instagram.com/",
    "tiktok": "https://www.tiktok.com/",
}
# A session is considered established once one of these cookies exists.
AUTH_COOKIES: dict[str, tuple[str, ...]] = {
    "instagram": ("sessionid",),
    "tiktok": ("sessionid", "sessionid_ss", "sid_tt"),
}


class SessionError(RuntimeError):
    """Base class for authentication problems."""


class SessionMissingError(SessionError):
    def __init__(self, platform: str, path: Path) -> None:
        super().__init__(f"No saved {platform} session at {path}. Run `audit auth {platform}` first.")


class SessionExpiredError(SessionError):
    def __init__(self, platform: str, detail: str = "") -> None:
        msg = f"The saved {platform} session is no longer valid"
        msg += f" ({detail})." if detail else "."
        super().__init__(f"{msg} Run `audit auth {platform}` to sign in again.")


@dataclass
class SessionStatus:
    platform: str
    path: Path
    exists: bool
    has_auth_cookie: bool = False
    expires_at: float | None = None
    account_id: str | None = None
    valid: bool | None = None
    detail: str = ""

    @property
    def expires_in_days(self) -> float | None:
        if self.expires_at is None:
            return None
        return (self.expires_at - time.time()) / 86400


def _launch_kwargs(settings: Settings, headless: bool) -> dict:
    kwargs: dict = {"headless": headless, "slow_mo": settings.browser.slow_mo_ms}
    if settings.browser.executable_path:
        kwargs["executable_path"] = str(settings.browser.executable_path)
    if settings.browser.extra_args:
        kwargs["args"] = list(settings.browser.extra_args)
    return kwargs


def context_kwargs(settings: Settings) -> dict:
    kwargs: dict = {
        "locale": settings.browser.locale,
        "timezone_id": settings.browser.timezone_id,
        "viewport": {
            "width": settings.browser.viewport_width,
            "height": settings.browser.viewport_height,
        },
    }
    if settings.browser.user_agent:
        kwargs["user_agent"] = settings.browser.user_agent
    return kwargs


def _auth_cookie(cookies: list[dict], platform: str) -> dict | None:
    wanted = AUTH_COOKIES[platform]
    for cookie in cookies:
        if cookie.get("name") in wanted and cookie.get("value"):
            return cookie
    return None


def _write_state_securely(state: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(state, fh, indent=2)
    os.replace(tmp, path)
    os.chmod(path, 0o600)


async def interactive_login(
    platform: Platform,
    settings: Settings,
    timeout_s: int = 600,
    on_status: callable | None = None,  # type: ignore[valid-type]
) -> Path:
    """Open a headed browser, wait for the operator to log in, persist state."""

    path = settings.session_file(platform)
    notify = on_status or (lambda msg: log.info(msg))
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(**_launch_kwargs(settings, headless=False))
        context = await browser.new_context(**context_kwargs(settings))
        page = await context.new_page()
        await page.goto(LOGIN_URLS[platform], wait_until="domcontentloaded")
        notify(
            f"A browser window is open on the {platform} login page. Sign in there "
            f"(complete any 2FA/captcha). Waiting up to {timeout_s // 60} minutes..."
        )
        deadline = time.monotonic() + timeout_s
        cookie: dict | None = None
        while time.monotonic() < deadline:
            cookie = _auth_cookie(await context.cookies(), platform)
            if cookie and "/login" not in page.url and "/challenge" not in page.url:
                break
            await asyncio.sleep(2)
        if not cookie:
            await browser.close()
            raise SessionError(f"Timed out waiting for {platform} login to complete.")
        # Let post-login redirects and consent cookies settle before saving.
        notify("Login detected - letting the session settle for a few seconds...")
        await asyncio.sleep(5)
        try:
            await page.goto(HOME_URLS[platform], wait_until="domcontentloaded")
            await asyncio.sleep(3)
        except Exception:  # navigation is best-effort; cookies are what matter
            pass
        state = await context.storage_state()
        await browser.close()
    _write_state_securely(state, path)
    if platform == "tiktok":
        export_netscape_cookies(path, settings.tiktok_cookie_file())
    notify(f"Saved {platform} session to {path}")
    return path


def inspect_session(platform: Platform, settings: Settings) -> SessionStatus:
    """Static check of a saved session file (no network)."""

    path = settings.session_file(platform)
    status = SessionStatus(platform=platform, path=path, exists=path.exists())
    if not status.exists:
        status.detail = "missing"
        return status
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        status.detail = f"unreadable: {exc}"
        return status
    cookies = state.get("cookies", [])
    cookie = _auth_cookie(cookies, platform)
    status.has_auth_cookie = cookie is not None
    if cookie and cookie.get("expires", -1) and cookie.get("expires", -1) > 0:
        status.expires_at = float(cookie["expires"])
    if platform == "instagram":
        ds = next((c for c in cookies if c.get("name") == "ds_user_id"), None)
        status.account_id = ds.get("value") if ds else None
    if not cookie:
        status.detail = "no auth cookie"
    elif status.expires_at and status.expires_at < time.time():
        status.detail = "auth cookie expired"
    else:
        status.detail = "ok"
    return status


async def validate_session(platform: Platform, settings: Settings) -> SessionStatus:
    """Load the session headlessly and confirm the platform still treats it as logged in."""

    status = inspect_session(platform, settings)
    if not status.exists or not status.has_auth_cookie:
        status.valid = False
        return status
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(**_launch_kwargs(settings, settings.browser.headless))
        context = await browser.new_context(storage_state=str(status.path), **context_kwargs(settings))
        try:
            status.valid, status.detail = await _probe_logged_in(platform, context, settings)
        finally:
            await browser.close()
    return status


async def _probe_logged_in(platform: str, context: BrowserContext, settings: Settings) -> tuple[bool, str]:
    page = await context.new_page()
    await page.goto(HOME_URLS[platform], wait_until="domcontentloaded")
    await asyncio.sleep(3)
    if "/accounts/login" in page.url or "/challenge" in page.url or page.url.rstrip("/").endswith("/login"):
        return False, f"redirected to {page.url}"
    if platform == "instagram":
        result = await page.evaluate(
            """async (appId) => {
                const r = await fetch('/api/v1/accounts/current_user/?edit=true', {
                    credentials: 'include',
                    headers: {'X-IG-App-ID': appId, 'X-Requested-With': 'XMLHttpRequest'}
                });
                let body = null; try { body = await r.json(); } catch (e) {}
                return {status: r.status, body};
            }""",
            settings.scrape.ig_app_id,
        )
        user = (result.get("body") or {}).get("user") or {}
        if result.get("status") == 200 and user.get("username"):
            return True, f"logged in as @{user['username']}"
        return False, f"current_user returned HTTP {result.get('status')}"
    cookie = _auth_cookie(await context.cookies(), platform)
    return (cookie is not None), ("session cookie present" if cookie else "session cookie missing")


def export_netscape_cookies(state_path: Path, out_path: Path) -> Path:
    """Convert a Playwright storage_state file into a Netscape cookies.txt for yt-dlp."""

    state = json.loads(state_path.read_text(encoding="utf-8"))
    lines = ["# Netscape HTTP Cookie File", "# Generated by audit engine - do not share", ""]
    for cookie in state.get("cookies", []):
        domain = cookie.get("domain", "")
        if not domain:
            continue
        include_sub = "TRUE" if domain.startswith(".") else "FALSE"
        secure = "TRUE" if cookie.get("secure") else "FALSE"
        expires = cookie.get("expires", -1)
        expires = int(expires) if expires and expires > 0 else 0
        name = cookie.get("name", "")
        value = str(cookie.get("value", "")).replace("\n", "")
        prefix = "#HttpOnly_" if cookie.get("httpOnly") else ""
        lines.append(
            "\t".join([f"{prefix}{domain}", include_sub, cookie.get("path", "/"), secure, str(expires), name, value])
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(out_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return out_path
