# Cross-Platform Creator Profile & Deep Persona Audit Engine

Given an Instagram and/or TikTok handle, this engine collects the creator's complete public
profile footprint in an authenticated browser session. It then distills every video and photo
locally (keyframe contact sheets and speech transcripts) and runs a two-stage Claude
map-reduce that writes a 10-section **Persona, Aesthetic, Lifestyle & Content Intelligence
Report**: `reports/{handle}_full_report.md`.

```text
audit auth instagram        # one-time headed login → auth/sessions/ig_state.json
audit auth tiktok           # one-time headed login → auth/sessions/tiktok_state.json
audit run --ig some.creator --tiktok somecreator
```

---

## What gets collected

| Platform | Surface | How |
|---|---|---|
| Instagram | Bio, metrics, links, category, HD avatar | `web_profile_info` called from inside the logged-in page |
| | Every feed post: photos, multi-slide carousels, Reels (captions, timestamps, location labels, metrics, comment sample) | `/api/v1/feed/user/` pagination, plus the Reels tab (`/api/v1/clips/user/`) for Reels not shared to the grid |
| | Every active Story slide (stickers, links, polls, music) | `reels_media`; falls back to stepping through the story viewer |
| | Every Highlight tray and every slide in it | `highlights_tray` + batched `reels_media`; falls back to the viewer |
| | Every post in `/{username}/tagged/` | `/api/v1/usertags/` pagination |
| TikTok | Bio, metrics, bio link, avatar | Page rehydration JSON (`__UNIVERSAL_DATA_FOR_REHYDRATION__`) |
| | Every original post (video or photo-mode, cover, caption, audio, stats) | Human-paced scrolling while intercepting `/api/post/item_list/` |
| | Every item on the **Reposts** tab | Opens the tab and intercepts `/api/repost/item_list/` |

Every JSON response the page receives is also captured through `page.on("response")` hooks and
written to `data/{run}/raw/{platform}/network_capture.jsonl`. When an internal endpoint changes
shape, the Instagram scraper falls back to scrolling the relevant tab and pulling media objects
out of whatever the web app fetched itself, whether REST or GraphQL. TikTok item lists are only
ever read from intercepted traffic, because those endpoints need request signatures that only
the web app can compute.

Media files are downloaded through the browser context (same cookies and user agent). TikTok
videos are fetched with `yt-dlp`, using cookies exported from the saved session, with a direct
CDN download as fallback.

## Pipeline

```text
          ┌──────────── scrape ────────────┐   ┌──────── process ────────┐   ┌───────────────── report ─────────────────┐
 handle → │ Playwright + storage_state     │ → │ ffmpeg scene keyframes  │ → │ MAP: Claude vision, batched per surface  │
          │ network hooks, paced scrolling │   │ → labeled contact sheet │   │      → structured ItemObservations       │
          │ downloads (CDN / yt-dlp)       │   │ ffmpeg audio → whisper  │   │ REDUCE 1: per-surface SurfaceDigests     │
          │ normalizer → ProfileManifest   │   │ Pillow photo normalize  │   │ REDUCE 2: 10 sections over a cached      │
          └────────────────────────────────┘   └─────────────────────────┘   │           evidence pack → Markdown       │
                                                                             └──────────────────────────────────────────┘
                         every step checkpointed in SQLite (data/cache.sqlite3); re-run any command to resume
```

* **Contact sheets.** Each video gives 4-6 keyframes: an opening "hook" frame, then the
  strongest ffmpeg scene cuts (sampled just after each cut to avoid transitions). If budget
  remains, the longest unsampled stretches are filled. Frames are tiled into one JPEG with
  timestamp labels, at up to 2400px on the long edge so burned-in captions stay legible.
  Carousels and photo-mode posts become labeled slide sheets.
* **Transcripts.** Audio is demuxed to 16 kHz mono WAV and transcribed with `faster-whisper`
  on your machine. Segments that look like music or silence are dropped.
* **Map stage.** Items are packed into batches capped by item count, image count and request
  size. Each item is sent as a text header (caption, hashtags, audio, metrics, stickers,
  comment sample, transcript) followed by its image. The model returns a validated
  `BatchObservation` through structured outputs. A batch the API rejects or the model declines
  is split until the problem item is isolated.
* **Reduce stage.** Statistics (cadence, engagement rate, medians, hashtags, audio, top
  performers) are computed locally. Per-surface digests condense the observations. One
  evidence pack is sent as a cached prompt prefix: profiles, stats, digests and all
  observations. Each of sections 2-10 is a separate request against that prefix; the first
  warms the cache and the rest run concurrently. The Executive Summary is written last, from
  the finished sections.

### The 10 report sections

1. Executive Summary
2. Profile Footprint & Cross-Platform Identity
3. Content Pillars & Format Strategy
4. Visual Aesthetic & Production Style
5. Voice, Tone & On-Camera Persona
6. Lifestyle & Interest Signals
7. Curation & Amplification: Stories, Highlights & Reposts
8. Community Perception, Tagged Content & Audience Engagement
9. Brand Partnerships, Commercial Signals & Brand Safety
10. Strategic Opportunities, Risks & Recommendations

The appendices cover data coverage, engagement tables, hashtags and audio, and methodology and
limitations. A companion `{handle}_item_index.csv` maps every cited item id to its date,
surface and URL.

---

## Setup

Requirements: Python 3.11+, `ffmpeg`/`ffprobe` on `PATH`, and Chromium for Playwright.

```bash
python -m venv .venv
```

```bash
source .venv/bin/activate
```

```bash
pip install -e ".[dev]"
```

```bash
playwright install chromium
```

```bash
cp .env.example .env
```

Set `ANTHROPIC_API_KEY` in `.env`, or use an `ant auth login` profile, which the SDK picks up
automatically. `audit doctor` checks binaries, sessions and credentials.

If Playwright's bundled Chromium can't be installed, point the engine at any Chromium build
with `AUDIT_BROWSER__EXECUTABLE_PATH=/path/to/chrome`.

### Models

| Stage | Setting | Default |
|---|---|---|
| Map (vision batches) | `AUDIT_ANTHROPIC__VISION_MODEL` / `__VISION_EFFORT` | `claude-sonnet-5-5` / `medium` |
| Reduce (digests + sections) | `AUDIT_ANTHROPIC__SYNTHESIS_MODEL` / `__SYNTHESIS_EFFORT` | `claude-sonnet-5-5` / `high` |

`claude-sonnet-4-20250514` was retired in June 2026, so the defaults use the current Sonnet.
Any vision-capable model id works, e.g. `claude-opus-5-5` for the synthesis stage. Requests
opt into server-side refusal fallbacks (`fallbacks="default"`); set
`AUDIT_ANTHROPIC__REFUSAL_FALLBACKS=false` to turn that off. Token usage, cache hits and
refusals are printed at the end of every `report` run.

## Usage

```bash
audit auth instagram
```

```bash
audit auth tiktok --check
```

```bash
audit scrape --ig some.creator --tiktok somecreator --max-items 200
```

```bash
audit process --name some.creator --whisper-model small
```

```bash
audit report --name some.creator
```

```bash
audit run --ig some.creator --tiktok somecreator --headed
```

```bash
audit status some.creator
```

```bash
audit purge some.creator --yes
```

* `--surfaces ig_post,ig_reel,ig_story,ig_highlight,ig_tagged,tt_video,tt_repost` limits collection.
* `--headed` shows the browser. TikTok sometimes shows a verification puzzle; in headed mode
  the scraper pauses so you can solve it by hand.
* Every stage is checkpointed. After an interruption or a rate-limit stop, run the same
  command again and finished work is reused. `--refresh` forces re-collection.
* `--name` sets the run/report name. It defaults to the Instagram handle, otherwise the
  TikTok handle.

Output layout:

```text
data/{run}/raw/{platform}/scrape_result.json     raw platform payloads + asset references
data/{run}/raw/{platform}/network_capture.jsonl  every captured JSON response
data/{run}/media/{platform}/{surface}/{id}/      downloaded photos / videos
data/{run}/derived/                              contact sheets, normalized images
data/{run}/manifest.json                         canonical cross-platform ProfileManifest
data/{run}/observations.jsonl                    map-stage observations
data/{run}/evidence_pack.txt                     exact evidence the sections were written from
reports/{run}_full_report.md                     the report
reports/{run}_item_index.csv                     citation lookup table
```

## Responsible use

This is a creator-content analysis tool, and some of its limits are built in on purpose:

* **Public creator accounts only.** Private profiles are refused on both platforms. The one
  exception is the operator's own Instagram account, detected through the session's
  `ds_user_id`.
* **Data minimization.** Normalization drops GPS coordinates and commenter identities.
  Comment text is kept only as an audience-sentiment sample, and the analysis stages never
  see the raw payloads.
* **Report guardrails.** Every prompt tells the model not to infer sensitive characteristics
  (health, sexuality, religion, politics, ethnicity, finances and so on), not to estimate age
  or comment on bodies, not to work out where someone lives or their routines and
  whereabouts, and not to profile third parties or attempt facial recognition. Locations
  appear only at city or country level, as content themes.
* **Retention.** Collected media and raw payloads stay on your machine. `audit purge` deletes
  a run's data and checkpoints.
* **Platform terms and law.** Automated collection may conflict with Instagram's and
  TikTok's terms of service. Data protection laws (GDPR, CCPA and others) apply to personal
  data even when it is public. You are responsible for having a legitimate purpose (for
  example, creator vetting with the creator's knowledge), a lawful basis, and a retention
  policy. Keep the request pacing defaults conservative: the rate limits are there to avoid
  putting load on the platforms, not to evade detection.

Session files in `auth/sessions/` are full login credentials. They are written with `0600`
permissions and git-ignored. Never commit or share them.

## Development

```bash
pytest -q
```

```bash
ruff check . && ruff format --check .
```

The test suite covers:

* normalization and data minimization;
* the SQLite checkpoint store and the session/cookie tooling;
* ffmpeg keyframe selection and contact sheets on synthesized videos;
* both Claude stages, using a fake client (batching, refusal splitting, id repair, caching,
  the cached evidence prefix);
* the CLI end to end.

`tests/test_scrapers_browser.py` runs the real Instagram and TikTok scrapers in Chromium. It
uses Playwright request routing to simulate both platforms from fixtures, so no live site is
contacted. Tests marked `browser` need a launchable Chromium; tests marked `whisper` download
the `tiny` model.

Repository layout:

```text
config/settings.py            Pydantic settings (paths, pacing, media, whisper, models)
config/prompts.py             vision batch, digest, section and executive-summary prompts
auth/session_manager.py       headed login, storage_state persistence, cookie export
scrapers/base_scraper.py      Playwright controller: hooks, pacing, in-page fetch, downloads
scrapers/instagram_scraper.py posts, Reels, Stories, Highlights, Tagged
scrapers/tiktok_scraper.py    Videos + Reposts tabs via XHR interception, yt-dlp fetcher
pipeline/media_processor.py   keyframes, contact sheets, audio demux, image normalization
pipeline/transcriber.py       faster-whisper wrapper
pipeline/normalizer.py        raw payloads → canonical ProfileManifest
intelligence/schemas.py       canonical records, observations, digests, report sections
intelligence/vision_analyzer.py  map stage + ClaudeGateway (fallbacks, refusals, usage)
intelligence/report_synthesizer.py  stats, digests, evidence pack, sections, rendering
storage/cache_manager.py      SQLite checkpoints
cli.py                        Typer + Rich CLI
```
