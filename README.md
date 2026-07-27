# AI-Witness

Upload a video → get an editable news-article draft (headline, tags, body) plus an
extracted hero image. A web-app + Flask backend, sibling to Sous-Vision (deliberately
**not** a native DAT-iOS app — same pattern that worked for Kitchen).

## How it works

`POST /draft` (multipart `video`) runs two Gemini calls:

1. **Draft** — a video-understanding model (`gemini-2.5-flash`, swappable) returns
   `{headline, tags, body, hero_timestamp}` as JSON.
2. **Hero image** — probes whether the Nano-Banana image model (`gemini-3.1-flash-image`)
   returns real image *bytes* from **video** input directly (`hero_source: "direct_model"`).
   If it does not, it falls back to `ffmpeg` extracting the exact frame at
   `hero_timestamp` as a real JPEG (`hero_source: "ffmpeg_fallback"`).

The result is cached in memory **and** persisted to `latest_cache.json`, so it survives a
restart (the Sous-Vision "ripeness bug": an in-memory-only cache 404s after restart).

## Endpoints

| Route     | Method | Purpose                                              |
|-----------|--------|------------------------------------------------------|
| `/`       | GET    | Upload + editor page                                 |
| `/draft`  | POST   | `video` file → article draft + hero image (JSON)     |
| `/latest` | GET    | Last stored draft, or 404 if none yet                |
| `/hero`   | GET    | Last hero JPEG bytes                                 |
| `/health` | GET    | `{ok, draft_model, image_model, key_configured}`     |

## Run

```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # then set AI_WITNESS_GOOGLE_API_KEY
python server.py              # http://0.0.0.0:8081
```

## API key — the "three places" (and separation)

`AI_WITNESS_GOOGLE_API_KEY` is a **separate** key from Sous-Vision's
`SOUS_VISION_GOOGLE_API_KEY` and Clutch's `GOOGLE_API_KEY`. It appears in exactly three
places, all consistent:

1. `.env.example` — documented template with the "separate key" note.
2. `server.py` — read via `os.environ.get(...)`, warns at startup if missing, and
   re-checks before every Gemini call (returns 500 if unset).
3. `GET /health` — reports `key_configured` for a quick diagnostic.
© 2026 Brandi Kinard. Originally authored and first published July 26, 2026.
