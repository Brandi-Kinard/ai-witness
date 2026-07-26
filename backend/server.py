"""AI-Witness backend — turn uploaded media into an editable news-article draft + hero image.

Web-app + Flask pattern (sibling to Sous-Vision, NOT a native DAT-iOS app). POST one or more
media files (photos and/or videos) to /draft and get back a headline, 2-4 tags, a
multi-paragraph body, and a hero image. The last result is cached in memory AND persisted to
disk so it survives restarts (the Sous-Vision "ripeness bug"). The editor is served at GET /.

Hero image is ALWAYS a genuine captured moment — never AI-generated. A single Gemini draft
call both writes the article AND names the best hero source (hero_choice). Then, deterministic:
  - "photo"       → the uploaded photo's RAW bytes, unaltered ("uploaded_photo").
  - "video_frame" → ffmpeg extracts the exact frame at the named timestamp from that uploaded
                    video ("video_frame_extracted").
An empirical test confirmed the Nano-Banana image model regenerates/restyles scenes rather
than returning real frames even when told not to, so that path was removed entirely.
"""

import base64
import json
import logging
import os
import subprocess
import tempfile
import time

from dotenv import load_dotenv
from flask import Flask, Response, request, send_file, send_from_directory
from flask_cors import CORS
from google import genai
from google.genai import types

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ai-witness")

# AI-Witness uses its OWN key, separate from Sous-Vision's SOUS_VISION_GOOGLE_API_KEY
# and Clutch's GOOGLE_API_KEY.
API_KEY = os.environ.get("AI_WITNESS_GOOGLE_API_KEY")
if not API_KEY:
    logger.warning("AI_WITNESS_GOOGLE_API_KEY is not set — /draft will return 500.")

# Draft model as a single swappable constant (owner may test another tier in one line).
DRAFT_MODEL = "gemini-2.5-flash"

# Inline base64 (Part.from_bytes) supports up to ~100MB (raised from 20MB, Jan 2026).
# Larger files must go through the Files API. Keep a safety margin under 100MB.
INLINE_MAX_BYTES = 90 * 1024 * 1024

HERE = os.path.dirname(os.path.abspath(__file__))
LATEST_PATH = os.path.join(HERE, "latest_cache.json")
HERO_BASENAME = "hero_latest"  # extension chosen per hero_mime

DRAFT_PROMPT = (
    "You are a newsroom video editor. You are given one or more uploaded media files "
    "(photos and/or videos), each labeled [File i] with its 0-based index i and its kind. "
    "Write a publishable news article draft based strictly on what the media actually shows, "
    "and choose the single best REAL hero-image source. Respond ONLY with JSON matching: "
    '{"headline": string, "tags": [2 to 4 short lowercase topic strings], "body": string, '
    '"hero_choice": {"type": "photo" | "video_frame", "source_index": integer, "timestamp": "MM:SS"}}. '
    "The body must be multiple paragraphs in a factual news-article voice (separate paragraphs "
    "with \\n\\n). hero_choice.source_index is the [File i] index of the chosen source. Use "
    '"photo" to use an uploaded photo as-is, or "video_frame" to use a still frame from an '
    'uploaded video; include "timestamp" (MM:SS) ONLY for video_frame, naming the exact best '
    "frame moment. Base everything strictly on the real media — do not invent events."
)


def _load_latest():
    """Load the last persisted draft from disk (survives restarts)."""
    try:
        with open(LATEST_PATH) as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    return None


def _save_latest():
    try:
        with open(LATEST_PATH, "w") as f:
            json.dump(LATEST, f)
    except OSError as e:
        logger.warning("Could not persist LATEST: %s", e)


LATEST = _load_latest()

app = Flask(__name__)
CORS(app)  # parity with Sous-Vision; allows external callers to POST /draft & GET /latest


@app.get("/health")
def health():
    return {"ok": True, "draft_model": DRAFT_MODEL, "key_configured": bool(API_KEY)}


@app.get("/")
def index():
    """Serve the single-page upload + editor."""
    return send_from_directory(HERE, "index.html")


@app.get("/assets/<path:filename>")
def assets(filename):
    """Serve static assets (favicon, icons) from backend/assets/."""
    return send_from_directory(os.path.join(HERE, "assets"), filename)


@app.post("/draft")
def draft():
    """Turn uploaded media into an article draft + a genuine hero image; cache + return it."""
    if not API_KEY:
        return _json_error("AI_WITNESS_GOOGLE_API_KEY not set", 500)

    uploads = request.files.getlist("files") or request.files.getlist("video")
    files = []
    for f in uploads:
        data = f.read()
        if not data:
            continue
        mime = f.mimetype or _guess_mime(f.filename)
        files.append({
            "name": f.filename or "upload",
            "mime": mime,
            "kind": "video" if mime.startswith("video") else "photo",
            "bytes": data,
        })
    if not files:
        return _json_error("no media files in request", 400)

    logger.info("Received %d file(s): %s", len(files),
                ", ".join(f"{x['name']}({x['kind']},{len(x['bytes'])}B)" for x in files))

    started = time.monotonic()
    client = genai.Client(api_key=API_KEY)

    # Build one draft call: a per-file manifest + media part, then the prompt.
    try:
        parts = []
        for i, x in enumerate(files):
            parts.append(types.Part.from_text(text=f"[File {i}] name={x['name']} kind={x['kind']}"))
            parts.append(_media_part(client, x["bytes"], x["mime"], x["name"]))
        parts.append(types.Part.from_text(text=DRAFT_PROMPT))
    except Exception as e:  # noqa: BLE001
        logger.exception("Media prepare failed")
        return _json_error(f"media prepare failed: {e}", 502)

    # ONE Gemini call: article + hero_choice.
    try:
        draft_obj = _run_draft(client, parts)
    except Exception as e:  # noqa: BLE001
        logger.exception("Draft Gemini call failed")
        return _json_error(f"draft call failed: {e}", 502)

    hero_choice = draft_obj.get("hero_choice") if isinstance(draft_obj, dict) else None
    try:
        hero_bytes, hero_mime, hero_source = _resolve_hero(files, hero_choice)
    except Exception as e:  # noqa: BLE001
        logger.exception("Hero resolve failed")
        return _json_error(f"hero resolve failed: {e}", 502)

    elapsed = round(time.monotonic() - started, 2)

    hero_path = os.path.join(HERE, HERO_BASENAME + _ext_for(hero_mime))
    with open(hero_path, "wb") as f:
        f.write(hero_bytes)
    hero_data_url = f"data:{hero_mime};base64," + base64.b64encode(hero_bytes).decode("ascii")

    result = {
        "headline": draft_obj.get("headline") if isinstance(draft_obj, dict) else None,
        "tags": draft_obj.get("tags") if isinstance(draft_obj, dict) else None,
        "body": draft_obj.get("body") if isinstance(draft_obj, dict) else None,
        "hero_choice": hero_choice,
        "hero_source": hero_source,
        "hero_mime": hero_mime,
        "hero_image_data_url": hero_data_url,
        "hero_path": hero_path,
        "draft_model": DRAFT_MODEL,
        "elapsed_seconds": elapsed,
    }
    if isinstance(draft_obj, dict) and "raw" in draft_obj:
        result["raw"] = draft_obj["raw"]

    global LATEST
    LATEST = result
    _save_latest()  # persist so the draft survives backend restarts

    logger.info("draft done in %ss; hero_source=%s; headline=%r",
                elapsed, hero_source, result.get("headline"))
    return Response(json.dumps(result), status=200, mimetype="application/json")


@app.get("/latest")
def latest():
    """Return the last stored draft, or 404 if nothing has been drafted yet."""
    if LATEST is None:
        return _json_error("nothing drafted yet", 404)
    return Response(json.dumps(LATEST), status=200, mimetype="application/json")


@app.get("/hero")
def hero():
    """Return the last hero image bytes with its real mime, or 404 if none yet."""
    if not LATEST or not LATEST.get("hero_path") or not os.path.exists(LATEST["hero_path"]):
        return _json_error("no hero image yet", 404)
    return send_file(LATEST["hero_path"], mimetype=LATEST.get("hero_mime", "image/jpeg"))


# --- Gemini + hero helpers -------------------------------------------------

def _media_part(client, data: bytes, mime: str, filename: str):
    """Return a Part for any media (photo or video). Inline under the limit, else Files API."""
    if len(data) <= INLINE_MAX_BYTES:
        return types.Part.from_bytes(data=data, mime_type=mime)
    logger.info("File over inline limit (%d bytes); using Files API…", len(data))
    with tempfile.NamedTemporaryFile(suffix=_suffix(filename), delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        uploaded = client.files.upload(file=tmp_path)
        while getattr(uploaded.state, "name", uploaded.state) == "PROCESSING":
            time.sleep(1)
            uploaded = client.files.get(name=uploaded.name)
        return types.Part.from_uri(file_uri=uploaded.uri, mime_type=mime)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def _run_draft(client, parts) -> dict:
    resp = client.models.generate_content(
        model=DRAFT_MODEL,
        contents=types.Content(role="user", parts=parts),
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.3,
        ),
    )
    text = resp.text or "{}"
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        logger.warning("Draft model returned non-JSON; wrapping raw text")
        return {"raw": text}


def _resolve_hero(files, hero_choice):
    """Deterministically produce the hero from a REAL source. Returns (bytes, mime, source).

    Never AI-generates. Honors hero_choice when valid; otherwise falls back to the first
    uploaded photo, else a frame from the first uploaded video.
    """
    n = len(files)
    ts = "00:00"
    if isinstance(hero_choice, dict):
        htype = hero_choice.get("type")
        idx = hero_choice.get("source_index")
        ts = hero_choice.get("timestamp") or "00:00"
        if isinstance(idx, int) and 0 <= idx < n:
            f = files[idx]
            if htype == "photo" and f["kind"] == "photo":
                logger.info("Hero: uploaded_photo idx=%d (%s), %d bytes", idx, f["name"], len(f["bytes"]))
                return f["bytes"], f["mime"], "uploaded_photo"
            if htype == "video_frame" and f["kind"] == "video":
                logger.info("Hero: video_frame_extracted idx=%d (%s) @ %s", idx, f["name"], ts)
                return _ffmpeg_frame(f["bytes"], ts), "image/jpeg", "video_frame_extracted"
        logger.warning("hero_choice invalid/mismatched (%s); using deterministic default.", hero_choice)

    # Deterministic default: first photo, else first video frame.
    for f in files:
        if f["kind"] == "photo":
            logger.info("Hero default: first uploaded photo (%s)", f["name"])
            return f["bytes"], f["mime"], "uploaded_photo"
    for f in files:
        if f["kind"] == "video":
            logger.info("Hero default: frame from first video (%s) @ %s", f["name"], ts)
            return _ffmpeg_frame(f["bytes"], ts), "image/jpeg", "video_frame_extracted"
    raise ValueError("no usable hero source among uploaded files")


def _ffmpeg_frame(video_bytes: bytes, hero_ts: str) -> bytes:
    """Extract a single JPEG frame at hero_ts using the bundled imageio-ffmpeg binary."""
    import imageio_ffmpeg

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    ts = _normalize_ts(hero_ts)
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as vtmp:
        vtmp.write(video_bytes)
        vpath = vtmp.name
    opath = vpath + ".jpg"
    try:
        subprocess.run(
            [ffmpeg, "-y", "-ss", ts, "-i", vpath, "-frames:v", "1", "-q:v", "2", opath],
            capture_output=True, check=True,
        )
        with open(opath, "rb") as f:
            return f.read()
    finally:
        for p in (vpath, opath):
            try:
                os.unlink(p)
            except OSError:
                pass


# --- small utilities -------------------------------------------------------

def _normalize_ts(ts: str) -> str:
    """Accept 'MM:SS' or 'HH:MM:SS' or seconds; return an ffmpeg-friendly timestamp."""
    ts = (ts or "").strip()
    return ts if ts else "0"


def _ext_for(mime: str) -> str:
    return {"image/jpeg": ".jpg", "image/jpg": ".jpg", "image/png": ".png",
            "image/webp": ".webp", "image/gif": ".gif"}.get((mime or "").lower(), ".img")


def _guess_mime(filename: str) -> str:
    ext = _suffix(filename).lower()
    return {
        ".mp4": "video/mp4", ".mov": "video/quicktime", ".webm": "video/webm",
        ".m4v": "video/mp4", ".avi": "video/x-msvideo", ".mkv": "video/x-matroska",
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
        ".webp": "image/webp", ".gif": "image/gif", ".heic": "image/heic",
    }.get(ext, "application/octet-stream")


def _suffix(filename: str) -> str:
    _, ext = os.path.splitext(filename or "")
    return ext or ".bin"


def _json_error(message: str, status: int) -> Response:
    return Response(json.dumps({"error": message}), status=status, mimetype="application/json")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8081"))
    logger.info("AI-Witness backend on http://0.0.0.0:%d  (/ /draft /latest /hero /health)", port)
    app.run(host="0.0.0.0", port=port)
