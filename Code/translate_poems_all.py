"""
=============================================================================
Urdu Ghazal Translation -- Multi-LLM Master Script
DATS 6312 Final Project: Style-Preserving Hindi/Urdu to English Translation
=============================================================================
Runs ALL 5 LLM translators in a single script, adding keys:
    en_groq      -- LLaMA 3.3 70B via Groq
    en_gemini    -- Gemini 1.5 Flash via Google AI Studio
    en_meta      -- LLaMA 3.1 8B via HuggingFace Inference API
    en_grok      -- Grok-2 via xAI API
    en_mistral   -- Mistral Large via Mistral AI

CRASH-SAFE RESUME:
    Progress is saved after EVERY single poem using an atomic write
    (writes to .tmp then renames). If the terminal is interrupted at
    any point, just rerun the exact same command — already-translated
    poems are detected and skipped automatically.

API KEYS (set as environment variables before running):
    export GROQ_API_KEY="gsk_..."         # groq.com            -- free
    export GEMINI_API_KEY="AI..."         # aistudio.google.com -- free
    export AWS-NLP-HF="hf_..."            # huggingface.co      -- free
    export XAI_API_KEY="xai-..."          # console.x.ai        -- free
    export MISTRAL_API_KEY="..."          # console.mistral.ai  -- free

INSTALL:
    pip install groq google-generativeai mistralai

USAGE:
    # Test mode -- 2 poems per LLM, fast
    python translate_poems_all.py --input data.json --test

    # Full run -- all 1294 poems
    python translate_poems_all.py --input data.json --all

    # Resume after interruption (same command, skips done poems automatically)
    python translate_poems_all.py --input data.json --all

    # Run specific LLMs only
    python translate_poems_all.py --input data.json --all --only en_groq en_gemini

OUTPUT:
    Test: .../Results/outputs/poetry_data_test.json
    Full: .../Results/outputs/poetry_data_translated.json
=============================================================================
"""

import os
import json
import time
import argparse
import logging
import urllib.request
import urllib.error
from pathlib import Path
from datetime import datetime

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# ── Project base ──────────────────────────────────────────────────────────────
PROJECT_BASE = Path("/home/ubuntu/Natural Language Processing - DATS6312/Final-Project-Group2")

# ── Load .env file ────────────────────────────────────────────────────────────
def load_env(env_path: Path = PROJECT_BASE / ".env") -> None:
    """
    Load API keys from .env file into os.environ.
    Handles INI-style sections [section_name] — loads keys from ALL sections.
    Skips blank lines, comments, section headers, and empty values.
    Does not override keys already set in the environment.
    """
    if not env_path.exists():
        logger.warning(f".env not found at {env_path} — relying on existing env vars")
        return
    loaded = []
    skipped_empty = []
    with open(env_path, "r") as f:
        for line in f:
            line = line.strip()
            # skip blanks, comments, and INI section headers like [default]
            if not line or line.startswith("#") or line.startswith("["):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key   = key.strip()
            value = value.strip().strip('"').strip("'")
            if not key:
                continue
            if not value:
                skipped_empty.append(key)   # key exists but value is blank
                continue
            if key not in os.environ:
                os.environ[key] = value
                loaded.append(key)
    if loaded:
        logger.info(f"Loaded {len(loaded)} API keys from .env: {loaded}")
    if skipped_empty:
        logger.warning(f"Skipped {len(skipped_empty)} empty keys in .env: {skipped_empty}")

load_env()
RESULTS_DIR  = PROJECT_BASE / "Results" / "outputs"
TEST_OUTPUT  = RESULTS_DIR / "poetry_data_test.json"
FULL_OUTPUT  = RESULTS_DIR / "poetry_data_translated.json"

# ── Input file (already has en_t and en_anthropic) ────────────────────────────
# Single source of truth for both test and full runs
DATA_DIR     = PROJECT_BASE / "Data" / "raw" / "dataset - urdu-ghazal-dataset-32-poets-and-their-ghazals"
INPUT_FILE   = DATA_DIR / "poetry_complete.json"
TEST_INPUT   = INPUT_FILE
FULL_INPUT   = INPUT_FILE

# ── Shared system prompt ──────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a poetic translator specializing in Hindi/Urdu to English translation.
Translate the given Urdu/Hindi poem lines into English while:
1. Preserving the poetic meaning and emotional depth
2. Maintaining the same number of lines as the input
3. Keeping rhyme schemes and rhythm where possible without sacrificing meaning
4. Preserving cultural imagery, metaphors, and the ghazal's romantic/melancholic tone

Return ONLY a JSON array of translated strings, one per input line.
No explanations, no markdown fences, no extra text. Just the raw JSON array."""


# =============================================================================
# HELPERS
# =============================================================================

def parse_response(raw: str) -> list:
    """
    Robustly parse a JSON array from LLM response.
    Handles: markdown fences, extra preamble text, trailing commas,
    and cases where the model wraps the array in extra prose.
    """
    raw = raw.strip()

    # Strip markdown fences
    if "```" in raw:
        parts = raw.split("```")
        # Take the content inside the first fence pair
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("["):
                raw = part
                break

    # Find the JSON array bounds — handles preamble/postamble prose
    start = raw.find("[")
    end   = raw.rfind("]")
    if start != -1 and end != -1 and end > start:
        raw = raw[start:end+1]

    return json.loads(raw.strip())


def align_lines(en_lines: list, hi_lines: list, slug: str) -> list:
    """Ensure translated line count matches source line count."""
    if len(en_lines) != len(hi_lines):
        logger.warning(f"    Line mismatch [{slug[:40]}]: {len(hi_lines)} hi vs {len(en_lines)} en — fixing")
        while len(en_lines) < len(hi_lines):
            en_lines.append("")
        en_lines = en_lines[:len(hi_lines)]
    return en_lines


def save_checkpoint(data: dict, path: Path, silent: bool = False) -> None:
    """
    Atomically save progress to disk after every poem.

    Writes to a .tmp file first, then renames it to the real path.
    This means a crash mid-write can never corrupt the saved file —
    you either have the old version or the new version, never a broken one.
    On rerun, already-translated poems are skipped via the key-exists check.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp_path.replace(path)          # atomic rename — safe on Linux/macOS
    if not silent:
        logger.info(f"  Checkpoint -> {path}")


# =============================================================================
# LLM TRANSLATORS
# =============================================================================

def translate_groq(lines: list) -> list:
    """LLaMA 3.3 70B via Groq."""
    try:
        from groq import Groq
    except ImportError:
        raise ImportError("Run: pip install groq")
    key = os.environ.get("GROQ_API_KEY", "")
    if not key:
        raise ValueError("GROQ_API_KEY not set")
    client = Groq(api_key=key)
    resp   = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": json.dumps(lines, ensure_ascii=False)},
        ],
        temperature=0.7,
        max_tokens=4000,
    )
    return parse_response(resp.choices[0].message.content)


def translate_gemini(lines: list) -> list:
    """Gemini 2.0 Flash via Google AI Studio (new google.genai SDK).

    Uses gemini-2.0-flash (non-thinking) for reliable JSON structured output.
    gemini-2.5-flash is a reasoning model that interleaves thinking tokens
    which corrupt JSON parsing — 2.0-flash is the correct choice here.
    """
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        raise ImportError("Run: pip install google-genai")
    key = os.environ.get("GEMINI_API_KEY", "")
    if not key:
        raise ValueError("GEMINI_API_KEY not set")

    client = genai.Client(api_key=key)
    resp = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=json.dumps(lines, ensure_ascii=False),
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.7,
            max_output_tokens=4000,
        ),
    )
    # Extract text safely from response parts (avoids thinking token bleed)
    raw_text = ""
    if resp.candidates and resp.candidates[0].content and resp.candidates[0].content.parts:
        for part in resp.candidates[0].content.parts:
            if hasattr(part, "text") and part.text:
                raw_text += part.text
    else:
        raw_text = resp.text or ""
    return parse_response(raw_text)


def translate_meta(lines: list) -> list:
    """Meta Llama 3.3 70B via HuggingFace Inference API.

    IMPORTANT: You must accept the model license on HuggingFace first:
    1. Go to huggingface.co/meta-llama/Llama-3.3-70B-Instruct
    2. Click "Expand to review and access model files"
    3. Accept Meta license (approval is usually instant)
    4. Then this will work with your AWS-NLP-HF token
    """
    key = os.environ.get("AWS-NLP-HF", "")
    if not key:
        raise ValueError("AWS-NLP-HF not set in .env")
    model = "meta-llama/Llama-3.3-70B-Instruct"
    url   = f"https://api-inference.huggingface.co/models/{model}/v1/chat/completions"
    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": json.dumps(lines, ensure_ascii=False)},
        ],
        "temperature": 0.7,
        "max_tokens":  4000,
    }).encode("utf-8")
    req = urllib.request.Request(
        url, data=payload,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            result = json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        if e.code == 403:
            raise ValueError(
                "HF 403 — you need to accept the Llama license at "
                "huggingface.co/meta-llama/Llama-3.3-70B-Instruct first"
            )
        if e.code == 404:
            raise ValueError(
                "HF 404 — model not available on free inference tier. "
                "Accept license first, or use Together AI as fallback."
            )
        raise RuntimeError(f"HF API error {e.code}: {body}")
    return parse_response(result["choices"][0]["message"]["content"])


def translate_grok(lines: list) -> list:
    """Grok-4.20 via xAI API."""
    key = os.environ.get("XAI_API_KEY", "")
    if not key:
        raise ValueError("XAI_API_KEY not set")
    payload = json.dumps({
        "model": "grok-4.20",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": json.dumps(lines, ensure_ascii=False)},
        ],
        "temperature": 0.7,
        "max_tokens":  4000,
    }).encode("utf-8")
    req = urllib.request.Request(
        "https://api.x.ai/v1/chat/completions", data=payload,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            result = json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        if e.code == 403:
            raise ValueError("xAI API 403 Forbidden — verify XAI_API_KEY at console.x.ai and ensure API access is enabled")
        if e.code == 401:
            raise ValueError("xAI API 401 Unauthorized — check XAI_API_KEY in .env")
        raise RuntimeError(f"xAI error {e.code}: {body}")
    return parse_response(result["choices"][0]["message"]["content"])


def translate_mistral(lines: list) -> list:
    """Mistral Large via Mistral AI API."""
    key = os.environ.get("MISTRAL_API_KEY", "")
    if not key:
        raise ValueError("MISTRAL_API_KEY not set")
    payload = json.dumps({
        "model": "mistral-large-latest",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": json.dumps(lines, ensure_ascii=False)},
        ],
        "temperature": 0.7,
        "max_tokens":  4000,
    }).encode("utf-8")
    req = urllib.request.Request(
        "https://api.mistral.ai/v1/chat/completions", data=payload,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        result = json.loads(r.read().decode("utf-8"))
    return parse_response(result["choices"][0]["message"]["content"])


# =============================================================================
# LLM REGISTRY
# (output_key, translate_fn, rate_limit_sleep_seconds)
# =============================================================================
LLM_REGISTRY = [
    ("en_groq",    translate_groq,    2.0),   # Meta Llama 3.3 70B via Groq -- free
    ("en_gemini",  translate_gemini,  4.0),   # Gemini 2.0 Flash via Google -- free tier
    ("en_mistral", translate_mistral, 3.0),   # Mistral Large via Mistral AI -- free (rate limit: ~20 RPM)
    # en_meta removed: same model as en_groq (Llama 3.3 70B)
    # en_grok removed: xAI requires billing setup
]


# =============================================================================
# CORE PROCESSING
# =============================================================================

def run_llm(data: dict, key: str, translate_fn, sleep: float,
            out_path: Path, test_mode: bool, only: list) -> tuple:
    """
    Run one LLM over all poems.

    Saves progress after EVERY poem so interruptions never lose work.
    On rerun, poems that already have `key` are silently skipped.
    """
    if only and key not in only:
        logger.info(f"\n[{key}] Skipped (not in --only list)")
        return data, {"translated": 0, "skipped": 0, "errors": 0}

    logger.info(f"\n{'='*60}")
    logger.info(f"  Running: {key.upper()}")
    logger.info(f"{'='*60}")

    translated = 0
    skipped    = 0
    errors     = 0
    poems_done = 0

    for poet, ghazals in data.items():
        for slug, poem in ghazals.items():
            hi_lines = poem.get("hi")

            # ── Skip: no Urdu text ────────────────────────────────────────────
            if not hi_lines:
                skipped += 1
                continue

            # ── Skip: already translated (crash-safe resume) ──────────────────
            if key in poem:
                skipped += 1
                continue

            # ── Translate ─────────────────────────────────────────────────────
            try:
                en_lines       = translate_fn(hi_lines)
                en_lines       = align_lines(en_lines, hi_lines, slug)
                poem[key]      = en_lines
                translated    += 1
                poems_done    += 1

                logger.info(f"  [{translated:>4}] {poet[:20]:<20} | {slug[:45]}")

                # ── Save after every poem (atomic write) ──────────────────────
                save_checkpoint(data, out_path, silent=True)

                time.sleep(sleep)

            except ValueError as e:
                # Config errors (missing key, bad setup) — stop this LLM immediately
                logger.error(f"  [{key}] Config error: {e} — skipping this LLM entirely")
                return data, {"translated": translated, "skipped": skipped, "errors": errors + 1}

            except Exception as e:
                err_str = str(e)

                # ── Groq daily token limit (TPD) ──────────────────────────────
                if "429" in err_str and "tokens per day" in err_str.lower():
                    import re
                    delay_match = re.search(r"try again in ([0-9]+)m([0-9.]+)s", err_str)
                    if delay_match:
                        wait_mins  = int(delay_match.group(1))
                        wait_secs  = float(delay_match.group(2))
                        wait_total = int(wait_mins * 60 + wait_secs) + 10
                    else:
                        wait_total = 2100   # default 35 min
                    logger.warning(f"  [{key}] Daily token limit hit — waiting {wait_total//60}m {wait_total%60}s for reset...")
                    time.sleep(wait_total)
                    logger.info(f"  [{key}] Resuming after rate limit wait")

                # ── Gemini daily quota exhausted ──────────────────────────────
                elif "429" in err_str and "RESOURCE_EXHAUSTED" in err_str:
                    if "per_day" in err_str.lower() or "limit: 0" in err_str.lower() or "GenerateRequestsPerDay" in err_str:
                        logger.warning(f"  [{key}] Daily quota exhausted — skipping remaining poems for this LLM")
                        logger.warning(f"  [{key}] Quota resets at midnight Pacific time. Rerun to continue.")
                        return data, {"translated": translated, "skipped": skipped, "errors": errors}
                    else:
                        import re
                        delay_match = re.search(r"retry.*?([0-9]+)s", err_str)
                        wait = int(delay_match.group(1)) + 5 if delay_match else 30
                        logger.warning(f"  [{key}] Rate limited — waiting {wait}s before retry...")
                        time.sleep(wait)
                        errors += 1

                # ── Generic per-minute rate limit ─────────────────────────────
                elif "429" in err_str:
                    import re
                    delay_match = re.search(r"([0-9]+)s", err_str)
                    wait = int(delay_match.group(1)) + 5 if delay_match else 60
                    logger.warning(f"  [{key}] Rate limited — waiting {wait}s...")
                    time.sleep(wait)
                    errors += 1

                else:
                    logger.error(f"  ERROR [{slug[:40]}]: {e}")
                    errors += 1
                    time.sleep(sleep * 3)

            # ── Test mode: stop after 2 poems per LLM ────────────────────────
            if test_mode and poems_done >= 2:
                logger.info(f"  Test mode: stopping {key} after 2 poems")
                break

        if test_mode and poems_done >= 2:
            break

    stats = {"translated": translated, "skipped": skipped, "errors": errors}
    logger.info(f"\n  [{key}] Done — translated: {translated} | skipped: {skipped} | errors: {errors}")
    return data, stats


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Multi-LLM Urdu ghazal translator — crash-safe resume")
    parser.add_argument("--test",  action="store_true", help="Test mode: 2 poems per LLM, uses poetry_data_test.json")
    parser.add_argument("--all",   action="store_true", help="Full run: all poems, uses poetry_data_full.json")
    parser.add_argument("--only",  nargs="+", default=[], metavar="KEY",
                        help="Run only specific LLMs e.g. --only en_groq en_gemini")
    args = parser.parse_args()

    if not args.test and not args.all:
        parser.error("Specify --test (2 poems) or --all (full dataset)")

    # ── Paths (all hardcoded — no --input needed) ─────────────────────────────
    input_path = TEST_INPUT  if args.test else FULL_INPUT
    out_path   = TEST_OUTPUT if args.test else FULL_OUTPUT
    mode_label = "TEST (2 poems per LLM)" if args.test else "FULL RUN (all poems)"

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info(f"  Multi-LLM Translation — {mode_label}")
    logger.info(f"  Input:    {input_path}")
    logger.info(f"  Output:   {out_path}")
    logger.info(f"  LLMs:     {[k for k, _, _ in LLM_REGISTRY]}")
    logger.info(f"  Auto-saves after EVERY poem (crash-safe)")
    logger.info("=" * 60)

    # ── Load — prefer output file if it exists (resume from last run) ─────────
    resume_source = out_path if out_path.exists() else input_path
    if out_path.exists():
        logger.info(f"Resuming from existing output: {out_path}")
    else:
        logger.info(f"Starting fresh from: {input_path}")

    with open(resume_source, "r", encoding="utf-8") as f:
        data = json.load(f)

    total_poems = sum(len(ghazals) for ghazals in data.values())
    logger.info(f"Loaded {len(data)} poets, {total_poems} poems\n")

    # ── Run each LLM ──────────────────────────────────────────────────────────
    all_stats  = {}
    start_time = datetime.now()

    for output_key, translate_fn, sleep_secs in LLM_REGISTRY:
        data, stats = run_llm(
            data         = data,
            key          = output_key,
            translate_fn = translate_fn,
            sleep        = sleep_secs,
            out_path     = out_path,
            test_mode    = args.test,
            only         = args.only,
        )
        all_stats[output_key] = stats
        # Explicit end-of-LLM checkpoint (non-silent)
        save_checkpoint(data, out_path, silent=False)

    # ── Final summary ─────────────────────────────────────────────────────────
    elapsed = datetime.now() - start_time
    logger.info("\n" + "=" * 60)
    logger.info("  FINAL SUMMARY")
    logger.info("=" * 60)
    for key, s in all_stats.items():
        logger.info(f"  {key:<15} translated: {s['translated']:>4} | skipped: {s['skipped']:>4} | errors: {s['errors']:>3}")
    logger.info(f"\n  Total time : {elapsed}")
    logger.info(f"  Output     : {out_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()