"""
Translate Hindi/Urdu poem lines to English using Claude via AWS Bedrock.
Uses both "hi" (Devanagari) and "en" (romanization) as input for richer context.
Adds results as "en_anthropic" key — separate from "en_t" (Google Translate baseline).

Install:
    pip install boto3

Requires:
    .env file at project root containing AWS credentials under section:
    [240143401157_CCAS-DATS-NLP-Student]
    aws_access_key_id=...
    aws_secret_access_key=...
    aws_session_token=...

Usage:
    python translate_poems_anthropic.py --test      # 2 poems → test_results/output/
    python translate_poems_anthropic.py --all       # all poems → Results/outputs/ (resumable)
    python translate_poems_anthropic.py --poet ahmad-faraz --test
"""

import json
import time
import argparse
import os
import boto3
from configparser import ConfigParser, ExtendedInterpolation

# ── Default paths ─────────────────────────────────────────────────────────────
BASE_DIR    = "/home/ubuntu/Natural Language Processing - DATS6312/Final-Project-Group2"
INPUT_PATH  = os.path.join(
    BASE_DIR,
    "Data/processed/poetry_data_translated.json"
)
TEST_OUTPUT = os.path.join(BASE_DIR, "Results/outputs/poetry_data_test.json")
FULL_OUTPUT = os.path.join(BASE_DIR, "Results/outputs/poetry_data_translated.json")

ENV_FILE    = os.path.join(BASE_DIR, ".env")
AWS_SECTION = "240143401157_CCAS-DATS-NLP-Student"
# ─────────────────────────────────────────────────────────────────────────────

MODEL_ID   = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
OUTPUT_KEY = "en_anthropic"

SYSTEM_PROMPT = """You are an expert translator specializing in Hindi/Urdu classical poetry (ghazals, nazms).
Your task is to translate poem lines into English while:
1. Preserving the emotional depth and poetic meaning
2. Maintaining the rhyme scheme where possible (ghazals have a strict radif/qafia pattern)
3. Keeping the same number of lines as the input
4. Preserving cultural metaphors and imagery (candle/moth, wine/beloved, etc.)
5. Capturing the rhythm and cadence of the original

You will receive:
- Hindi (Devanagari script): the poem lines
- Romanization: the same lines in Roman script with pronunciation markers — use this for rhyme and rhythm cues

Return ONLY a valid JSON array of translated strings, one string per input line.
No explanations, no markdown fences, no extra text. Just the JSON array."""


def build_bedrock_client():
    """Read credentials from .env (ini-format) and return a Bedrock client."""
    if not os.path.exists(ENV_FILE):
        raise FileNotFoundError(f".env file not found at: {ENV_FILE}")

    config = ConfigParser(interpolation=ExtendedInterpolation())
    config.read(ENV_FILE)

    if AWS_SECTION not in config:
        raise KeyError(
            f"Section [{AWS_SECTION}] not found in .env\n"
            f"Available sections: {config.sections()}"
        )

    creds = config[AWS_SECTION]
    session = boto3.Session(
        aws_access_key_id     = creds["aws_access_key_id"],
        aws_secret_access_key = creds["aws_secret_access_key"],
        aws_session_token     = creds["aws_session_token"],
    )
    return session.client("bedrock-runtime", region_name="us-east-1")


def build_user_prompt(hi_lines: list, en_lines: list) -> str:
    hi_block = "\n".join(f"{i+1}. {line}" for i, line in enumerate(hi_lines))
    en_block = "\n".join(f"{i+1}. {line}" for i, line in enumerate(en_lines))
    return (
        f"Hindi (Devanagari):\n{hi_block}\n\n"
        f"Romanization:\n{en_block}\n\n"
        f"Translate all {len(hi_lines)} lines to English, "
        f"preserving poetic style and rhyme scheme."
    )


def translate_poem(client, hi_lines: list, en_lines: list) -> list:
    """Translate one poem using Claude via Bedrock."""
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1024,
        "temperature": 0,
        "system": SYSTEM_PROMPT,
        "messages": [
            {"role": "user", "content": build_user_prompt(hi_lines, en_lines)}
        ]
    }

    response      = client.invoke_model(modelId=MODEL_ID, body=json.dumps(body))
    response_body = json.loads(response["body"].read().decode())
    raw           = response_body["content"][0]["text"].strip()

    # Strip markdown fences if model adds them despite instructions
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    translations = json.loads(raw.strip())

    if len(translations) != len(hi_lines):
        raise ValueError(
            f"Line count mismatch: input={len(hi_lines)}, output={len(translations)}"
        )
    return translations


def save_progress(data: dict, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def count_status(data: dict) -> tuple:
    done  = sum(1 for poet in data for poem in data[poet].values() if OUTPUT_KEY in poem)
    total = sum(len(data[poet]) for poet in data)
    return done, total


def process_poems(data: dict, output_path: str, test_mode: bool = False, poet_filter: str = None) -> dict:
    client    = build_bedrock_client()
    processed = 0
    skipped   = 0

    for poet in data:
        if poet_filter and poet != poet_filter:
            continue

        for poem_key, poem in data[poet].items():

            if "hi" not in poem or "en" not in poem:
                print(f"  Skipping [{poet}] {poem_key[:55]} — missing hi/en")
                skipped += 1
                continue

            if OUTPUT_KEY in poem:
                print(f"  Already done:  {poem_key[:55]}")
                skipped += 1
                continue

            print(f"  Translating [{poet}] {poem_key[:55]}...")

            try:
                translations     = translate_poem(client, poem["hi"], poem["en"])
                poem[OUTPUT_KEY] = translations
                processed        += 1
                print(f"    ✓ {len(translations)} lines translated")

                # Incremental save after every poem
                save_progress(data, output_path)
                print(f"    💾 Saved ({processed} translated, {skipped} skipped)")

                time.sleep(0.3)

            except json.JSONDecodeError as e:
                print(f"    ✗ JSON parse error: {e} — skipping")
            except ValueError as e:
                print(f"    ✗ {e} — skipping")
            except Exception as e:
                print(f"    ✗ Unexpected error: {e} — skipping")

            if test_mode and processed >= 2:
                print("\n[Test mode] Stopping after 2 poems.")
                return data

    print(f"\nDone. Translated: {processed} | Skipped: {skipped}")
    return data


def main():
    parser = argparse.ArgumentParser(description="Claude via Bedrock — poetic translation")
    parser.add_argument("--input",  default=INPUT_PATH)
    parser.add_argument("--output", default=None, help="Override output path")
    parser.add_argument("--test",   action="store_true", help="Translate 2 poems to test output dir")
    parser.add_argument("--all",    dest="all_poems", action="store_true")
    parser.add_argument("--poet",   default=None)
    args = parser.parse_args()

    if not args.test and not args.all_poems and not args.poet:
        parser.print_help()
        print("\nError: specify --test, --all, or --poet <poet-key>")
        return

    # Resolve output path
    if args.output:
        output_path = args.output
    elif args.test:
        output_path = TEST_OUTPUT
    else:
        output_path = FULL_OUTPUT

    # Resume: for --all, if output already exists load it instead of input
    if args.all_poems and os.path.exists(output_path):
        print(f"Resuming from: {output_path}")
        input_path = output_path
    else:
        input_path = args.input

    print(f"Input  : {input_path}")
    print(f"Output : {output_path}")
    print(f"Model  : {MODEL_ID}")
    print(f"Mode   : {'TEST (2 poems)' if args.test else 'ALL' if args.all_poems else f'poet={args.poet}'}\n")

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    done, total = count_status(data)
    print(f"Status : {done}/{total} already translated, {total - done} remaining\n")

    process_poems(data, output_path, test_mode=args.test, poet_filter=args.poet)


if __name__ == "__main__":
    main()