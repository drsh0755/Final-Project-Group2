"""
Translate Hindi poem lines to English using Google Translate (baseline benchmark).
Adds results as "en_t" key in the JSON structure.

Install:
    pip install deep-translator

Usage:
    python translate_poems_google.py --test                        # 2 poems only (default paths)
    python translate_poems_google.py --all                         # all poems (default paths)
    python translate_poems_google.py --poet ahmad-faraz --test     # one poet, 2 poems
    python translate_poems_google.py --input /custom/path.json --output /custom/out.json --all
"""

import json
import time
import argparse
import os
from deep_translator import GoogleTranslator

# ── Default paths ────────────────────────────────────────────────────────────
BASE_DIR   = "/home/ubuntu/Natural Language Processing - DATS6312/Final-Project-Group2"
INPUT_PATH = os.path.join(
    BASE_DIR,
    "Data/raw/dataset - urdu-ghazal-dataset-32-poets-and-their-ghazals/poetry_complete.json"
)
OUTPUT_DIR = os.path.join(BASE_DIR, "Results/outputs")
# ─────────────────────────────────────────────────────────────────────────────


def translate_lines(hi_lines: list[str]) -> list[str]:
    """Translate a list of Hindi lines to English using Google Translate."""
    translator = GoogleTranslator(source="hi", target="en")
    translated = []
    for line in hi_lines:
        line = line.strip()
        if not line:
            translated.append("")
            continue
        try:
            result = translator.translate(line)
            translated.append(result)
        except Exception as e:
            print(f"      Warning: could not translate line '{line[:40]}': {e}")
            translated.append("")
        time.sleep(0.1)  # avoid hammering Google
    return translated


def process_poems(data: dict, output_path: str, test_mode: bool = False, poet_filter: str = None) -> dict:

    """Add 'en_t' key to each poem by translating its 'hi' lines."""
    processed = 0

    for poet in data:
        if poet_filter and poet != poet_filter:
            continue

        for poem_key, poem in data[poet].items():
            if "hi" not in poem:
                print(f"  Skipping [{poet}] {poem_key[:55]} — no 'hi' key")
                continue

            if "en_t" in poem:
                print(f"  Already translated: {poem_key[:55]}")
                continue

            print(f"  Translating [{poet}] {poem_key[:55]}...")
            translations = translate_lines(poem["hi"])
            poem["en_t"] = translations
            processed += 1
            print(f"    ✓ {len(translations)} lines translated")

            # Save progress after every poem
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"    💾 Saved progress ({processed} poems so far)")

            if test_mode and processed >= 2:
                print("\n[Test mode] Stopping after 2 poems.")
                return data

    print(f"\nDone. Translated {processed} poem(s).")
    return data


def main():
    parser = argparse.ArgumentParser(description="Google Translate baseline for Hindi poetry")
    parser.add_argument("--input",  default=INPUT_PATH, help="Input JSON file")
    parser.add_argument("--output", default=None,       help="Output JSON file path (default: OUTPUT_DIR/poetry_data_translated.json)")
    parser.add_argument("--test",   action="store_true", help="Translate only 2 poems")
    parser.add_argument("--all",    dest="all_poems", action="store_true", help="Translate all poems")
    parser.add_argument("--poet",   default=None,       help="Translate only poems by this poet key")
    args = parser.parse_args()

    if not args.test and not args.all_poems and not args.poet:
        parser.print_help()
        print("\nError: specify --test, --all, or --poet <poet-key>")
        return

    # Resolve output path
    if args.output:
        output_path = args.output
    else:
        suffix = "_test" if args.test else "_translated"
        if args.poet:
            suffix = f"_{args.poet}{suffix}"
        output_path = os.path.join(OUTPUT_DIR, f"poetry_data{suffix}.json")

    # Make sure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"Input  : {args.input}")
    print(f"Output : {output_path}")
    print(f"Mode   : {'TEST (2 poems)' if args.test else 'ALL' if args.all_poems else f'poet={args.poet}'}\n")

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    data = process_poems(data, output_path=output_path, test_mode=args.test, poet_filter=args.poet)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()