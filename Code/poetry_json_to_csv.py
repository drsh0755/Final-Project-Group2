"""
=============================================================================
Poetry JSON → Train/Val/Test CSV Converter
DATS 6312 Final Project: Style-Preserving Hindi/Urdu to English Translation
=============================================================================
Input:  ../Data/processed/poetry_data_translated.json
Output: ../Data/processed/poetry_train.csv, poetry_val.csv, poetry_test.csv,
        ../Data/processed/poetry_poems_test.csv
        ../Data/processed/poetry_conversion_report.txt

Run from: Code/ directory

Columns: en | hi | en_t | en_anthropic | en_mistral | en_gemini | en_grok | poet | poem_slug

Notes:
    - en          : romanized transliteration (kept for reference, not for training)
    - hi          : Devanagari source — model INPUT
    - en_t        : deep-translator literal — baseline reference at evaluation
    - en_anthropic: Claude style-preserving — PRIMARY training target
    - en_mistral  : Mistral style-preserving — filled in when available
    - en_gemini   : Gemini style-preserving — filled in when available
    - en_grok     : Grok style-preserving   — filled in when available
    - poet        : poet slug for stratified splitting
    - poem_slug   : poem identifier

Split: 70% train / 15% val / 15% test (stratified by poet)
Line-level rows for training; full-poem rows in poetry_poems_test.csv for evaluation
=============================================================================
"""

import json
import csv
import random
from pathlib import Path
from collections import defaultdict

# ── Paths (run from Code/ directory) ─────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent   # Final-Project-Group2/
DATA_DIR   = BASE_DIR / "Data" / "processed"
INPUT_JSON = DATA_DIR / "poetry_data_translated.json"
OUT_DIR    = DATA_DIR
RANDOM_SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15

COLUMNS = ["en", "hi", "en_t", "en_anthropic", "en_mistral", "en_gemini", "en_grok", "poet", "poem_slug"]

random.seed(RANDOM_SEED)

# ── Load ──────────────────────────────────────────────────────────────────────
print("=" * 60)
print("POETRY JSON → CSV CONVERTER")
print("=" * 60)

with open(INPUT_JSON, encoding="utf-8") as f:
    data = json.load(f)

poets = list(data.keys())
print(f"\nPoets found : {len(poets)}")
print(f"First 5     : {poets[:5]}")

# ── Parse into line-level rows grouped by poet ────────────────────────────────
poet_rows  = defaultdict(list)   # poet -> list of line-level rows
poet_poems = defaultdict(list)   # poet -> list of full-poem rows
stats      = defaultdict(int)

def pad(lst, length):
    return lst + [""] * (length - len(lst))

for poet, poems in data.items():
    for slug, content in poems.items():
        hi_lines = content.get("hi",           [])
        if not hi_lines:
            stats["skipped_no_hi"] += 1
            continue

        n = len(hi_lines)
        en_lines = pad(content.get("en",           []), n)
        en_t     = pad(content.get("en_t",         []), n)
        en_anth  = pad(content.get("en_anthropic", []), n)
        en_mist  = pad(content.get("en_mistral",   []), n)
        en_gem   = pad(content.get("en_gemini",    []), n)
        en_grok  = pad(content.get("en_grok",      []), n)

        # Line-level rows
        for i in range(n):
            if not hi_lines[i].strip():
                continue
            poet_rows[poet].append({
                "en":           en_lines[i].strip(),
                "hi":           hi_lines[i].strip(),
                "en_t":         en_t[i].strip(),
                "en_anthropic": en_anth[i].strip(),
                "en_mistral":   en_mist[i].strip(),
                "en_gemini":    en_gem[i].strip(),
                "en_grok":      en_grok[i].strip(),
                "poet":         poet,
                "poem_slug":    slug,
            })
            stats["total_lines"] += 1

        # Full-poem row (lines joined with \n for evaluation)
        def join(lst):
            return "\n".join(l.strip() for l in lst if l.strip())

        poet_poems[poet].append({
            "en":           join(en_lines),
            "hi":           join(hi_lines),
            "en_t":         join(en_t),
            "en_anthropic": join(en_anth),
            "en_mistral":   join(en_mist),
            "en_gemini":    join(en_gem),
            "en_grok":      join(en_grok),
            "poet":         poet,
            "poem_slug":    slug,
        })
        stats["total_poems"] += 1

        # Coverage tracking
        if any(l.strip() for l in en_anth):  stats["has_en_anthropic"] += 1
        if any(l.strip() for l in en_t):     stats["has_en_t"]         += 1
        if any(l.strip() for l in en_mist):  stats["has_en_mistral"]   += 1
        if any(l.strip() for l in en_gem):   stats["has_en_gemini"]    += 1
        if any(l.strip() for l in en_grok):  stats["has_en_grok"]      += 1

print(f"\nParsed:")
print(f"  Total poems  : {stats['total_poems']}")
print(f"  Total lines  : {stats['total_lines']}")
print(f"\nTranslation coverage (poems with at least 1 translated line):")
print(f"  en_anthropic : {stats['has_en_anthropic']}")
print(f"  en_t         : {stats['has_en_t']}")
print(f"  en_mistral   : {stats['has_en_mistral']}")
print(f"  en_gemini    : {stats['has_en_gemini']}")
print(f"  en_grok      : {stats['has_en_grok']}")

# ── Stratified split by poet ──────────────────────────────────────────────────
print("\nSplitting stratified by poet...")

train_rows, val_rows, test_rows   = [], [], []
train_poems, val_poems, test_poems = [], [], []

for poet in poets:
    rows  = poet_rows[poet]
    poems = poet_poems[poet]

    poem_slugs = list({r["poem_slug"] for r in rows})
    random.shuffle(poem_slugs)

    n       = len(poem_slugs)
    n_train = max(1, int(n * TRAIN_RATIO))
    n_val   = max(1, int(n * VAL_RATIO))

    train_slugs = set(poem_slugs[:n_train])
    val_slugs   = set(poem_slugs[n_train:n_train + n_val])
    test_slugs  = set(poem_slugs[n_train + n_val:])

    # Guarantee at least 1 poem in test if enough poems exist
    if not test_slugs and n >= 3:
        moved = poem_slugs[-1]
        test_slugs  = {moved}
        train_slugs.discard(moved)
        val_slugs.discard(moved)

    for r in rows:
        s = r["poem_slug"]
        if s in train_slugs:   train_rows.append(r)
        elif s in val_slugs:   val_rows.append(r)
        else:                  test_rows.append(r)

    for p in poems:
        s = p["poem_slug"]
        if s in train_slugs:   train_poems.append(p)
        elif s in val_slugs:   val_poems.append(p)
        else:                  test_poems.append(p)

random.shuffle(train_rows)
random.shuffle(val_rows)
random.shuffle(test_rows)

print(f"\n  Train lines  : {len(train_rows)}")
print(f"  Val lines    : {len(val_rows)}")
print(f"  Test lines   : {len(test_rows)}")
print(f"  Test poems   : {len(test_poems)}  ← used for Tier 1/2/3 evaluation")

# ── Save CSVs ─────────────────────────────────────────────────────────────────
def save_csv(rows, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved {len(rows):>5} rows → {path}")

print("\nSaving...")
save_csv(train_rows,  OUT_DIR / "poetry_train.csv")
save_csv(val_rows,    OUT_DIR / "poetry_val.csv")
save_csv(test_rows,   OUT_DIR / "poetry_test.csv")
save_csv(test_poems,  OUT_DIR / "poetry_poems_test.csv")

# ── Conversion report ─────────────────────────────────────────────────────────
report_path = OUT_DIR / "poetry_conversion_report.txt"
with open(report_path, "w", encoding="utf-8") as f:
    f.write("=" * 60 + "\n")
    f.write("POETRY JSON → CSV CONVERSION REPORT\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Input           : {INPUT_JSON}\n")
    f.write(f"Poets           : {len(poets)}\n")
    f.write(f"Total poems     : {stats['total_poems']}\n")
    f.write(f"Total lines     : {stats['total_lines']}\n\n")
    f.write("Translation coverage:\n")
    f.write(f"  en_anthropic  : {stats['has_en_anthropic']} poems\n")
    f.write(f"  en_t          : {stats['has_en_t']} poems\n")
    f.write(f"  en_mistral    : {stats['has_en_mistral']} poems\n")
    f.write(f"  en_gemini     : {stats['has_en_gemini']} poems\n")
    f.write(f"  en_grok       : {stats['has_en_grok']} poems\n\n")
    f.write("Splits:\n")
    f.write(f"  Train lines   : {len(train_rows)}\n")
    f.write(f"  Val lines     : {len(val_rows)}\n")
    f.write(f"  Test lines    : {len(test_rows)}\n")
    f.write(f"  Test poems    : {len(test_poems)}\n\n")
    f.write("Training note:\n")
    f.write("  hi → en_anthropic  : primary training pair\n")
    f.write("  en_t               : literal baseline for evaluation\n")
    f.write("  en_mistral/gemini/grok: add when available\n")

print(f"  Saved report  → {report_path}")

# ── Sample output ─────────────────────────────────────────────────────────────
print("\n── Sample row ───────────────────────────────────────────────")
if train_rows:
    for k, v in train_rows[0].items():
        print(f"  {k:<15}: {str(v)[:80]}")

print("\n✅ CONVERSION COMPLETE")
print(f"   Run on GCP from the processed/ directory:")
print(f"   python poetry_json_to_csv.py")