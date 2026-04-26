"""
=============================================================================
IIT Bombay Hindi-English Corpus — Data Preparation
DATS 6312 Final Project: Style-Preserving Hindi/Urdu to English Translation
=============================================================================
PURPOSE:
    Download, clean, filter, and save the IIT Bombay parallel corpus.
    This becomes Stage 1 fine-tuning data for Opus-MT (general Hindi->English),
    before Stage 2 fine-tuning on the Kaggle Urdu ghazal poetry dataset.

OUTPUT FILES:
    RAW (downloaded as-is, no changes):
        .../Data/raw/iitb_train_raw.csv
        .../Data/raw/iitb_val_raw.csv
        .../Data/raw/iitb_test_raw.csv

    PROCESSED (cleaned, filtered, sampled -- ready for model training):
        .../Data/processed/iitb_train_filtered.csv   -- 100K sampled & cleaned pairs
        .../Data/processed/iitb_val_filtered.csv     -- full validation split (cleaned)
        .../Data/processed/iitb_test_filtered.csv    -- full test split (cleaned)
        .../Data/processed/iitb_eda_report.txt       -- summary statistics
        .../Data/processed/iitb_length_distribution.png -- token length histograms
=============================================================================
"""

import os
import re
import logging
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from pathlib import Path
from datasets import load_dataset

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# ── Directory paths ───────────────────────────────────────────────────────────
PROJECT_BASE  = Path("/home/ubuntu/Natural Language Processing - DATS6312/Final-Project-Group2")
RAW_DIR       = PROJECT_BASE / "Data" / "raw"
PROCESSED_DIR = PROJECT_BASE / "Data" / "processed"

# ── Config ────────────────────────────────────────────────────────────────────
TRAIN_SAMPLE = 100_000    # rows to keep from train (full = ~1.49M)
MIN_TOKENS   = 3          # minimum words per sentence
MAX_TOKENS   = 80         # maximum words per sentence
RANDOM_SEED  = 42


# =============================================================================
# STEP 1: DOWNLOAD RAW DATA FROM HUGGINGFACE
# =============================================================================

def download_split(split: str) -> pd.DataFrame:
    """Download one split from HuggingFace and return as DataFrame."""
    logger.info(f"Downloading '{split}' split from cfilt/iitb-english-hindi ...")
    ds = load_dataset("cfilt/iitb-english-hindi", split=split)
    df = pd.DataFrame({
        "hindi":   [row["translation"]["hi"] for row in ds],
        "english": [row["translation"]["en"] for row in ds],
    })
    logger.info(f"  -> {len(df):,} rows downloaded")
    return df


def save_raw(df: pd.DataFrame, split: str) -> None:
    """Save raw (unmodified) download to the raw directory."""
    path = RAW_DIR / f"iitb_{split}_raw.csv"
    df.to_csv(path, index=False, encoding="utf-8")
    logger.info(f"  Raw saved  -> {path}  ({len(df):,} rows)")


# =============================================================================
# STEP 2: CLEAN & FILTER
# =============================================================================

def clean_text(text: str) -> str:
    """Basic cleaning: strip whitespace, collapse multiple spaces."""
    if not isinstance(text, str):
        return ""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def filter_df(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Apply cleaning + quality filters to a parallel corpus DataFrame."""
    original_len = len(df)

    # 1. Clean text
    df = df.copy()
    df["hindi"]   = df["hindi"].apply(clean_text)
    df["english"] = df["english"].apply(clean_text)

    # 2. Drop nulls / empty strings
    df = df[df["hindi"].str.len() > 0]
    df = df[df["english"].str.len() > 0]
    after_nulls = len(df)

    # 3. Token length filter
    df["hi_len"] = df["hindi"].str.split().str.len()
    df["en_len"] = df["english"].str.split().str.len()
    df = df[
        df["hi_len"].between(MIN_TOKENS, MAX_TOKENS) &
        df["en_len"].between(MIN_TOKENS, MAX_TOKENS)
    ]
    after_length = len(df)

    # 4. Drop exact duplicates on either side
    df = df.drop_duplicates(subset=["hindi"])
    df = df.drop_duplicates(subset=["english"])
    after_dedup = len(df)

    logger.info(
        f"  [{label}] Filter summary:\n"
        f"    Original:            {original_len:>10,}\n"
        f"    After null removal:  {after_nulls:>10,}\n"
        f"    After length filter: {after_length:>10,}\n"
        f"    After dedup:         {after_dedup:>10,}"
    )
    return df


# =============================================================================
# STEP 3: EDA -- stats + plot (saved to processed dir)
# =============================================================================

def generate_eda(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> None:
    """Print and save EDA statistics + length distribution plot."""

    report_lines = []

    def section(title, df, name):
        lines = [
            f"\n{'='*60}",
            f"  {title}",
            f"{'='*60}",
            f"  Rows:                {len(df):,}",
            f"  Hindi avg tokens:    {df['hi_len'].mean():.1f}  (std {df['hi_len'].std():.1f})",
            f"  English avg tokens:  {df['en_len'].mean():.1f}  (std {df['en_len'].std():.1f})",
            f"  Hindi avg chars:     {df['hindi'].str.len().mean():.1f}",
            f"  English avg chars:   {df['english'].str.len().mean():.1f}",
            f"\n  Sample rows ({name}):"
        ]
        for _, row in df.sample(3, random_state=RANDOM_SEED).iterrows():
            lines.append(f"    HI: {row['hindi'][:80]}")
            lines.append(f"    EN: {row['english'][:80]}")
            lines.append("")
        return lines

    report_lines += section("TRAIN SPLIT (100K sample)", train, "train")
    report_lines += section("VALIDATION SPLIT", val, "val")
    report_lines += section("TEST SPLIT", test, "test")

    report_text = "\n".join(report_lines)
    print(report_text)

    report_path = PROCESSED_DIR / "iitb_eda_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    logger.info(f"  EDA report saved   -> {report_path}")

    # ── Length distribution plot ──────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("IIT Bombay Corpus -- Token Length Distribution (Train, 100K sample)", fontsize=13)

    axes[0].hist(train["hi_len"], bins=40, color="#4C72B0", edgecolor="white", alpha=0.85)
    axes[0].set_title("Hindi sentence lengths")
    axes[0].set_xlabel("Token count")
    axes[0].set_ylabel("Frequency")
    axes[0].axvline(train["hi_len"].mean(), color="red", linestyle="--",
                    label=f"Mean: {train['hi_len'].mean():.1f}")
    axes[0].legend()

    axes[1].hist(train["en_len"], bins=40, color="#55A868", edgecolor="white", alpha=0.85)
    axes[1].set_title("English sentence lengths")
    axes[1].set_xlabel("Token count")
    axes[1].set_ylabel("Frequency")
    axes[1].axvline(train["en_len"].mean(), color="red", linestyle="--",
                    label=f"Mean: {train['en_len'].mean():.1f}")
    axes[1].legend()

    plt.tight_layout()
    plot_path = PROCESSED_DIR / "iitb_length_distribution.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Length plot saved  -> {plot_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    # ── Create directories ─────────────────────────────────────────────────────
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("  IIT Bombay Data Preparation -- Starting")
    logger.info(f"  Raw dir:        {RAW_DIR}")
    logger.info(f"  Processed dir:  {PROCESSED_DIR}")
    logger.info("=" * 60)

    # ── Step 1: Download ───────────────────────────────────────────────────────
    logger.info("\nStep 1: Downloading from HuggingFace ...")
    train_raw = download_split("train")
    val_raw   = download_split("validation")
    test_raw  = download_split("test")

    # ── Step 2: Save raw ───────────────────────────────────────────────────────
    logger.info("\nStep 2: Saving raw files ...")
    save_raw(train_raw, "train")
    save_raw(val_raw,   "val")
    save_raw(test_raw,  "test")

    # ── Step 3: Clean & Filter ─────────────────────────────────────────────────
    logger.info("\nStep 3: Cleaning and filtering ...")
    train_clean = filter_df(train_raw, "TRAIN")
    val_clean   = filter_df(val_raw,   "VAL")
    test_clean  = filter_df(test_raw,  "TEST")

    # ── Sample train (100K is sufficient for fine-tuning Opus-MT) ──────────────
    if len(train_clean) > TRAIN_SAMPLE:
        logger.info(f"\nSampling {TRAIN_SAMPLE:,} rows from train (was {len(train_clean):,}) ...")
        train_clean = train_clean.sample(TRAIN_SAMPLE, random_state=RANDOM_SEED).reset_index(drop=True)

    # ── Step 4: EDA ────────────────────────────────────────────────────────────
    logger.info("\nStep 4: Generating EDA ...")
    generate_eda(train_clean, val_clean, test_clean)

    # ── Step 5: Save processed ─────────────────────────────────────────────────
    logger.info("\nStep 5: Saving processed files ...")
    cols = ["hindi", "english"]

    train_path = PROCESSED_DIR / "iitb_train_filtered.csv"
    val_path   = PROCESSED_DIR / "iitb_val_filtered.csv"
    test_path  = PROCESSED_DIR / "iitb_test_filtered.csv"

    train_clean[cols].to_csv(train_path, index=False, encoding="utf-8")
    val_clean[cols].to_csv(val_path,     index=False, encoding="utf-8")
    test_clean[cols].to_csv(test_path,   index=False, encoding="utf-8")

    logger.info(f"  Processed -> {train_path}  ({len(train_clean):,} rows)")
    logger.info(f"  Processed -> {val_path}    ({len(val_clean):,} rows)")
    logger.info(f"  Processed -> {test_path}   ({len(test_clean):,} rows)")

    logger.info("\n" + "=" * 60)
    logger.info("  Done! Summary:")
    logger.info(f"    Raw files       -> {RAW_DIR}")
    logger.info(f"    Processed files -> {PROCESSED_DIR}")
    logger.info("  Next step: run Stage 1 fine-tuning on iitb_train_filtered.csv")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()