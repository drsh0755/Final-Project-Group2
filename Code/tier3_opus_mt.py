"""
=============================================================================
Tier 3 — Fine-tune Helsinki-NLP/opus-mt-hi-en (Transformer)
DATS 6312 Final Project: Style-Preserving Hindi/Urdu to English Translation
=============================================================================
Architecture:
    Base model : Helsinki-NLP/opus-mt-hi-en (pretrained MarianMT)
    Fine-tuning: Two-phase
        Phase 1 — IIT Bombay filtered CSV (general Hindi→English, 3 epochs)
        Phase 2 — Poetry CSV hi→en_anthropic (style-preserving, 10 epochs)

Run from: Code/ directory
    python3 tier3_opus_mt.py

Output:
    Models/tier3_opusmt/best_model/        ← HuggingFace model files
    Results/tier3/tier3_translations.csv
    Results/tier3/tier3_metrics.json
    Results/tier3/tier3_report.txt
    Plots/tier3_loss_curve.png
=============================================================================
"""

import torch
import pandas as pd
import numpy as np
import json
import math
import time
import csv
import sacrebleu
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from transformers import (
    MarianMTModel,
    MarianTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)
from torch.utils.data import Dataset

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent.parent
DATA_DIR    = BASE_DIR / "Data"  / "processed"
MODELS_DIR  = BASE_DIR / "Models" / "tier3_opusmt"
RESULTS_DIR = BASE_DIR / "Results" / "tier3"
PLOTS_DIR   = BASE_DIR / "Plots"
for d in [MODELS_DIR, RESULTS_DIR, PLOTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME      = "Helsinki-NLP/opus-mt-hi-en"
MAX_LEN         = 128

# Phase 1 — IIT Bombay general fine-tuning
P1_EPOCHS       = 3
P1_BATCH_SIZE   = 64
P1_LR           = 5e-5
P1_SAMPLE       = 50_000    # rows from iitb_train_filtered.csv

# Phase 2 — Poetry style fine-tuning
P2_EPOCHS       = 15
P2_BATCH_SIZE   = 32
P2_LR           = 2e-5

RANDOM_SEED     = 42
torch.manual_seed(RANDOM_SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU    : {torch.cuda.get_device_name(0)}")
    print(f"VRAM   : {round(torch.cuda.get_device_properties(0).total_memory/1e9,1)} GB")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — Dataset
# ─────────────────────────────────────────────────────────────────────────────

class TranslationDataset(Dataset):
    def __init__(self, src_texts, tgt_texts, tokenizer, max_len=MAX_LEN):
        self.src_texts  = src_texts
        self.tgt_texts  = tgt_texts
        self.tokenizer  = tokenizer
        self.max_len    = max_len

    def __len__(self):
        return len(self.src_texts)

    def __getitem__(self, idx):
        src = str(self.src_texts[idx]).strip()
        tgt = str(self.tgt_texts[idx]).strip()

        model_inputs = self.tokenizer(
            src,
            max_length=self.max_len,
            truncation=True,
            padding=False,
        )
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                tgt,
                max_length=self.max_len,
                truncation=True,
                padding=False,
            )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — Metrics for Trainer
# ─────────────────────────────────────────────────────────────────────────────

def build_compute_metrics(tokenizer):
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]

        # Decode predictions
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        decoded_preds  = tokenizer.batch_decode(preds,  skip_special_tokens=True)

        # Decode labels
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # sacrebleu expects list of references per hypothesis
        result = sacrebleu.corpus_bleu(
            decoded_preds,
            [[l] for l in decoded_labels]
        )
        return {"bleu": round(result.score, 2)}
    return compute_metrics

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — Training function
# ─────────────────────────────────────────────────────────────────────────────

def fine_tune(
    model, tokenizer, train_src, train_tgt,
    val_src, val_tgt,
    output_dir, phase_name,
    epochs, batch_size, lr,
    warmup_steps=100,
):
    print(f"\n{'='*60}")
    print(f"{phase_name}")
    print(f"{'='*60}")
    print(f"  Train samples : {len(train_src):,}")
    print(f"  Val samples   : {len(val_src):,}")
    print(f"  Epochs        : {epochs}")
    print(f"  Batch size    : {batch_size}")
    print(f"  LR            : {lr}")

    train_dataset = TranslationDataset(train_src, train_tgt, tokenizer)
    val_dataset   = TranslationDataset(val_src,   val_tgt,   tokenizer)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer, model=model, padding=True, pad_to_multiple_of=8
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir              = str(output_dir),
        num_train_epochs        = epochs,
        per_device_train_batch_size = batch_size,
        per_device_eval_batch_size  = batch_size * 2,
        learning_rate           = lr,
        warmup_steps            = warmup_steps,
        weight_decay            = 0.01,
        eval_strategy           = "epoch",
        save_strategy           = "epoch",
        load_best_model_at_end  = True,
        metric_for_best_model   = "bleu",
        greater_is_better       = True,
        predict_with_generate   = True,
        generation_max_length   = MAX_LEN,
        fp16                    = (DEVICE.type == "cuda"),
        logging_steps           = 50,
        logging_dir             = str(output_dir / "logs"),
        report_to               = "none",
        seed                    = RANDOM_SEED,
        dataloader_num_workers  = 2,
        save_total_limit        = 2,
    )

    trainer = Seq2SeqTrainer(
        model           = model,
        args            = training_args,
        train_dataset   = train_dataset,
        eval_dataset    = val_dataset,
        tokenizer       = tokenizer,
        data_collator   = data_collator,
        compute_metrics = build_compute_metrics(tokenizer),
        callbacks       = [EarlyStoppingCallback(early_stopping_patience=3)],
    )

    train_result = trainer.train()
    print(f"\n  Training complete.")
    print(f"  Train loss : {train_result.training_loss:.4f}")

    # Extract loss history for plotting
    logs = [l for l in trainer.state.log_history if "loss" in l]
    return trainer, logs

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — Inference
# ─────────────────────────────────────────────────────────────────────────────

def translate_line(model, tokenizer, text, device=DEVICE):
    """Translate a single Hindi line."""
    inputs = tokenizer(
        text, return_tensors="pt",
        max_length=MAX_LEN, truncation=True
    ).to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=MAX_LEN,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def translate_poem(model, tokenizer, poem_hi, device=DEVICE):
    """Translate full poem line by line."""
    lines = [l.strip() for l in poem_hi.split("\n") if l.strip()]
    translated = [translate_line(model, tokenizer, l, device) for l in lines]
    return "\n".join(translated)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — Evaluation helpers
# ─────────────────────────────────────────────────────────────────────────────

def rhyme_density(text):
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if not lines:
        return 0.0
    endings = [l.split()[-1][-3:] if l.split() else "" for l in lines]
    counts = defaultdict(int)
    for e in endings:
        counts[e] += 1
    rhyming = sum(1 for e in endings if counts[e] > 1)
    return round(rhyming / len(endings), 3)

def syllable_alignment_score(src_lines, tgt_lines):
    """
    Simple syllable alignment: compare vowel count per line
    as a proxy for syllabic similarity.
    """
    VOWELS_EN = set("aeiouAEIOU")
    def count_vowels(text):
        return sum(1 for c in text if c in VOWELS_EN)

    scores = []
    for s, t in zip(src_lines, tgt_lines):
        sv = count_vowels(s)
        tv = count_vowels(t)
        if sv == 0 and tv == 0:
            scores.append(1.0)
        elif sv == 0 or tv == 0:
            scores.append(0.0)
        else:
            scores.append(min(sv, tv) / max(sv, tv))
    return round(sum(scores) / len(scores), 3) if scores else 0.0

def evaluate_on_poems(model, tokenizer, max_poems=100):
    poems_df = pd.read_csv(DATA_DIR / "poetry_poems_test.csv")
    results  = []
    model.eval()

    for i, row in tqdm(poems_df.head(max_poems).iterrows(),
                       total=min(max_poems, len(poems_df)),
                       desc="  Evaluating poems"):
        hi        = str(row["hi"]).strip()
        ref_lit   = str(row["en_t"]).strip()
        ref_style = str(row["en_anthropic"]).strip()

        hyp = translate_poem(model, tokenizer, hi)

        # Line-level for syllable alignment
        hi_lines  = [l.strip() for l in hi.split("\n")  if l.strip()]
        hyp_lines = [l.strip() for l in hyp.split("\n") if l.strip()]

        # sacrebleu sentence BLEU
        bleu_lit   = sacrebleu.sentence_bleu(hyp, [ref_lit]).score   if ref_lit   else 0.0
        bleu_style = sacrebleu.sentence_bleu(hyp, [ref_style]).score if ref_style else 0.0

        results.append({
            "id":                    i,
            "poet":                  row["poet"],
            "poem_slug":             row["poem_slug"],
            "source":                hi,
            "ref_literal":           ref_lit,
            "ref_style":             ref_style,
            "hypothesis":            hyp,
            "bleu_vs_literal":       round(bleu_lit,   2),
            "bleu_vs_style":         round(bleu_style, 2),
            "src_rhyme_density":     rhyme_density(hi),
            "tgt_rhyme_density":     rhyme_density(hyp),
            "syllable_alignment":    syllable_alignment_score(hi_lines, hyp_lines),
        })
    return results

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — Loss curve plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_loss_curves(p1_logs, p2_logs):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    fig.suptitle("Tier 3 Opus-MT Fine-tuning Loss", fontsize=13)

    for ax, logs, title in zip(
        axes,
        [p1_logs, p2_logs],
        ["Phase 1 — IIT Bombay", "Phase 2 — Poetry Style"]
    ):
        steps  = [l["step"] for l in logs if "loss" in l]
        losses = [l["loss"] for l in logs if "loss" in l]
        if steps:
            ax.plot(steps, losses, color="steelblue")
            ax.set_title(title)
            ax.set_xlabel("Step")
            ax.set_ylabel("Loss")

    plt.tight_layout()
    path = PLOTS_DIR / "tier3_loss_curve.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved → {path}")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("TIER 3 — OPUS-MT FINE-TUNING (MarianMT)")
    print("=" * 60)

    # ── Load tokenizer & base model ───────────────────────────────────────────
    print(f"\nLoading base model: {MODEL_NAME}")
    tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)
    model     = MarianMTModel.from_pretrained(MODEL_NAME).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {total_params:,}")

    # ── PHASE 1 — IIT Bombay general fine-tuning ──────────────────────────────
    iitb_path = DATA_DIR / "iitb_train_filtered.csv"
    p1_logs   = []

    if iitb_path.exists():
        print(f"\nLoading IIT Bombay data: {iitb_path}")
        iitb_df = pd.read_csv(iitb_path).dropna(subset=["hindi", "english"])

        # Sample for phase 1
        if len(iitb_df) > P1_SAMPLE:
            iitb_df = iitb_df.sample(P1_SAMPLE, random_state=RANDOM_SEED)
        print(f"  Rows: {len(iitb_df):,}")

        # Simple 90/10 split for phase 1
        split = int(len(iitb_df) * 0.9)
        p1_train = iitb_df.iloc[:split]
        p1_val   = iitb_df.iloc[split:]

        p1_out = MODELS_DIR / "phase1"
        p1_out.mkdir(exist_ok=True)

        trainer1, p1_logs = fine_tune(
            model, tokenizer,
            train_src = p1_train["hindi"].tolist(),
            train_tgt = p1_train["english"].tolist(),
            val_src   = p1_val["hindi"].tolist(),
            val_tgt   = p1_val["english"].tolist(),
            output_dir  = p1_out,
            phase_name  = "PHASE 1 — IIT Bombay General Fine-tuning",
            epochs      = P1_EPOCHS,
            batch_size  = P1_BATCH_SIZE,
            lr          = P1_LR,
        )
        # Save phase 1 model
        model.save_pretrained(MODELS_DIR / "phase1_model")
        tokenizer.save_pretrained(MODELS_DIR / "phase1_model")
        print(f"  Phase 1 model saved → {MODELS_DIR / 'phase1_model'}")
    else:
        print(f"\n  IIT Bombay CSV not found at {iitb_path}")
        print("  Skipping Phase 1 — proceeding directly to Phase 2")

    # ── PHASE 2 — Poetry style fine-tuning ───────────────────────────────────
    print(f"\nLoading poetry data...")
    train_df = pd.read_csv(DATA_DIR / "poetry_train.csv").dropna(subset=["hi", "en_anthropic"])
    val_df   = pd.read_csv(DATA_DIR / "poetry_val.csv").dropna(subset=["hi", "en_anthropic"])
    train_df = train_df[train_df["en_anthropic"].str.strip() != ""]
    val_df   = val_df[val_df["en_anthropic"].str.strip() != ""]
    print(f"  Train: {len(train_df):,} rows")
    print(f"  Val  : {len(val_df):,} rows")

    p2_out = MODELS_DIR / "phase2"
    p2_out.mkdir(exist_ok=True)

    trainer2, p2_logs = fine_tune(
        model, tokenizer,
        train_src  = train_df["hi"].tolist(),
        train_tgt  = train_df["en_anthropic"].tolist(),
        val_src    = val_df["hi"].tolist(),
        val_tgt    = val_df["en_anthropic"].tolist(),
        output_dir = p2_out,
        phase_name = "PHASE 2 — Poetry Style Fine-tuning",
        epochs     = P2_EPOCHS,
        batch_size = P2_BATCH_SIZE,
        lr         = P2_LR,
    )

    # Save final best model
    best_model_dir = MODELS_DIR / "best_model"
    model.save_pretrained(best_model_dir)
    tokenizer.save_pretrained(best_model_dir)
    print(f"\n  Best model saved → {best_model_dir}")

    # ── Plot loss curves ──────────────────────────────────────────────────────
    print("\nPlotting loss curves...")
    plot_loss_curves(p1_logs, p2_logs)

    # ── Evaluate on full poems ────────────────────────────────────────────────
    print("\nEvaluating on full poems...")
    results = evaluate_on_poems(model, tokenizer, max_poems=100)

    # ── Print sample translations ─────────────────────────────────────────────
    print("\n── Sample Translations ──────────────────────────────────────")
    for r in results[:2]:
        print(f"\nSOURCE:\n{r['source'][:200]}")
        print(f"REF STYLE:\n{r['ref_style'][:200]}")
        print(f"HYPOTHESIS:\n{r['hypothesis'][:200]}")
        print(f"BLEU literal: {r['bleu_vs_literal']}  "
              f"BLEU style: {r['bleu_vs_style']}  "
              f"Syllable alignment: {r['syllable_alignment']}")

    # ── Save results ──────────────────────────────────────────────────────────
    print("\nSaving results...")
    csv_path = RESULTS_DIR / "tier3_translations.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"  Saved → {csv_path}")

    bleu_lit   = [r["bleu_vs_literal"]    for r in results]
    bleu_style = [r["bleu_vs_style"]      for r in results]
    src_rhyme  = [r["src_rhyme_density"]  for r in results]
    tgt_rhyme  = [r["tgt_rhyme_density"]  for r in results]
    syl_align  = [r["syllable_alignment"] for r in results]

    metrics = {
        "model":                     "Tier3_OpusMT_FineTuned",
        "base_model":                MODEL_NAME,
        "num_poems":                 len(results),
        "avg_bleu_vs_literal":       round(sum(bleu_lit)  /len(bleu_lit),   2),
        "avg_bleu_vs_style":         round(sum(bleu_style)/len(bleu_style), 2),
        "avg_src_rhyme_density":     round(sum(src_rhyme) /len(src_rhyme),  3),
        "avg_tgt_rhyme_density":     round(sum(tgt_rhyme) /len(tgt_rhyme),  3),
        "avg_syllable_alignment":    round(sum(syl_align) /len(syl_align),  3),
        "hyperparams": {
            "phase1_epochs":   P1_EPOCHS,   "phase1_lr": P1_LR,
            "phase2_epochs":   P2_EPOCHS,   "phase2_lr": P2_LR,
            "max_len":         MAX_LEN,     "p1_sample": P1_SAMPLE,
        }
    }

    json_path = RESULTS_DIR / "tier3_metrics.json"
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved → {json_path}")

    # Report
    report_path = RESULTS_DIR / "tier3_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("TIER 3 — OPUS-MT FINE-TUNED — REPORT\n")
        f.write("=" * 70 + "\n\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")
        f.write("\n" + "-"*70 + "\n\n")
        for r in results[:10]:
            f.write(f"[{r['id']}] {r['poet']}\n")
            f.write(f"SOURCE:\n{r['source']}\n\n")
            f.write(f"REF LITERAL:\n{r['ref_literal']}\n\n")
            f.write(f"REF STYLE:\n{r['ref_style']}\n\n")
            f.write(f"HYPOTHESIS:\n{r['hypothesis']}\n\n")
            f.write(f"BLEU literal: {r['bleu_vs_literal']}  "
                    f"BLEU style: {r['bleu_vs_style']}  "
                    f"Syllable alignment: {r['syllable_alignment']}\n")
            f.write("-"*70 + "\n\n")
    print(f"  Saved → {report_path}")

    print("\n── Final Metrics ────────────────────────────────────────────")
    for k, v in metrics.items():
        if k != "hyperparams":
            print(f"  {k:<30} {v}")

    print("\n✅ TIER 3 COMPLETE")
    print(f"   Model  → {best_model_dir}")
    print(f"   Results → {RESULTS_DIR}")
    print(f"   Plots   → {PLOTS_DIR}")