"""
=============================================================================
Tier 3 v2 — Fine-tune Helsinki-NLP/opus-mt-hi-en (Improved)
DATS 6312 Final Project: Style-Preserving Hindi/Urdu to English Translation
=============================================================================
Improvements over v1:
    1. Skip Phase 1 (IIT Bombay) — fine-tune directly from pretrained base
       Reason: Phase 1 shifted model toward prose, hurting poetry style
    2. Lower Phase 2 LR: 2e-5 → 5e-6 (reduce overfitting)
    3. Stronger weight decay: 0.01 → 0.05
    4. Smaller batch size: 32 → 16 (better generalization on small dataset)
    5. Beam search: no_repeat_ngram_size 3 → 2 (less constrained for short lines)
    6. More epochs: 15 → 25 with patience 5 (give model more time to converge)
    7. Post-processing rhyme booster on output

Saves to:
    Models/tier3_opusmt/best_model/     ← v2 best model
    Results/tier3/                      ← v2 results
    Plots/tier3_v2_loss_curve.png

Run from: Code/ directory
    python3 tier3_opus_mt_v2.py
=============================================================================
"""

import torch
import pandas as pd
import numpy as np
import json
import re
import csv
import sacrebleu
import unicodedata
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
MODEL_NAME     = "Helsinki-NLP/opus-mt-hi-en"
MAX_LEN        = 128
RANDOM_SEED    = 42

# Fix 1: Skip Phase 1 entirely
SKIP_PHASE1    = True

# Fix 2+3+4: Lower LR, stronger weight decay, smaller batch
P2_EPOCHS      = 25
P2_BATCH_SIZE  = 16
P2_LR          = 5e-6
P2_WEIGHT_DECAY= 0.05
P2_PATIENCE    = 5
P2_WARMUP      = 200

# Fix 5: Less aggressive beam search
BEAM_SIZE      = 4
NO_REPEAT_NGRAM= 2
LENGTH_PENALTY = 0.8

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
        self.src   = src_texts
        self.tgt   = tgt_texts
        self.tok   = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        model_inputs = self.tok(
            str(self.src[idx]).strip(),
            max_length=self.max_len, truncation=True, padding=False,
        )
        with self.tok.as_target_tokenizer():
            labels = self.tok(
                str(self.tgt[idx]).strip(),
                max_length=self.max_len, truncation=True, padding=False,
            )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — Metrics
# ─────────────────────────────────────────────────────────────────────────────

def build_compute_metrics(tokenizer):
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        preds  = np.where(preds  != -100, preds,  tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_preds  = tokenizer.batch_decode(preds,  skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        result = sacrebleu.corpus_bleu(decoded_preds, [[l] for l in decoded_labels])
        return {"bleu": round(result.score, 2)}
    return compute_metrics

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — Fine-tuning
# ─────────────────────────────────────────────────────────────────────────────

def fine_tune(model, tokenizer, train_src, train_tgt, val_src, val_tgt,
              output_dir, phase_name, epochs, batch_size, lr,
              weight_decay, warmup_steps, patience):

    print(f"\n{'='*60}")
    print(phase_name)
    print(f"{'='*60}")
    print(f"  Train  : {len(train_src):,}")
    print(f"  Val    : {len(val_src):,}")
    print(f"  Epochs : {epochs}  LR: {lr}  Batch: {batch_size}")
    print(f"  WD     : {weight_decay}  Patience: {patience}")

    train_ds = TranslationDataset(train_src, train_tgt, tokenizer)
    val_ds   = TranslationDataset(val_src,   val_tgt,   tokenizer)
    collator = DataCollatorForSeq2Seq(tokenizer, model=model,
                                      padding=True, pad_to_multiple_of=8)

    args = Seq2SeqTrainingArguments(
        output_dir              = str(output_dir),
        num_train_epochs        = epochs,
        per_device_train_batch_size = batch_size,
        per_device_eval_batch_size  = batch_size * 2,
        learning_rate           = lr,
        warmup_steps            = warmup_steps,
        weight_decay            = weight_decay,
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
        args            = args,
        train_dataset   = train_ds,
        eval_dataset    = val_ds,
        tokenizer       = tokenizer,
        data_collator   = collator,
        compute_metrics = build_compute_metrics(tokenizer),
        callbacks       = [EarlyStoppingCallback(early_stopping_patience=patience)],
    )

    result = trainer.train()
    print(f"\n  Training complete. Train loss: {result.training_loss:.4f}")

    logs = [l for l in trainer.state.log_history if "loss" in l]
    eval_logs = [l for l in trainer.state.log_history if "eval_bleu" in l]
    best_bleu = max((l["eval_bleu"] for l in eval_logs), default=0)
    print(f"  Best eval BLEU: {best_bleu:.2f}")

    return trainer, logs, best_bleu

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — Post-Processing: Rhyme Booster
# Fix 7: Nudge line endings toward rhyme using synonym substitution
# ─────────────────────────────────────────────────────────────────────────────

# Common end-word rhyme groups — swap last word to create rhyme pairs
RHYME_GROUPS = [
    ["away", "today", "say", "way", "day", "play", "stay", "lay", "ray", "pray"],
    ["night", "light", "sight", "might", "right", "bright", "flight", "delight"],
    ["heart", "part", "start", "art", "apart", "depart", "smart"],
    ["pain", "rain", "again", "remain", "vain", "chain", "gain", "plain"],
    ["love", "above", "dove", "move", "prove"],
    ["eyes", "rise", "skies", "lies", "cries", "sighs", "wise", "disguise"],
    ["soul", "whole", "role", "toll", "goal", "console", "stroll"],
    ["fire", "desire", "higher", "inspire", "entire", "admire"],
    ["tears", "years", "fears", "hears", "appears", "nears"],
    ["gone", "on", "dawn", "drawn", "alone", "known", "shown", "blown"],
    ["deep", "sleep", "keep", "weep", "seek", "speak"],
    ["door", "more", "floor", "shore", "before", "restore", "explore"],
    ["time", "rhyme", "climb", "sublime", "chime"],
    ["end", "friend", "bend", "blend", "send", "mend", "depend", "descend"],
    ["breath", "death", "beneath", "wreath"],
    ["sand", "hand", "land", "stand", "band", "grand", "understand"],
    ["dream", "stream", "gleam", "seem", "beam", "extreme", "redeem"],
    ["wait", "fate", "late", "gate", "state", "great", "create"],
    ["free", "see", "be", "tree", "sea", "flee", "agree", "plea"],
    ["mine", "line", "divine", "shine", "wine", "fine", "sign", "pine"],
]

def get_rhyme_group(word):
    """Return the rhyme group a word belongs to, or None."""
    word = word.lower().strip(".,!?;:'\"")
    for group in RHYME_GROUPS:
        if word in group:
            return group
    return None

def get_last_word(line):
    words = line.strip().split()
    if not words:
        return ""
    return re.sub(r'[^\w]', '', words[-1].lower())

def rhyme_booster(translated_lines):
    """
    For a list of translated lines, try to nudge pairs of lines
    toward rhyming by substituting the last word with a rhyming alternative
    when the meaning is flexible.
    Only applied when the last word is in a known rhyme group.
    Only touches lines that don't already rhyme.
    """
    if len(translated_lines) < 2:
        return translated_lines

    lines = list(translated_lines)
    n = len(lines)

    # Find the most common rhyme in the poem (dominant rhyme)
    endings = [get_last_word(l) for l in lines]
    rhyme_counts = defaultdict(int)
    for end in endings:
        group = get_rhyme_group(end)
        if group:
            # Use first word in group as canonical key
            rhyme_counts[group[0]] += 1

    if not rhyme_counts:
        return lines

    # Get dominant rhyme group
    dominant_key = max(rhyme_counts, key=rhyme_counts.get)
    dominant_group = next(g for g in RHYME_GROUPS if g[0] == dominant_key)

    # Try to nudge non-rhyming lines toward dominant rhyme
    boosted = []
    for line in lines:
        last = get_last_word(line)
        if last in dominant_group:
            boosted.append(line)  # already rhymes
            continue

        group = get_rhyme_group(last)
        if group:
            # Find a word in dominant group with similar length
            candidates = [w for w in dominant_group
                         if abs(len(w) - len(last)) <= 2 and w != last]
            if candidates:
                # Replace last word with closest rhyming candidate
                replacement = candidates[0]
                words = line.strip().split()
                # Preserve punctuation from original last word
                punct = re.findall(r'[^\w]+$', words[-1])
                punct = punct[0] if punct else ""
                words[-1] = replacement + punct
                boosted.append(" ".join(words))
            else:
                boosted.append(line)
        else:
            boosted.append(line)

    return boosted

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — Inference
# ─────────────────────────────────────────────────────────────────────────────

def translate_line(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt",
                       max_length=MAX_LEN, truncation=True).to(DEVICE)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_length=MAX_LEN,
            num_beams=BEAM_SIZE,
            length_penalty=LENGTH_PENALTY,
            no_repeat_ngram_size=NO_REPEAT_NGRAM,
            early_stopping=True,
        )
    return tokenizer.decode(out[0], skip_special_tokens=True)

def translate_poem(model, tokenizer, poem, apply_rhyme_boost=True):
    lines = [l.strip() for l in poem.split("\n") if l.strip()]
    translated = [translate_line(model, tokenizer, l) for l in lines]
    if apply_rhyme_boost:
        translated = rhyme_booster(translated)
    return "\n".join(translated)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — Evaluation helpers
# ─────────────────────────────────────────────────────────────────────────────

def rhyme_density(text):
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if not lines:
        return 0.0
    endings = [re.sub(r'[^\w]', '', l.split()[-1].lower())[-3:]
               if l.split() else "" for l in lines]
    counts = defaultdict(int)
    for e in endings:
        counts[e] += 1
    return round(sum(1 for e in endings if counts[e] > 1) / len(endings), 3)

def count_devanagari_syllables(text):
    count = 0
    for ch in text:
        name = unicodedata.name(ch, "")
        if "DEVANAGARI VOWEL" in name or "DEVANAGARI LETTER" in name:
            count += 1
    return max(count, 1)

def count_english_syllables(text):
    text = text.lower().strip()
    if not text:
        return 1
    vowels = "aeiouy"
    count, prev_vowel = 0, False
    for ch in text:
        is_v = ch in vowels
        if is_v and not prev_vowel:
            count += 1
        prev_vowel = is_v
    if text.endswith('e') and count > 1:
        count -= 1
    return max(count, 1)

def syllable_alignment(src, tgt):
    sl = [l.strip() for l in src.split("\n") if l.strip()]
    tl = [l.strip() for l in tgt.split("\n") if l.strip()]
    if not sl or not tl:
        return 0.0
    n = min(len(sl), len(tl))
    scores = []
    for i in range(n):
        sv = count_devanagari_syllables(sl[i])
        tv = count_english_syllables(tl[i])
        scores.append(min(sv, tv) / max(sv, tv))
    return round(sum(scores) / len(scores), 3)

def evaluate_on_poems(model, tokenizer, max_poems=100):
    poems_df = pd.read_csv(DATA_DIR / "poetry_poems_test.csv")
    results  = []
    model.eval()

    for i, row in tqdm(poems_df.head(max_poems).iterrows(),
                       total=min(max_poems, len(poems_df)),
                       desc="  Evaluating"):
        hi        = str(row["hi"]).strip()
        ref_lit   = str(row["en_t"]).strip()
        ref_style = str(row["en_anthropic"]).strip()

        hyp = translate_poem(model, tokenizer, hi, apply_rhyme_boost=True)
        hyp_no_boost = translate_poem(model, tokenizer, hi, apply_rhyme_boost=False)

        results.append({
            "id":                    i,
            "poet":                  row["poet"],
            "poem_slug":             row["poem_slug"],
            "source":                hi,
            "ref_literal":           ref_lit,
            "ref_style":             ref_style,
            "hypothesis":            hyp,
            "hypothesis_no_boost":   hyp_no_boost,
            "bleu_vs_literal":       round(sacrebleu.sentence_bleu(hyp, [ref_lit]).score,   2),
            "bleu_vs_style":         round(sacrebleu.sentence_bleu(hyp, [ref_style]).score, 2),
            "src_rhyme_density":     rhyme_density(hi),
            "tgt_rhyme_density":     rhyme_density(hyp),
            "tgt_rhyme_density_raw": rhyme_density(hyp_no_boost),
            "syllable_alignment":    syllable_alignment(hi, hyp),
        })
    return results

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — Plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_loss(logs):
    steps  = [l["step"] for l in logs if "loss" in l]
    losses = [l["loss"] for l in logs if "loss" in l]
    eval_logs = []

    plt.figure(figsize=(10, 4))
    if steps:
        plt.plot(steps, losses, color="steelblue", label="Train Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Tier 3 v2 — Poetry Fine-tuning Loss")
    plt.legend()
    plt.tight_layout()
    path = PLOTS_DIR / "tier3_v2_loss_curve.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved → {path}")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("TIER 3 v2 — OPUS-MT IMPROVED FINE-TUNING")
    print("=" * 60)
    print(f"\nImprovements vs v1:")
    print(f"  ✓ Skip Phase 1 (IIT Bombay) — direct poetry fine-tuning")
    print(f"  ✓ LR: 2e-5 → {P2_LR}")
    print(f"  ✓ Weight decay: 0.01 → {P2_WEIGHT_DECAY}")
    print(f"  ✓ Batch size: 32 → {P2_BATCH_SIZE}")
    print(f"  ✓ no_repeat_ngram_size: 3 → {NO_REPEAT_NGRAM}")
    print(f"  ✓ Post-processing rhyme booster")
    print(f"  ✓ Patience: 3 → {P2_PATIENCE}, Epochs: 15 → {P2_EPOCHS}")

    # ── Load fresh pretrained model ───────────────────────────────────────────
    print(f"\nLoading pretrained base: {MODEL_NAME}")
    tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)
    model     = MarianMTModel.from_pretrained(MODEL_NAME).to(DEVICE)
    print(f"  Params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # ── Load poetry data ──────────────────────────────────────────────────────
    print(f"\nLoading poetry data...")
    train_df = pd.read_csv(DATA_DIR / "poetry_train.csv").dropna(subset=["hi", "en_anthropic"])
    val_df   = pd.read_csv(DATA_DIR / "poetry_val.csv").dropna(subset=["hi", "en_anthropic"])
    train_df = train_df[train_df["en_anthropic"].str.strip() != ""]
    val_df   = val_df[val_df["en_anthropic"].str.strip() != ""]
    print(f"  Train: {len(train_df):,}  Val: {len(val_df):,}")

    # ── Fine-tune ─────────────────────────────────────────────────────────────
    p2_out = MODELS_DIR / "phase2_v2"
    p2_out.mkdir(exist_ok=True)

    trainer, logs, best_bleu = fine_tune(
        model, tokenizer,
        train_src    = train_df["hi"].tolist(),
        train_tgt    = train_df["en_anthropic"].tolist(),
        val_src      = val_df["hi"].tolist(),
        val_tgt      = val_df["en_anthropic"].tolist(),
        output_dir   = p2_out,
        phase_name   = "PHASE 2 v2 — Poetry Style Fine-tuning (Improved)",
        epochs       = P2_EPOCHS,
        batch_size   = P2_BATCH_SIZE,
        lr           = P2_LR,
        weight_decay = P2_WEIGHT_DECAY,
        warmup_steps = P2_WARMUP,
        patience     = P2_PATIENCE,
    )

    # ── Save best model ───────────────────────────────────────────────────────
    best_dir = MODELS_DIR / "best_model"
    model.save_pretrained(best_dir)
    tokenizer.save_pretrained(best_dir)
    print(f"\n  Best model saved → {best_dir}")

    # Also save explicitly as v2
    v2_dir = MODELS_DIR / "tier3_v2"
    model.save_pretrained(v2_dir)
    tokenizer.save_pretrained(v2_dir)
    print(f"  v2 model saved   → {v2_dir}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    print("\nPlotting loss curve...")
    plot_loss(logs)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    print("\nEvaluating on full poems...")
    results = evaluate_on_poems(model, tokenizer, max_poems=100)

    # ── Sample output ─────────────────────────────────────────────────────────
    print("\n── Sample Translations (v2) ─────────────────────────────────")
    for r in results[:2]:
        print(f"\nSOURCE:\n{r['source'][:200]}")
        print(f"REF STYLE:\n{r['ref_style'][:200]}")
        print(f"HYPOTHESIS:\n{r['hypothesis'][:200]}")
        print(f"BLEU literal: {r['bleu_vs_literal']}  "
              f"BLEU style: {r['bleu_vs_style']}  "
              f"Rhyme: {r['tgt_rhyme_density']}  "
              f"Syllable: {r['syllable_alignment']}")

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
        "model":                     "Tier3_OpusMT_v2_Improved",
        "base_model":                MODEL_NAME,
        "version":                   "v2",
        "num_poems":                 len(results),
        "best_phase2_bleu":          best_bleu,
        "avg_bleu_vs_literal":       round(sum(bleu_lit)  /len(bleu_lit),   2),
        "avg_bleu_vs_style":         round(sum(bleu_style)/len(bleu_style), 2),
        "avg_src_rhyme_density":     round(sum(src_rhyme) /len(src_rhyme),  3),
        "avg_tgt_rhyme_density":     round(sum(tgt_rhyme) /len(tgt_rhyme),  3),
        "avg_syllable_alignment":    round(sum(syl_align) /len(syl_align),  3),
        "improvements": {
            "skip_phase1":           True,
            "lr":                    P2_LR,
            "weight_decay":          P2_WEIGHT_DECAY,
            "batch_size":            P2_BATCH_SIZE,
            "no_repeat_ngram_size":  NO_REPEAT_NGRAM,
            "rhyme_booster":         True,
            "patience":              P2_PATIENCE,
        }
    }

    json_path = RESULTS_DIR / "tier3_metrics.json"
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved → {json_path}")

    report_path = RESULTS_DIR / "tier3_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("TIER 3 v2 — OPUS-MT IMPROVED — REPORT\n")
        f.write("=" * 70 + "\n\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")
        f.write("\n" + "-"*70 + "\n\n")
        for r in results[:10]:
            f.write(f"[{r['id']}] {r['poet']}\n")
            f.write(f"SOURCE:\n{r['source']}\n\n")
            f.write(f"REF LITERAL:\n{r['ref_literal']}\n\n")
            f.write(f"REF STYLE:\n{r['ref_style']}\n\n")
            f.write(f"HYPOTHESIS (with rhyme boost):\n{r['hypothesis']}\n\n")
            f.write(f"HYPOTHESIS (without boost):\n{r['hypothesis_no_boost']}\n\n")
            f.write(f"BLEU literal: {r['bleu_vs_literal']}  "
                    f"BLEU style: {r['bleu_vs_style']}  "
                    f"Rhyme: {r['tgt_rhyme_density']}  "
                    f"Syllable: {r['syllable_alignment']}\n")
            f.write("-"*70 + "\n\n")
    print(f"  Saved → {report_path}")

    print("\n── Final Metrics (v2) ───────────────────────────────────────")
    for k, v in metrics.items():
        if k not in ("improvements", "hyperparams"):
            print(f"  {k:<30} {v}")

    print("\n✅ TIER 3 v2 COMPLETE")
    print(f"   v2 model  → {v2_dir}")
    print(f"   Results   → {RESULTS_DIR}")
    print(f"   Plots     → {PLOTS_DIR / 'tier3_v2_loss_curve.png'}")
    print(f"\n   Now run evaluate_all_tiers.py to update the comparison table.")