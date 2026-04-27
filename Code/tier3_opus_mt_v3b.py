"""
=============================================================================
Tier 3 v3b — Fine-tune Opus-MT: Option B (Curriculum Learning)
DATS 6312 Final Project: Style-Preserving Hindi/Urdu to English Translation
=============================================================================
Strategy:
    Phase A: Fine-tune on hi→en_anthropic for 15 epochs (primary style target)
    Phase B: Continue fine-tuning on hi→en_mistral for 10 epochs (secondary)
    Curriculum: learn one style first, then adapt to variation

Saves to:
    Models/tier3_opusmt/tier3_v3b/
    Results/tier3_v3b/
    Plots/tier3_v3b_loss_curve.png

Run from: Code/ directory
    python3 tier3_opus_mt_v3b.py
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
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from transformers import (
    MarianMTModel, MarianTokenizer,
    Seq2SeqTrainer, Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq, EarlyStoppingCallback,
)
from torch.utils.data import Dataset

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent.parent
DATA_DIR    = BASE_DIR / "Data"  / "processed"
MODELS_DIR  = BASE_DIR / "Models" / "tier3_opusmt"
RESULTS_DIR = BASE_DIR / "Results" / "tier3_v3b"
PLOTS_DIR   = BASE_DIR / "Plots"
for d in [MODELS_DIR, RESULTS_DIR, PLOTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME      = "Helsinki-NLP/opus-mt-hi-en"
MAX_LEN         = 128
RANDOM_SEED     = 42

# Phase A — learn primary style (en_anthropic)
PA_EPOCHS       = 15
PA_BATCH_SIZE   = 16
PA_LR           = 5e-6
PA_WEIGHT_DECAY = 0.05
PA_PATIENCE     = 5
PA_WARMUP       = 200

# Phase B — adapt to secondary style (en_mistral), lower LR
PB_EPOCHS       = 10
PB_BATCH_SIZE   = 16
PB_LR           = 2e-6      # even lower LR for fine adjustment
PB_WEIGHT_DECAY = 0.05
PB_PATIENCE     = 4
PB_WARMUP       = 100

BEAM_SIZE       = 4
NO_REPEAT_NGRAM = 2
LENGTH_PENALTY  = 0.8

torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

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
        self.src = src_texts
        self.tgt = tgt_texts
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        model_inputs = self.tok(
            str(self.src[idx]).strip(),
            max_length=self.max_len, truncation=True, padding=False)
        with self.tok.as_target_tokenizer():
            labels = self.tok(
                str(self.tgt[idx]).strip(),
                max_length=self.max_len, truncation=True, padding=False)
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
# SECTION 3 — Training function
# ─────────────────────────────────────────────────────────────────────────────

def fine_tune(model, tokenizer, train_src, train_tgt, val_src, val_tgt,
              out_dir, phase_name, epochs, batch_size, lr,
              weight_decay, warmup, patience):
    print(f"\n{'='*60}")
    print(phase_name)
    print(f"{'='*60}")
    print(f"  Train: {len(train_src):,}  Val: {len(val_src):,}")
    print(f"  Epochs: {epochs}  LR: {lr}  Patience: {patience}")

    train_ds = TranslationDataset(train_src, train_tgt, tokenizer)
    val_ds   = TranslationDataset(val_src,   val_tgt,   tokenizer)
    collator = DataCollatorForSeq2Seq(tokenizer, model=model,
                                      padding=True, pad_to_multiple_of=8)

    args = Seq2SeqTrainingArguments(
        output_dir              = str(out_dir),
        num_train_epochs        = epochs,
        per_device_train_batch_size = batch_size,
        per_device_eval_batch_size  = batch_size * 2,
        learning_rate           = lr,
        warmup_steps            = warmup,
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
        logging_dir             = str(out_dir / "logs"),
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

    result  = trainer.train()
    logs    = [l for l in trainer.state.log_history if "loss" in l and "eval_loss" not in l]
    eval_logs = [l for l in trainer.state.log_history if "eval_bleu" in l]
    best_bleu = max((l["eval_bleu"] for l in eval_logs), default=0)

    print(f"\n  Train loss : {result.training_loss:.4f}")
    print(f"  Best BLEU  : {best_bleu:.2f}")
    return trainer, logs, best_bleu

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — Rhyme booster
# ─────────────────────────────────────────────────────────────────────────────

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

def get_last_word(line):
    words = line.strip().split()
    return re.sub(r'[^\w]', '', words[-1].lower()) if words else ""

def get_rhyme_group(word):
    for group in RHYME_GROUPS:
        if word in group:
            return group
    return None

def rhyme_booster(lines):
    if len(lines) < 2:
        return lines
    endings = [get_last_word(l) for l in lines]
    rhyme_counts = defaultdict(int)
    for end in endings:
        group = get_rhyme_group(end)
        if group:
            rhyme_counts[group[0]] += 1
    if not rhyme_counts:
        return lines
    dominant_key   = max(rhyme_counts, key=rhyme_counts.get)
    dominant_group = next(g for g in RHYME_GROUPS if g[0] == dominant_key)
    boosted = []
    for line in lines:
        last = get_last_word(line)
        if last in dominant_group:
            boosted.append(line)
            continue
        group = get_rhyme_group(last)
        if group:
            candidates = [w for w in dominant_group
                         if abs(len(w)-len(last)) <= 2 and w != last]
            if candidates:
                words = line.strip().split()
                punct = re.findall(r'[^\w]+$', words[-1])
                punct = punct[0] if punct else ""
                words[-1] = candidates[0] + punct
                boosted.append(" ".join(words))
            else:
                boosted.append(line)
        else:
            boosted.append(line)
    return boosted

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — Inference & Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def translate_line(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt",
                       max_length=MAX_LEN, truncation=True).to(DEVICE)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_length=MAX_LEN,
            num_beams=BEAM_SIZE, length_penalty=LENGTH_PENALTY,
            no_repeat_ngram_size=NO_REPEAT_NGRAM, early_stopping=True,
        )
    return tokenizer.decode(out[0], skip_special_tokens=True)

def translate_poem(model, tokenizer, poem, boost=True):
    lines = [l.strip() for l in poem.split("\n") if l.strip()]
    translated = [translate_line(model, tokenizer, l) for l in lines]
    if boost:
        translated = rhyme_booster(translated)
    return "\n".join(translated)

def rhyme_density(text):
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if not lines:
        return 0.0
    ends = [re.sub(r'[^\w]','',l.split()[-1].lower())[-3:]
            if l.split() else "" for l in lines]
    cnt = defaultdict(int)
    for e in ends:
        cnt[e] += 1
    return round(sum(1 for e in ends if cnt[e]>1)/len(ends), 3)

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
    count, prev = 0, False
    for ch in text:
        v = ch in vowels
        if v and not prev:
            count += 1
        prev = v
    if text.endswith('e') and count > 1:
        count -= 1
    return max(count, 1)

def syllable_alignment(src, tgt):
    sl = [l.strip() for l in src.split("\n") if l.strip()]
    tl = [l.strip() for l in tgt.split("\n") if l.strip()]
    if not sl or not tl:
        return 0.0
    n = min(len(sl), len(tl))
    scores = [min(count_devanagari_syllables(sl[i]),
                  count_english_syllables(tl[i])) /
              max(count_devanagari_syllables(sl[i]),
                  count_english_syllables(tl[i]))
              for i in range(n)]
    return round(sum(scores)/len(scores), 3)

def evaluate_poems(model, tokenizer, max_poems=100):
    poems_df = pd.read_csv(DATA_DIR / "poetry_poems_test.csv")
    results  = []
    model.eval()
    for i, row in tqdm(poems_df.head(max_poems).iterrows(),
                       total=min(max_poems, len(poems_df)),
                       desc="  Evaluating"):
        hi        = str(row["hi"]).strip()
        ref_lit   = str(row["en_t"]).strip()
        ref_style = str(row["en_anthropic"]).strip()
        hyp       = translate_poem(model, tokenizer, hi, boost=True)
        results.append({
            "id":                 i,
            "poet":               row["poet"],
            "source":             hi,
            "ref_literal":        ref_lit,
            "ref_style":          ref_style,
            "hypothesis":         hyp,
            "bleu_vs_literal":    round(sacrebleu.sentence_bleu(hyp,[ref_lit]).score,   2),
            "bleu_vs_style":      round(sacrebleu.sentence_bleu(hyp,[ref_style]).score, 2),
            "src_rhyme_density":  rhyme_density(hi),
            "tgt_rhyme_density":  rhyme_density(hyp),
            "syllable_alignment": syllable_alignment(hi, hyp),
        })
    return results

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("TIER 3 v3b — OPUS-MT (CURRICULUM LEARNING)")
    print("=" * 60)
    print("\nPhase A: hi→en_anthropic (primary style)")
    print("Phase B: hi→en_mistral   (secondary style, lower LR)")

    # ── Load data ─────────────────────────────────────────────────────────────
    print("\nLoading data...")
    train_df = pd.read_csv(DATA_DIR / "poetry_train.csv")
    val_df   = pd.read_csv(DATA_DIR / "poetry_val.csv")

    # Phase A data — en_anthropic only
    train_a = train_df.dropna(subset=["hi","en_anthropic"])
    train_a = train_a[train_a["en_anthropic"].str.strip() != ""]
    val_a   = val_df.dropna(subset=["hi","en_anthropic"])
    val_a   = val_a[val_a["en_anthropic"].str.strip() != ""]

    # Phase B data — en_mistral only
    train_b = train_df.dropna(subset=["hi","en_mistral"])
    train_b = train_b[train_b["en_mistral"].str.strip() != ""]
    val_b   = val_df.dropna(subset=["hi","en_mistral"])
    val_b   = val_b[val_b["en_mistral"].str.strip() != ""]

    print(f"  Phase A train: {len(train_a):,}  val: {len(val_a):,}")
    print(f"  Phase B train: {len(train_b):,}  val: {len(val_b):,}")

    # ── Load fresh pretrained model ───────────────────────────────────────────
    print(f"\nLoading pretrained base: {MODEL_NAME}")
    tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)
    model     = MarianMTModel.from_pretrained(MODEL_NAME).to(DEVICE)
    print(f"  Params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # ── Phase A ───────────────────────────────────────────────────────────────
    pa_out = MODELS_DIR / "phase_v3b_a"
    pa_out.mkdir(exist_ok=True)
    trainer_a, logs_a, best_bleu_a = fine_tune(
        model, tokenizer,
        train_src    = train_a["hi"].tolist(),
        train_tgt    = train_a["en_anthropic"].tolist(),
        val_src      = val_a["hi"].tolist(),
        val_tgt      = val_a["en_anthropic"].tolist(),
        out_dir      = pa_out,
        phase_name   = "PHASE A — Primary Style (en_anthropic)",
        epochs       = PA_EPOCHS,
        batch_size   = PA_BATCH_SIZE,
        lr           = PA_LR,
        weight_decay = PA_WEIGHT_DECAY,
        warmup       = PA_WARMUP,
        patience     = PA_PATIENCE,
    )

    # Save Phase A model
    pa_model_dir = MODELS_DIR / "tier3_v3b_phaseA"
    model.save_pretrained(pa_model_dir)
    tokenizer.save_pretrained(pa_model_dir)
    print(f"  Phase A model saved → {pa_model_dir}")

    # ── Phase B ───────────────────────────────────────────────────────────────
    pb_out = MODELS_DIR / "phase_v3b_b"
    pb_out.mkdir(exist_ok=True)
    trainer_b, logs_b, best_bleu_b = fine_tune(
        model, tokenizer,
        train_src    = train_b["hi"].tolist(),
        train_tgt    = train_b["en_mistral"].tolist(),
        val_src      = val_b["hi"].tolist(),
        val_tgt      = val_b["en_mistral"].tolist(),
        out_dir      = pb_out,
        phase_name   = "PHASE B — Secondary Style (en_mistral)",
        epochs       = PB_EPOCHS,
        batch_size   = PB_BATCH_SIZE,
        lr           = PB_LR,
        weight_decay = PB_WEIGHT_DECAY,
        warmup       = PB_WARMUP,
        patience     = PB_PATIENCE,
    )

    # ── Save final model ──────────────────────────────────────────────────────
    v3b_dir = MODELS_DIR / "tier3_v3b"
    model.save_pretrained(v3b_dir)
    tokenizer.save_pretrained(v3b_dir)
    print(f"\n  Final model saved → {v3b_dir}")

    # ── Loss plot ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    fig.suptitle("Tier 3 v3b — Curriculum Learning Loss", fontsize=13)
    for ax, logs, title in zip(axes, [logs_a, logs_b],
                               ["Phase A (en_anthropic)", "Phase B (en_mistral)"]):
        steps  = [l["step"] for l in logs]
        losses = [l["loss"] for l in logs]
        if steps:
            ax.plot(steps, losses, color="steelblue")
        ax.set_title(title); ax.set_xlabel("Step"); ax.set_ylabel("Loss")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "tier3_v3b_loss_curve.png", dpi=150)
    plt.close()

    # ── Evaluate ──────────────────────────────────────────────────────────────
    print("\nEvaluating on full poems...")
    results = evaluate_poems(model, tokenizer, max_poems=100)

    # ── Print samples ─────────────────────────────────────────────────────────
    print("\n── Sample Translations ──────────────────────────────────────")
    for r in results[:2]:
        print(f"\nSOURCE:\n{r['source'][:150]}")
        print(f"REF STYLE:\n{r['ref_style'][:150]}")
        print(f"HYPOTHESIS:\n{r['hypothesis'][:150]}")
        print(f"BLEU lit: {r['bleu_vs_literal']}  "
              f"BLEU style: {r['bleu_vs_style']}  "
              f"Rhyme: {r['tgt_rhyme_density']}  "
              f"Syllable: {r['syllable_alignment']}")

    # ── Save results ──────────────────────────────────────────────────────────
    csv_path = RESULTS_DIR / "tier3_v3b_translations.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader(); writer.writerows(results)

    bleu_lit   = [r["bleu_vs_literal"]    for r in results]
    bleu_style = [r["bleu_vs_style"]      for r in results]
    tgt_rhyme  = [r["tgt_rhyme_density"]  for r in results]
    syl_align  = [r["syllable_alignment"] for r in results]

    metrics = {
        "model":                  "Tier3_v3b_Curriculum",
        "strategy":               "Phase A: en_anthropic → Phase B: en_mistral",
        "phase_a_best_bleu":      best_bleu_a,
        "phase_b_best_bleu":      best_bleu_b,
        "avg_bleu_vs_literal":    round(sum(bleu_lit)  /len(bleu_lit),   2),
        "avg_bleu_vs_style":      round(sum(bleu_style)/len(bleu_style), 2),
        "avg_tgt_rhyme_density":  round(sum(tgt_rhyme) /len(tgt_rhyme),  3),
        "avg_syllable_alignment": round(sum(syl_align) /len(syl_align),  3),
    }

    with open(RESULTS_DIR / "tier3_v3b_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n── Final Metrics (v3b) ──────────────────────────────────────")
    for k, v in metrics.items():
        print(f"  {k:<30} {v}")

    print(f"\n✅ TIER 3 v3b COMPLETE")
    print(f"   Model   → {v3b_dir}")
    print(f"   Results → {RESULTS_DIR}")