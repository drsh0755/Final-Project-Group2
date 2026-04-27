"""
=============================================================================
Tier 3 v3a — Fine-tune Opus-MT: Option A (Interleaved)
DATS 6312 Final Project: Style-Preserving Hindi/Urdu to English Translation
=============================================================================
Strategy:
    Train on hi→en_anthropic + hi→en_mistral interleaved (~29K pairs)
    Model sees same Hindi line twice with different style-preserving targets
    Teaches range of acceptable poetic English

Saves to:
    Models/tier3_opusmt/tier3_v3a/
    Results/tier3_v3a/
    Plots/tier3_v3a_loss_curve.png

Run from: Code/ directory
    python3 tier3_opus_mt_v3a.py
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
RESULTS_DIR = BASE_DIR / "Results" / "tier3_v3a"
PLOTS_DIR   = BASE_DIR / "Plots"
for d in [MODELS_DIR, RESULTS_DIR, PLOTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME      = "Helsinki-NLP/opus-mt-hi-en"
MAX_LEN         = 128
RANDOM_SEED     = 42
P2_EPOCHS       = 30
P2_BATCH_SIZE   = 16
P2_LR           = 5e-6
P2_WEIGHT_DECAY = 0.05
P2_PATIENCE     = 5
P2_WARMUP       = 300
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
# SECTION 1 — Data loading (interleaved augmentation)
# ─────────────────────────────────────────────────────────────────────────────

def load_interleaved(split="train"):
    """
    Load poetry CSV and create pairs from both en_anthropic and en_mistral.
    Interleaved: [hi→anth_1, hi→mist_1, hi→anth_2, hi→mist_2, ...]
    """
    df = pd.read_csv(DATA_DIR / f"poetry_{split}.csv")
    pairs = []
    for _, row in df.iterrows():
        hi = str(row["hi"]).strip()
        if not hi or hi == "nan":
            continue
        anth  = str(row.get("en_anthropic", "")).strip()
        mist  = str(row.get("en_mistral",   "")).strip()
        if anth and anth != "nan":
            pairs.append((hi, anth))
        if mist and mist != "nan":
            pairs.append((hi, mist))
    # Shuffle while keeping pairs together
    random.shuffle(pairs)
    return pairs

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — Dataset
# ─────────────────────────────────────────────────────────────────────────────

class TranslationDataset(Dataset):
    def __init__(self, pairs, tokenizer, max_len=MAX_LEN):
        self.pairs = pairs
        self.tok   = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src, tgt = self.pairs[idx]
        model_inputs = self.tok(
            src, max_length=self.max_len, truncation=True, padding=False)
        with self.tok.as_target_tokenizer():
            labels = self.tok(
                tgt, max_length=self.max_len, truncation=True, padding=False)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — Metrics
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
# SECTION 4 — Rhyme booster (same as v2)
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
# SECTION 5 — Inference
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

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — Evaluation helpers
# ─────────────────────────────────────────────────────────────────────────────

def rhyme_density(text):
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if not lines:
        return 0.0
    ends  = [re.sub(r'[^\w]','',l.split()[-1].lower())[-3:]
             if l.split() else "" for l in lines]
    cnt   = defaultdict(int)
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
    print("TIER 3 v3a — OPUS-MT (INTERLEAVED AUGMENTATION)")
    print("=" * 60)
    print("\nStrategy: hi→en_anthropic + hi→en_mistral interleaved")

    # ── Load data ─────────────────────────────────────────────────────────────
    print("\nLoading interleaved training data...")
    train_pairs = load_interleaved("train")
    val_pairs   = load_interleaved("val")
    print(f"  Train pairs: {len(train_pairs):,}  (both translations)")
    print(f"  Val pairs  : {len(val_pairs):,}")

    # ── Load fresh pretrained model ───────────────────────────────────────────
    print(f"\nLoading pretrained base: {MODEL_NAME}")
    tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)
    model     = MarianMTModel.from_pretrained(MODEL_NAME).to(DEVICE)
    print(f"  Params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_ds = TranslationDataset(train_pairs, tokenizer)
    val_ds   = TranslationDataset(val_pairs,   tokenizer)
    collator = DataCollatorForSeq2Seq(tokenizer, model=model,
                                      padding=True, pad_to_multiple_of=8)

    # ── Training args ─────────────────────────────────────────────────────────
    out_dir = MODELS_DIR / "phase2_v3a"
    out_dir.mkdir(exist_ok=True)

    args = Seq2SeqTrainingArguments(
        output_dir              = str(out_dir),
        num_train_epochs        = P2_EPOCHS,
        per_device_train_batch_size = P2_BATCH_SIZE,
        per_device_eval_batch_size  = P2_BATCH_SIZE * 2,
        learning_rate           = P2_LR,
        warmup_steps            = P2_WARMUP,
        weight_decay            = P2_WEIGHT_DECAY,
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
        callbacks       = [EarlyStoppingCallback(early_stopping_patience=P2_PATIENCE)],
    )

    print(f"\nTraining — {P2_EPOCHS} epochs max, patience {P2_PATIENCE}...")
    result = trainer.train()
    print(f"\n  Train loss : {result.training_loss:.4f}")

    eval_logs = [l for l in trainer.state.log_history if "eval_bleu" in l]
    best_bleu = max((l["eval_bleu"] for l in eval_logs), default=0)
    print(f"  Best BLEU  : {best_bleu:.2f}")

    # ── Save models ───────────────────────────────────────────────────────────
    v3a_dir = MODELS_DIR / "tier3_v3a"
    model.save_pretrained(v3a_dir)
    tokenizer.save_pretrained(v3a_dir)
    print(f"\n  Saved → {v3a_dir}")

    # ── Loss plot ─────────────────────────────────────────────────────────────
    logs   = [l for l in trainer.state.log_history if "loss" in l and "eval_loss" not in l]
    steps  = [l["step"] for l in logs]
    losses = [l["loss"] for l in logs]
    plt.figure(figsize=(10, 4))
    if steps:
        plt.plot(steps, losses, color="steelblue", label="Train Loss")
    plt.xlabel("Step"); plt.ylabel("Loss")
    plt.title("Tier 3 v3a — Interleaved Fine-tuning Loss")
    plt.legend(); plt.tight_layout()
    plt.savefig(PLOTS_DIR / "tier3_v3a_loss_curve.png", dpi=150)
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
    csv_path = RESULTS_DIR / "tier3_v3a_translations.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader(); writer.writerows(results)

    bleu_lit   = [r["bleu_vs_literal"]    for r in results]
    bleu_style = [r["bleu_vs_style"]      for r in results]
    tgt_rhyme  = [r["tgt_rhyme_density"]  for r in results]
    syl_align  = [r["syllable_alignment"] for r in results]

    metrics = {
        "model":                  "Tier3_v3a_Interleaved",
        "strategy":               "hi→en_anthropic + hi→en_mistral interleaved",
        "train_pairs":            len(train_pairs),
        "best_eval_bleu":         best_bleu,
        "avg_bleu_vs_literal":    round(sum(bleu_lit)  /len(bleu_lit),   2),
        "avg_bleu_vs_style":      round(sum(bleu_style)/len(bleu_style), 2),
        "avg_tgt_rhyme_density":  round(sum(tgt_rhyme) /len(tgt_rhyme),  3),
        "avg_syllable_alignment": round(sum(syl_align) /len(syl_align),  3),
    }

    with open(RESULTS_DIR / "tier3_v3a_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n── Final Metrics (v3a) ──────────────────────────────────────")
    for k, v in metrics.items():
        print(f"  {k:<30} {v}")

    print(f"\n✅ TIER 3 v3a COMPLETE")
    print(f"   Model   → {v3a_dir}")
    print(f"   Results → {RESULTS_DIR}")