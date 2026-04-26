"""
=============================================================================
Unified Evaluation Script — All Three Tiers
DATS 6312 Final Project: Style-Preserving Hindi/Urdu to English Translation
=============================================================================
Loads saved models for all three tiers and evaluates on the same 100 poems.
Outputs a single comparison table for the report.

Run from: Code/ directory
    python3 evaluate_all_tiers.py

Output:
    Results/evaluation/comparison_table.csv
    Results/evaluation/comparison_table.txt   ← paste into report
    Results/evaluation/per_poem_results.csv
    Plots/evaluation_comparison.png
=============================================================================
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import json
import math
import csv
import re
from pathlib import Path
from collections import Counter, defaultdict
from tqdm import tqdm
import unicodedata

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sacrebleu

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent.parent
DATA_DIR    = BASE_DIR / "Data"  / "processed"
MODELS_DIR  = BASE_DIR / "Models"
RESULTS_DIR = BASE_DIR / "Results" / "evaluation"
PLOTS_DIR   = BASE_DIR / "Plots"
for d in [RESULTS_DIR, PLOTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_POEMS = 100

print("=" * 60)
print("UNIFIED EVALUATION — ALL THREE TIERS")
print("=" * 60)
print(f"Device : {DEVICE}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — Shared Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_bleu(hypothesis: str, reference: str) -> float:
    """Sentence-level BLEU using sacrebleu."""
    if not hypothesis.strip() or not reference.strip():
        return 0.0
    return round(sacrebleu.sentence_bleu(hypothesis, [reference]).score, 2)

def rhyme_density(text: str) -> float:
    """Fraction of lines sharing an end rhyme with at least one other line."""
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if not lines:
        return 0.0
    endings = []
    for l in lines:
        words = l.split()
        if words:
            # Last 3 chars of last word, lowercased, strip punctuation
            end = re.sub(r'[^\w]', '', words[-1].lower())[-3:]
            endings.append(end)
        else:
            endings.append("")
    counts = defaultdict(int)
    for e in endings:
        counts[e] += 1
    rhyming = sum(1 for e in endings if counts[e] > 1)
    return round(rhyming / len(endings), 3)

def count_devanagari_syllables(text: str) -> int:
    """
    Count syllables in Devanagari text.
    Each vowel sign (matra) or independent vowel = 1 syllable.
    Each consonant not followed by a vowel sign also carries inherent 'a' vowel.
    Simplified: count vowel-bearing units.
    """
    count = 0
    for ch in text:
        cat = unicodedata.category(ch)
        name = unicodedata.name(ch, "")
        # Devanagari vowels (independent) and vowel signs (matras)
        if "DEVANAGARI VOWEL" in name or "DEVANAGARI LETTER" in name:
            if cat in ("Lo", "Mc", "Mn"):
                count += 1
    return max(count, 1)

def count_english_syllables(text: str) -> int:
    """
    Count syllables in English text using vowel-group heuristic.
    """
    text = text.lower().strip()
    if not text:
        return 1
    vowels = "aeiouy"
    count = 0
    prev_vowel = False
    for ch in text:
        is_vowel = ch in vowels
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel
    # Silent e at end
    if text.endswith('e') and count > 1:
        count -= 1
    return max(count, 1)

def syllable_alignment_score(src_text: str, tgt_text: str) -> float:
    """
    Compare syllable counts line by line between source (Devanagari)
    and hypothesis (English). Score = mean(min/max) per line pair.
    """
    src_lines = [l.strip() for l in src_text.split("\n") if l.strip()]
    tgt_lines = [l.strip() for l in tgt_text.split("\n") if l.strip()]

    if not src_lines or not tgt_lines:
        return 0.0

    # Align to shorter length
    n = min(len(src_lines), len(tgt_lines))
    scores = []
    for i in range(n):
        sv = count_devanagari_syllables(src_lines[i])
        tv = count_english_syllables(tgt_lines[i])
        scores.append(min(sv, tv) / max(sv, tv))

    return round(sum(scores) / len(scores), 3)

def rhyme_scheme(text: str) -> str:
    """Detect AABB/ABAB style rhyme scheme."""
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    endings = []
    for l in lines:
        words = l.split()
        if words:
            end = re.sub(r'[^\w]', '', words[-1].lower())[-3:]
            endings.append(end)
    labels = {}
    scheme = []
    counter = 0
    for end in endings:
        if end not in labels:
            labels[end] = chr(ord('A') + counter % 26)
            counter += 1
        scheme.append(labels[end])
    return "".join(scheme)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — Tier 1: Rule-Based
# ─────────────────────────────────────────────────────────────────────────────

# Compact dictionary (key poetic vocabulary)
HINDI_DICT = {
    "मैं": "I", "मुझे": "me", "मेरा": "my", "मेरी": "my", "हम": "we",
    "तू": "you", "तुम": "you", "तेरा": "your", "तेरी": "your",
    "वह": "he/she", "वो": "he/she", "यह": "this", "वे": "they",
    "कोई": "someone", "कुछ": "something", "सब": "all", "हर": "every",
    "है": "is", "हैं": "are", "था": "was", "थी": "was", "थे": "were",
    "हो": "be", "हूँ": "am", "हुआ": "became",
    "जाना": "to go", "जाता": "goes", "जाती": "goes",
    "आना": "to come", "आता": "comes",
    "करना": "to do", "करता": "does", "कर": "do",
    "देना": "to give", "देता": "gives", "दिया": "gave",
    "कहना": "to say", "कहा": "said",
    "देखना": "to see", "देखा": "saw",
    "रहना": "to stay", "रहता": "stays",
    "जीना": "to live", "मरना": "to die",
    "खोना": "to lose", "खोया": "lost",
    "याद": "memory", "भूल": "forget",
    "दिल": "heart", "दिलों": "hearts",
    "प्यार": "love", "मुहब्बत": "love", "इश्क": "love", "प्रेम": "love",
    "दर्द": "pain", "गम": "sorrow", "तकलीफ": "suffering",
    "ख़ुशी": "happiness", "खुशी": "happiness",
    "आँसू": "tears", "आंसू": "tears",
    "उम्मीद": "hope", "आशा": "hope",
    "ख़्वाब": "dream", "सपना": "dream",
    "डर": "fear", "तड़प": "longing",
    "आरज़ू": "desire", "सुकून": "peace",
    "रात": "night", "दिन": "day",
    "चाँद": "moon", "सूरज": "sun",
    "आसमान": "sky", "तारे": "stars", "सितारे": "stars",
    "बादल": "cloud", "बारिश": "rain", "हवा": "wind",
    "फूल": "flower", "पेड़": "tree",
    "रोशनी": "light", "अँधेरा": "darkness", "आग": "fire",
    "शाम": "evening", "सुबह": "morning",
    "आँखें": "eyes", "होंठ": "lips", "हाथ": "hand",
    "दिमाग": "mind", "मन": "mind",
    "रूह": "soul", "आत्मा": "soul",
    "जिंदगी": "life", "ज़िंदगी": "life", "जीवन": "life",
    "मौत": "death", "साँस": "breath",
    "सुंदर": "beautiful", "खूबसूरत": "beautiful",
    "अच्छा": "good", "बुरा": "bad",
    "सच्चा": "true", "झूठा": "false",
    "में": "in", "पर": "on", "से": "from",
    "को": "to", "के": "of", "की": "of", "का": "of",
    "साथ": "with", "लिए": "for", "बिना": "without",
    "अब": "now", "फिर": "again", "कभी": "ever",
    "हमेशा": "always", "आज": "today",
    "नहीं": "not", "ना": "no", "और": "and",
    "या": "or", "लेकिन": "but", "भी": "also",
    "शायर": "poet", "शायरी": "poetry",
    "ग़ज़ल": "ghazal", "दीवाना": "madman/lover",
    "मंजिल": "destination", "राह": "path",
    "मुसाफिर": "traveler", "सफर": "journey",
    "वक्त": "time", "दुनिया": "world",
    "खुदा": "God", "दुआ": "prayer",
}

def tier1_translate_line(line: str) -> str:
    tokens = re.sub(r'[।\.!?,;:\-"\']+', ' ', line).split()
    translated = [HINDI_DICT.get(t, HINDI_DICT.get(t.lower(), f"[{t}]")) for t in tokens]
    return " ".join(translated)

def tier1_translate_poem(poem: str) -> str:
    lines = [l.strip() for l in poem.split("\n") if l.strip()]
    return "\n".join(tier1_translate_line(l) for l in lines)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — Tier 2: Seq2Seq LSTM
# ─────────────────────────────────────────────────────────────────────────────

PAD, SOS, EOS, UNK = "<pad>", "<sos>", "<eos>", "<unk>"
MAX_LEN = 50

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.LSTM(embed_dim, hidden_dim, n_layers,
                           bidirectional=True,
                           dropout=dropout if n_layers > 1 else 0,
                           batch_first=True)
        self.dropout   = nn.Dropout(dropout)
        self.fc_hidden = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc_cell   = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, src, src_lens):
        embedded = self.dropout(self.embedding(src))
        packed   = nn.utils.rnn.pack_padded_sequence(
            embedded, src_lens.cpu(), batch_first=True, enforce_sorted=False)
        outputs, (hidden, cell) = self.rnn(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        hidden = torch.tanh(self.fc_hidden(torch.cat([hidden[-2], hidden[-1]], 1)))
        cell   = torch.tanh(self.fc_cell(  torch.cat([cell[-2],   cell[-1]],   1)))
        hidden = hidden.unsqueeze(0).repeat(2, 1, 1)
        cell   = cell.unsqueeze(0).repeat(2, 1, 1)
        return outputs, hidden, cell

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_dim, enc_hidden_dim):
        super().__init__()
        self.W_s = nn.Linear(hidden_dim,     hidden_dim, bias=False)
        self.W_h = nn.Linear(enc_hidden_dim, hidden_dim, bias=False)
        self.v   = nn.Linear(hidden_dim, 1,              bias=False)

    def forward(self, dec_hidden, enc_outputs):
        src_len = enc_outputs.size(1)
        s = self.W_s(dec_hidden).unsqueeze(1).repeat(1, src_len, 1)
        h = self.W_h(enc_outputs)
        attn = self.v(torch.tanh(s + h)).squeeze(2)
        attn = torch.softmax(attn, dim=1)
        ctx  = torch.bmm(attn.unsqueeze(1), enc_outputs).squeeze(1)
        return ctx, attn

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, enc_hidden_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.attention = BahdanauAttention(hidden_dim, enc_hidden_dim)
        self.rnn       = nn.LSTM(embed_dim + enc_hidden_dim, hidden_dim, n_layers,
                                 dropout=dropout if n_layers > 1 else 0,
                                 batch_first=True)
        self.fc_out    = nn.Linear(hidden_dim + enc_hidden_dim + embed_dim, vocab_size)
        self.dropout   = nn.Dropout(dropout)

    def forward_step(self, token, hidden, cell, enc_outputs):
        emb  = self.dropout(self.embedding(token.unsqueeze(1)))
        ctx, attn = self.attention(hidden[-1], enc_outputs)
        out, (hidden, cell) = self.rnn(
            torch.cat([emb, ctx.unsqueeze(1)], dim=2), (hidden, cell))
        pred = self.fc_out(torch.cat([out.squeeze(1), ctx, emb.squeeze(1)], dim=1))
        return pred, hidden, cell, attn

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device  = device

    def translate(self, src_tensor, src_len, tgt_vocab, max_len=MAX_LEN):
        self.eval()
        with torch.no_grad():
            enc_out, hidden, cell = self.encoder(
                src_tensor.unsqueeze(0), torch.tensor([src_len]))
            token  = torch.tensor([tgt_vocab["word2idx"][SOS]]).to(self.device)
            tokens = []
            for _ in range(max_len):
                pred, hidden, cell, _ = self.decoder.forward_step(
                    token, hidden, cell, enc_out)
                top = pred.argmax(1).item()
                if top == tgt_vocab["word2idx"][EOS]:
                    break
                tokens.append(top)
                token = torch.tensor([top]).to(self.device)
        words = []
        for idx in tokens:
            w = tgt_vocab["idx2word"].get(str(idx), UNK)
            if w not in (PAD, SOS, EOS):
                words.append(w)
        return " ".join(words)

def load_tier2_model():
    ckpt_path  = MODELS_DIR / "tier2_seq2seq" / "best_model.pt"
    vocab_path = MODELS_DIR / "tier2_seq2seq" / "vocabs.pt"
    if not ckpt_path.exists():
        print(f"  ⚠️  Tier 2 model not found at {ckpt_path}")
        return None, None, None

    ckpt  = torch.load(ckpt_path,  map_location=DEVICE, weights_only=False)
    vocab = torch.load(vocab_path, map_location=DEVICE, weights_only=False)

    hp = ckpt["hyperparams"]
    src_vocab_size = ckpt["src_vocab_size"]
    tgt_vocab_size = ckpt["tgt_vocab_size"]

    encoder = Encoder(src_vocab_size, hp["embed_dim"], hp["hidden_dim"],
                      hp["enc_layers"], 0.0).to(DEVICE)
    decoder = Decoder(tgt_vocab_size, hp["embed_dim"], hp["hidden_dim"],
                      hp["hidden_dim"]*2, hp["dec_layers"], 0.0).to(DEVICE)
    model = Seq2Seq(encoder, decoder, DEVICE).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Convert idx2word keys to strings for lookup
    vocab["tgt_idx2word"] = {str(k): v for k, v in vocab["tgt_idx2word"].items()}
    print(f"  Tier 2 loaded — epoch {ckpt['epoch']}, val_loss {ckpt['val_loss']:.4f}")
    return model, vocab, hp

def tier2_translate_poem(model, vocab, poem: str) -> str:
    src_w2i = vocab["src_word2idx"]
    tgt_vocab_dict = {
        "word2idx": vocab["tgt_word2idx"],
        "idx2word": vocab["tgt_idx2word"],
    }
    lines = [l.strip() for l in poem.split("\n") if l.strip()]
    translated = []
    for line in lines:
        tokens = [src_w2i.get(SOS, 1)] + \
                 [src_w2i.get(t, src_w2i.get(UNK, 3)) for t in line.split()[:MAX_LEN]] + \
                 [src_w2i.get(EOS, 2)]
        src_t = torch.tensor(tokens, dtype=torch.long).to(DEVICE)
        translated.append(model.translate(src_t, len(tokens), tgt_vocab_dict))
    return "\n".join(translated)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — Tier 3: Opus-MT
# ─────────────────────────────────────────────────────────────────────────────

def load_tier3_model():
    from transformers import MarianMTModel, MarianTokenizer
    model_path = MODELS_DIR / "tier3_opusmt" / "best_model"
    if not model_path.exists():
        print(f"  ⚠️  Tier 3 model not found at {model_path}")
        return None, None
    tokenizer = MarianTokenizer.from_pretrained(str(model_path))
    model     = MarianMTModel.from_pretrained(str(model_path)).to(DEVICE)
    model.eval()
    print(f"  Tier 3 loaded from {model_path}")
    return model, tokenizer

def tier3_translate_poem(model, tokenizer, poem: str) -> str:
    lines = [l.strip() for l in poem.split("\n") if l.strip()]
    translated = []
    for line in lines:
        inputs = tokenizer(line, return_tensors="pt",
                           max_length=128, truncation=True).to(DEVICE)
        with torch.no_grad():
            out = model.generate(**inputs, max_length=128,
                                 num_beams=4, early_stopping=True,
                                 no_repeat_ngram_size=3)
        translated.append(tokenizer.decode(out[0], skip_special_tokens=True))
    return "\n".join(translated)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — Run Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_tier(name, translate_fn, poems_df):
    results = []
    for _, row in tqdm(poems_df.iterrows(), total=len(poems_df),
                       desc=f"  {name}"):
        hi        = str(row["hi"]).strip()
        ref_lit   = str(row["en_t"]).strip()
        ref_style = str(row["en_anthropic"]).strip()
        poet      = str(row["poet"])

        hyp = translate_fn(hi)

        results.append({
            "tier":               name,
            "poet":               poet,
            "poem_slug":          str(row["poem_slug"]),
            "bleu_vs_literal":    compute_bleu(hyp, ref_lit),
            "bleu_vs_style":      compute_bleu(hyp, ref_style),
            "src_rhyme_density":  rhyme_density(hi),
            "tgt_rhyme_density":  rhyme_density(hyp),
            "syllable_alignment": syllable_alignment_score(hi, hyp),
            "src_rhyme_scheme":   rhyme_scheme(hi),
            "tgt_rhyme_scheme":   rhyme_scheme(hyp),
            "hypothesis":         hyp,
            "ref_literal":        ref_lit,
            "ref_style":          ref_style,
            "source":             hi,
        })
    return results

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — Comparison Plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_comparison(summary):
    tiers   = [s["tier"] for s in summary]
    metrics = {
        "BLEU vs Literal":     [s["avg_bleu_vs_literal"]   for s in summary],
        "BLEU vs Style":       [s["avg_bleu_vs_style"]     for s in summary],
        "Tgt Rhyme Density":   [s["avg_tgt_rhyme_density"] for s in summary],
        "Syllable Alignment":  [s["avg_syllable_alignment"]for s in summary],
    }

    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    fig.suptitle("Model Comparison — All Three Tiers", fontsize=14, fontweight="bold")
    colors = ["#e74c3c", "#3498db", "#2ecc71"]

    for ax, (metric, values) in zip(axes, metrics.items()):
        bars = ax.bar(tiers, values, color=colors[:len(tiers)], edgecolor="white", width=0.5)
        ax.set_title(metric, fontsize=11)
        ax.set_ylim(0, max(values) * 1.3 + 0.01)
        ax.set_ylabel("Score")
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=9)
        ax.tick_params(axis='x', labelsize=8)

    plt.tight_layout()
    path = PLOTS_DIR / "evaluation_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # Load poems
    poems_df = pd.read_csv(DATA_DIR / "poetry_poems_test.csv").head(MAX_POEMS)
    print(f"\nEvaluating on {len(poems_df)} poems\n")

    all_results = []

    # ── Tier 1 ────────────────────────────────────────────────────────────────
    print("Loading Tier 1 (Rule-Based)...")
    t1_results = evaluate_tier("Tier1_RuleBased", tier1_translate_poem, poems_df)
    all_results.extend(t1_results)

    # ── Tier 2 ────────────────────────────────────────────────────────────────
    print("\nLoading Tier 2 (Seq2Seq LSTM)...")
    t2_model, t2_vocab, _ = load_tier2_model()
    if t2_model:
        t2_fn = lambda poem: tier2_translate_poem(t2_model, t2_vocab, poem)
        t2_results = evaluate_tier("Tier2_Seq2Seq", t2_fn, poems_df)
        all_results.extend(t2_results)
    else:
        t2_results = []

    # ── Tier 3 ────────────────────────────────────────────────────────────────
    print("\nLoading Tier 3 (Opus-MT)...")
    t3_model, t3_tokenizer = load_tier3_model()
    if t3_model:
        t3_fn = lambda poem: tier3_translate_poem(t3_model, t3_tokenizer, poem)
        t3_results = evaluate_tier("Tier3_OpusMT", t3_fn, poems_df)
        all_results.extend(t3_results)
    else:
        t3_results = []

    # ── Save per-poem results ─────────────────────────────────────────────────
    per_poem_path = RESULTS_DIR / "per_poem_results.csv"
    with open(per_poem_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\n  Saved per-poem results → {per_poem_path}")

    # ── Build summary table ───────────────────────────────────────────────────
    def summarize(results, name):
        if not results:
            return None
        return {
            "tier":                  name,
            "num_poems":             len(results),
            "avg_bleu_vs_literal":   round(sum(r["bleu_vs_literal"]    for r in results)/len(results), 2),
            "avg_bleu_vs_style":     round(sum(r["bleu_vs_style"]      for r in results)/len(results), 2),
            "avg_src_rhyme_density": round(sum(r["src_rhyme_density"]  for r in results)/len(results), 3),
            "avg_tgt_rhyme_density": round(sum(r["tgt_rhyme_density"]  for r in results)/len(results), 3),
            "avg_syllable_alignment":round(sum(r["syllable_alignment"] for r in results)/len(results), 3),
        }

    summary = [s for s in [
        summarize(t1_results, "Tier1_RuleBased"),
        summarize(t2_results, "Tier2_Seq2Seq"),
        summarize(t3_results, "Tier3_OpusMT"),
    ] if s]

    # ── Save comparison table CSV ─────────────────────────────────────────────
    table_csv = RESULTS_DIR / "comparison_table.csv"
    with open(table_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)
    print(f"  Saved comparison table → {table_csv}")

    # ── Save comparison table TXT (for report) ────────────────────────────────
    table_txt = RESULTS_DIR / "comparison_table.txt"
    col_w = 26
    metrics = ["avg_bleu_vs_literal", "avg_bleu_vs_style",
               "avg_tgt_rhyme_density", "avg_syllable_alignment"]
    metric_labels = {
        "avg_bleu_vs_literal":    "BLEU vs Literal",
        "avg_bleu_vs_style":      "BLEU vs Style",
        "avg_tgt_rhyme_density":  "Rhyme Density (tgt)",
        "avg_syllable_alignment": "Syllable Alignment",
    }

    with open(table_txt, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("MODEL COMPARISON TABLE\n")
        f.write("DATS 6312 Final Project — Style-Preserving Hindi/Urdu Translation\n")
        f.write("=" * 70 + "\n\n")

        # Header
        header = f"{'Metric':<28}" + "".join(f"{s['tier']:<22}" for s in summary)
        f.write(header + "\n")
        f.write("-" * 70 + "\n")

        for m in metrics:
            row = f"{metric_labels[m]:<28}" + \
                  "".join(f"{s[m]:<22}" for s in summary)
            f.write(row + "\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("Notes:\n")
        f.write("  BLEU vs Literal : compared against deep-translator output (en_t)\n")
        f.write("  BLEU vs Style   : compared against Claude style-preserving translation (en_anthropic)\n")
        f.write("  Rhyme Density   : fraction of output lines sharing end rhyme\n")
        f.write("  Syllable Align  : Devanagari syllable count vs English syllable count per line\n")

    print(f"  Saved comparison TXT  → {table_txt}")

    # ── Print table to terminal ───────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("FINAL COMPARISON TABLE")
    print("=" * 70)
    with open(table_txt) as f:
        print(f.read())

    # ── Plot ──────────────────────────────────────────────────────────────────
    print("Plotting comparison chart...")
    plot_comparison(summary)

    # ── Save summary JSON ─────────────────────────────────────────────────────
    json_path = RESULTS_DIR / "comparison_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved summary JSON → {json_path}")

    print("\n✅ EVALUATION COMPLETE")
    print(f"   Results → {RESULTS_DIR}")
    print(f"   Plots   → {PLOTS_DIR / 'evaluation_comparison.png'}")
    print(f"\n   Copy {table_txt} into your report Section 6.")