"""
=============================================================================
Tier 2 v2 — Improved Seq2Seq LSTM with Bahdanau Attention
DATS 6312 Final Project: Style-Preserving Hindi/Urdu to English Translation
=============================================================================
Improvements over v1:
    1. Data augmentation — use hi→en_anthropic + hi→en_mistral + hi→en_t
       (~43K pairs vs 14K in v1)
    2. Lower LR: 0.001 → 0.0005
    3. More epochs: 30 → 50 with patience 10
    4. Larger hidden dim: 512 → 768
    5. Separate embedding dropout (0.3) from LSTM dropout (0.2)
    6. Label smoothing in loss function

Run from: Code/ directory
    python3 tier2_seq2seq_lstm_v2.py

Output:
    Models/tier2_seq2seq_v2/
    Results/tier2_v2/
=============================================================================
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import json
import math
import random
import time
import csv
import re
from pathlib import Path
from collections import Counter
from tqdm import tqdm
import sacrebleu as sb

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent.parent
DATA_DIR    = BASE_DIR / "Data"  / "processed"
MODELS_DIR  = BASE_DIR / "Models" / "tier2_seq2seq_v2"
RESULTS_DIR = BASE_DIR / "Results" / "tier2_v2"
PLOTS_DIR   = BASE_DIR / "Plots"
for d in [MODELS_DIR, RESULTS_DIR, PLOTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Hyperparameters (v2) ──────────────────────────────────────────────────────
EMBED_DIM     = 256
HIDDEN_DIM    = 768       # v1: 512 → v2: 768
ENC_LAYERS    = 2
DEC_LAYERS    = 2
EMBED_DROPOUT = 0.3       # separate embedding dropout
LSTM_DROPOUT  = 0.2       # LSTM layer dropout
BATCH_SIZE    = 128
EPOCHS        = 50        # v1: 30 → v2: 50
LEARNING_RATE = 0.0005    # v1: 0.001 → v2: 0.0005
CLIP          = 1.0
TEACHER_FORCE = 0.5
LABEL_SMOOTH  = 0.1       # new: label smoothing
MIN_FREQ      = 2
MAX_LEN       = 50
PATIENCE      = 10        # v1: 7 → v2: 10
RANDOM_SEED   = 42

PAD, SOS, EOS, UNK = "<pad>", "<sos>", "<eos>", "<unk>"

torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU   : {torch.cuda.get_device_name(0)}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — Data Augmentation
# ─────────────────────────────────────────────────────────────────────────────

def load_augmented_data(split="train"):
    """
    Load poetry CSV and augment by using multiple translation columns
    as separate training pairs.
    Returns list of (src, tgt) tuples.
    """
    df = pd.read_csv(DATA_DIR / f"poetry_{split}.csv")
    pairs = []

    translation_cols = ["en_anthropic", "en_mistral", "en_t"]

    for _, row in df.iterrows():
        hi = str(row["hi"]).strip()
        if not hi:
            continue
        for col in translation_cols:
            tgt = str(row.get(col, "")).strip()
            if tgt and tgt != "nan":
                pairs.append((hi, tgt))

    # Deduplicate
    pairs = list(set(pairs))
    random.shuffle(pairs)
    return pairs

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — Vocabulary
# ─────────────────────────────────────────────────────────────────────────────

class Vocabulary:
    def __init__(self, name):
        self.name = name
        self.word2idx = {PAD: 0, SOS: 1, EOS: 2, UNK: 3}
        self.idx2word = {0: PAD, 1: SOS, 2: EOS, 3: UNK}
        self.word_count = Counter()

    def build(self, sentences, min_freq=MIN_FREQ):
        for sent in sentences:
            self.word_count.update(sent.split())
        for word, count in self.word_count.items():
            if count >= min_freq and word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word
        print(f"  [{self.name}] vocab: {len(self.word2idx):,}")

    def encode(self, sentence):
        tokens = sentence.split()[:MAX_LEN]
        return ([self.word2idx[SOS]] +
                [self.word2idx.get(t, self.word2idx[UNK]) for t in tokens] +
                [self.word2idx[EOS]])

    def decode(self, indices):
        words = []
        for idx in indices:
            w = self.idx2word.get(idx, UNK)
            if w == EOS:
                break
            if w not in (PAD, SOS):
                words.append(w)
        return " ".join(words)

    def __len__(self):
        return len(self.word2idx)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — Dataset
# ─────────────────────────────────────────────────────────────────────────────

class PoetryDataset(Dataset):
    def __init__(self, pairs, src_vocab, tgt_vocab):
        self.data = []
        for src, tgt in pairs:
            self.data.append((src_vocab.encode(src), tgt_vocab.encode(tgt)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_max = max(len(s) for s in src_batch)
    tgt_max = max(len(t) for t in tgt_batch)
    src_lens = [len(s) for s in src_batch]
    src_pad  = [s + [0]*(src_max-len(s)) for s in src_batch]
    tgt_pad  = [t + [0]*(tgt_max-len(t)) for t in tgt_batch]
    return (torch.tensor(src_pad, dtype=torch.long),
            torch.tensor(tgt_pad, dtype=torch.long),
            torch.tensor(src_lens, dtype=torch.long))

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — Model
# ─────────────────────────────────────────────────────────────────────────────

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_layers,
                 embed_dropout, lstm_dropout):
        super().__init__()
        self.embedding    = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embed_drop   = nn.Dropout(embed_dropout)
        self.rnn          = nn.LSTM(embed_dim, hidden_dim, n_layers,
                                    bidirectional=True,
                                    dropout=lstm_dropout if n_layers>1 else 0,
                                    batch_first=True)
        self.fc_hidden    = nn.Linear(hidden_dim*2, hidden_dim)
        self.fc_cell      = nn.Linear(hidden_dim*2, hidden_dim)

    def forward(self, src, src_lens):
        emb    = self.embed_drop(self.embedding(src))
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, src_lens.cpu(), batch_first=True, enforce_sorted=False)
        out, (h, c) = self.rnn(packed)
        out, _      = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        h = torch.tanh(self.fc_hidden(torch.cat([h[-2], h[-1]], 1)))
        c = torch.tanh(self.fc_cell(  torch.cat([c[-2], c[-1]], 1)))
        h = h.unsqueeze(0).repeat(DEC_LAYERS, 1, 1)
        c = c.unsqueeze(0).repeat(DEC_LAYERS, 1, 1)
        return out, h, c

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_dim, enc_hidden_dim):
        super().__init__()
        self.W_s = nn.Linear(hidden_dim,     hidden_dim, bias=False)
        self.W_h = nn.Linear(enc_hidden_dim, hidden_dim, bias=False)
        self.v   = nn.Linear(hidden_dim, 1,              bias=False)

    def forward(self, dec_h, enc_out):
        n = enc_out.size(1)
        s = self.W_s(dec_h).unsqueeze(1).repeat(1, n, 1)
        h = self.W_h(enc_out)
        a = torch.softmax(self.v(torch.tanh(s+h)).squeeze(2), dim=1)
        ctx = torch.bmm(a.unsqueeze(1), enc_out).squeeze(1)
        return ctx, a

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, enc_hidden_dim,
                 n_layers, embed_dropout, lstm_dropout):
        super().__init__()
        self.embedding  = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embed_drop = nn.Dropout(embed_dropout)
        self.attention  = BahdanauAttention(hidden_dim, enc_hidden_dim)
        self.rnn        = nn.LSTM(embed_dim+enc_hidden_dim, hidden_dim, n_layers,
                                  dropout=lstm_dropout if n_layers>1 else 0,
                                  batch_first=True)
        self.fc_out     = nn.Linear(hidden_dim+enc_hidden_dim+embed_dim, vocab_size)

    def forward_step(self, token, hidden, cell, enc_out):
        emb = self.embed_drop(self.embedding(token.unsqueeze(1)))
        ctx, attn = self.attention(hidden[-1], enc_out)
        out, (hidden, cell) = self.rnn(
            torch.cat([emb, ctx.unsqueeze(1)], dim=2), (hidden, cell))
        pred = self.fc_out(torch.cat([out.squeeze(1), ctx, emb.squeeze(1)], 1))
        return pred, hidden, cell, attn

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, tgt_vocab_size, device):
        super().__init__()
        self.encoder       = encoder
        self.decoder       = decoder
        self.tgt_vocab_size = tgt_vocab_size
        self.device        = device

    def forward(self, src, tgt, src_lens, tf_ratio=TEACHER_FORCE):
        B, T    = src.size(0), tgt.size(1)
        outputs = torch.zeros(B, T, self.tgt_vocab_size).to(self.device)
        enc_out, h, c = self.encoder(src, src_lens)
        inp = tgt[:, 0]
        for t in range(1, T):
            pred, h, c, _ = self.decoder.forward_step(inp, h, c, enc_out)
            outputs[:, t] = pred
            inp = tgt[:, t] if random.random() < tf_ratio else pred.argmax(1)
        return outputs

    def translate(self, src_tensor, src_len, tgt_vocab, max_len=MAX_LEN):
        self.eval()
        with torch.no_grad():
            enc_out, h, c = self.encoder(
                src_tensor.unsqueeze(0), torch.tensor([src_len]))
            inp    = torch.tensor([tgt_vocab.word2idx[SOS]]).to(self.device)
            tokens = []
            for _ in range(max_len):
                pred, h, c, _ = self.decoder.forward_step(inp, h, c, enc_out)
                top = pred.argmax(1).item()
                if top == tgt_vocab.word2idx[EOS]:
                    break
                tokens.append(top)
                inp = torch.tensor([top]).to(self.device)
        return tgt_vocab.decode(tokens)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — Training
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, clip):
    model.train()
    total = 0
    for src, tgt, lens in tqdm(loader, desc="  Train", leave=False):
        src, tgt = src.to(DEVICE), tgt.to(DEVICE)
        optimizer.zero_grad()
        out  = model(src, tgt, lens)
        loss = criterion(out[:, 1:].reshape(-1, out.size(-1)),
                         tgt[:, 1:].reshape(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        total += loss.item()
    return total / len(loader)

def val_epoch(model, loader, criterion):
    model.eval()
    total = 0
    with torch.no_grad():
        for src, tgt, lens in tqdm(loader, desc="  Val  ", leave=False):
            src, tgt = src.to(DEVICE), tgt.to(DEVICE)
            out  = model(src, tgt, lens, tf_ratio=0)
            loss = criterion(out[:, 1:].reshape(-1, out.size(-1)),
                             tgt[:, 1:].reshape(-1))
            total += loss.item()
    return total / len(loader)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — Evaluation helpers
# ─────────────────────────────────────────────────────────────────────────────

def translate_poem(model, src_vocab, tgt_vocab, poem):
    lines = [l.strip() for l in poem.split("\n") if l.strip()]
    out   = []
    for line in lines:
        enc = src_vocab.encode(line)
        src = torch.tensor(enc, dtype=torch.long).to(DEVICE)
        out.append(model.translate(src, len(enc), tgt_vocab))
    return "\n".join(out)

def rhyme_density(text):
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if not lines:
        return 0.0
    from collections import defaultdict
    ends  = [re.sub(r'[^\w]','',l.split()[-1].lower())[-3:]
             if l.split() else "" for l in lines]
    cnt   = defaultdict(int)
    for e in ends:
        cnt[e] += 1
    return round(sum(1 for e in ends if cnt[e]>1)/len(ends), 3)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("TIER 2 v2 — SEQ2SEQ LSTM (IMPROVED)")
    print("=" * 60)

    # ── Load augmented data ───────────────────────────────────────────────────
    print("\nLoading augmented training data...")
    train_pairs = load_augmented_data("train")
    val_pairs   = load_augmented_data("val")
    print(f"  Train pairs (augmented): {len(train_pairs):,}")
    print(f"  Val pairs   (augmented): {len(val_pairs):,}")

    # ── Build vocabularies ────────────────────────────────────────────────────
    print("\nBuilding vocabularies...")
    src_vocab = Vocabulary("hindi")
    tgt_vocab = Vocabulary("english")
    src_vocab.build([p[0] for p in train_pairs])
    tgt_vocab.build([p[1] for p in train_pairs])

    torch.save({
        "src_word2idx": src_vocab.word2idx,
        "src_idx2word": src_vocab.idx2word,
        "tgt_word2idx": tgt_vocab.word2idx,
        "tgt_idx2word": tgt_vocab.idx2word,
    }, MODELS_DIR / "vocabs.pt")

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_ds = PoetryDataset(train_pairs, src_vocab, tgt_vocab)
    val_ds   = PoetryDataset(val_pairs,   src_vocab, tgt_vocab)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_fn, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              collate_fn=collate_fn, num_workers=2, pin_memory=True)

    # ── Build model ───────────────────────────────────────────────────────────
    print("\nBuilding model (v2)...")
    encoder = Encoder(len(src_vocab), EMBED_DIM, HIDDEN_DIM, ENC_LAYERS,
                      EMBED_DROPOUT, LSTM_DROPOUT).to(DEVICE)
    decoder = Decoder(len(tgt_vocab), EMBED_DIM, HIDDEN_DIM, HIDDEN_DIM*2,
                      DEC_LAYERS, EMBED_DROPOUT, LSTM_DROPOUT).to(DEVICE)
    model   = Seq2Seq(encoder, decoder, len(tgt_vocab), DEVICE).to(DEVICE)
    print(f"  Params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # ── Optimizer + loss ──────────────────────────────────────────────────────
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=4)
    criterion = nn.CrossEntropyLoss(ignore_index=0,
                                    label_smoothing=LABEL_SMOOTH)

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"\nTraining for up to {EPOCHS} epochs (patience={PATIENCE})...")
    best_val, train_losses, val_losses = float("inf"), [], []
    no_improve = 0

    for epoch in range(1, EPOCHS+1):
        t0         = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, criterion, CLIP)
        val_loss   = val_epoch(model, val_loader, criterion)
        scheduler.step(val_loss)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"  Epoch {epoch:02d}/{EPOCHS} | "
              f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
              f"PPL: {math.exp(val_loss):.2f} | {time.time()-t0:.1f}s")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "epoch": epoch, "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_loss": val_loss,
                "src_vocab_size": len(src_vocab),
                "tgt_vocab_size": len(tgt_vocab),
                "hyperparams": {
                    "embed_dim": EMBED_DIM, "hidden_dim": HIDDEN_DIM,
                    "enc_layers": ENC_LAYERS, "dec_layers": DEC_LAYERS,
                }
            }, MODELS_DIR / "best_model.pt")
            print(f"    ✓ Saved best model")
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"\n  Early stopping at epoch {epoch}")
                break

    # ── Loss plot ─────────────────────────────────────────────────────────────
    plt.figure(figsize=(10, 4))
    plt.plot(train_losses, label="Train", color="steelblue")
    plt.plot(val_losses,   label="Val",   color="darkorange")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title("Tier 2 v2 — Training Loss")
    plt.legend(); plt.tight_layout()
    plt.savefig(PLOTS_DIR / "tier2_v2_loss_curve.png", dpi=150)
    plt.close()

    # ── Load best & evaluate ──────────────────────────────────────────────────
    print("\nLoading best model for evaluation...")
    ckpt = torch.load(MODELS_DIR / "best_model.pt", map_location=DEVICE,
                      weights_only=False)
    model.load_state_dict(ckpt["model_state"])

    poems_df = pd.read_csv(DATA_DIR / "poetry_poems_test.csv").head(100)
    results  = []
    for _, row in tqdm(poems_df.iterrows(), total=len(poems_df),
                       desc="  Evaluating"):
        hi        = str(row["hi"]).strip()
        ref_lit   = str(row["en_t"]).strip()
        ref_style = str(row["en_anthropic"]).strip()
        hyp       = translate_poem(model, src_vocab, tgt_vocab, hi)
        results.append({
            "id":                row.name,
            "poet":              row["poet"],
            "source":            hi,
            "ref_literal":       ref_lit,
            "ref_style":         ref_style,
            "hypothesis":        hyp,
            "bleu_vs_literal":   round(sb.sentence_bleu(hyp,[ref_lit]).score,  2),
            "bleu_vs_style":     round(sb.sentence_bleu(hyp,[ref_style]).score,2),
            "src_rhyme_density": rhyme_density(hi),
            "tgt_rhyme_density": rhyme_density(hyp),
        })

    # ── Save ──────────────────────────────────────────────────────────────────
    with open(RESULTS_DIR/"tier2_v2_translations.csv","w",newline="",encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader(); writer.writerows(results)

    bleu_lit   = [r["bleu_vs_literal"]  for r in results]
    bleu_style = [r["bleu_vs_style"]    for r in results]
    tgt_rhyme  = [r["tgt_rhyme_density"]for r in results]

    metrics = {
        "model":                 "Tier2_Seq2Seq_v2",
        "num_poems":             len(results),
        "best_epoch":            ckpt["epoch"],
        "best_val_loss":         round(ckpt["val_loss"],4),
        "best_val_ppl":          round(math.exp(ckpt["val_loss"]),2),
        "avg_bleu_vs_literal":   round(sum(bleu_lit)  /len(bleu_lit),  2),
        "avg_bleu_vs_style":     round(sum(bleu_style)/len(bleu_style),2),
        "avg_tgt_rhyme_density": round(sum(tgt_rhyme) /len(tgt_rhyme), 3),
        "train_pairs":           len(train_pairs),
        "improvements": {
            "hidden_dim":    HIDDEN_DIM,
            "lr":            LEARNING_RATE,
            "label_smooth":  LABEL_SMOOTH,
            "data_aug":      "en_anthropic+en_mistral+en_t",
        }
    }
    with open(RESULTS_DIR/"tier2_v2_metrics.json","w") as f:
        json.dump(metrics, f, indent=2)

    print("\n── Final Metrics (v2) ───────────────────────────────────────")
    for k, v in metrics.items():
        if k != "improvements":
            print(f"  {k:<30} {v}")

    print(f"\n✅ TIER 2 v2 COMPLETE")
    print(f"   Model   → {MODELS_DIR/'best_model.pt'}")
    print(f"   Results → {RESULTS_DIR}")