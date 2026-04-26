"""
=============================================================================
Tier 2 — Seq2Seq LSTM with Bahdanau Attention
DATS 6312 Final Project: Style-Preserving Hindi/Urdu to English Translation
=============================================================================
Architecture:
    Encoder: Bidirectional LSTM
    Attention: Bahdanau (additive) attention
    Decoder: Unidirectional LSTM with attention context

Training data:
    Input  : hi column (Devanagari Hindi/Urdu lines)
    Target : en_anthropic column (style-preserving English translations)

Run from: Code/ directory
    python3 tier2_seq2seq_lstm.py

Output:
    Models/tier2_seq2seq/best_model.pt
    Results/tier2/tier2_translations.csv
    Results/tier2/tier2_metrics.json
    Results/tier2/tier2_report.txt
    Plots/tier2_loss_curve.png
    Plots/tier2_attention_sample.png
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
from pathlib import Path
from collections import Counter
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent.parent
DATA_DIR    = BASE_DIR / "Data"  / "processed"
MODELS_DIR  = BASE_DIR / "Models" / "tier2_seq2seq"
RESULTS_DIR = BASE_DIR / "Results" / "tier2"
PLOTS_DIR   = BASE_DIR / "Plots"
for d in [MODELS_DIR, RESULTS_DIR, PLOTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Hyperparameters ───────────────────────────────────────────────────────────
# Sized for A10G 23.7GB — can increase if needed
EMBED_DIM    = 256
HIDDEN_DIM   = 512
ENC_LAYERS   = 2
DEC_LAYERS   = 2
DROPOUT      = 0.3
BATCH_SIZE   = 128
EPOCHS       = 30
LEARNING_RATE= 0.001
CLIP         = 1.0          # gradient clipping
TEACHER_FORCE= 0.5          # teacher forcing ratio
MIN_FREQ     = 2            # min word frequency for vocabulary
MAX_LEN      = 50           # max sequence length
RANDOM_SEED  = 42

# Special tokens
PAD_TOKEN = "<pad>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"

torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {round(torch.cuda.get_device_properties(0).total_memory/1e9,1)} GB")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — Vocabulary
# ─────────────────────────────────────────────────────────────────────────────

class Vocabulary:
    def __init__(self, name):
        self.name = name
        self.word2idx = {PAD_TOKEN: 0, SOS_TOKEN: 1, EOS_TOKEN: 2, UNK_TOKEN: 3}
        self.idx2word = {0: PAD_TOKEN, 1: SOS_TOKEN, 2: EOS_TOKEN, 3: UNK_TOKEN}
        self.word_count = Counter()

    def build(self, sentences, min_freq=MIN_FREQ):
        for sent in sentences:
            self.word_count.update(sent.split())
        for word, count in self.word_count.items():
            if count >= min_freq and word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word
        print(f"  [{self.name}] vocab size: {len(self.word2idx):,}")

    def encode(self, sentence, max_len=MAX_LEN):
        tokens = sentence.split()[:max_len]
        return (
            [self.word2idx.get(SOS_TOKEN)] +
            [self.word2idx.get(t, self.word2idx[UNK_TOKEN]) for t in tokens] +
            [self.word2idx.get(EOS_TOKEN)]
        )

    def decode(self, indices):
        words = []
        for idx in indices:
            word = self.idx2word.get(idx, UNK_TOKEN)
            if word == EOS_TOKEN:
                break
            if word not in (PAD_TOKEN, SOS_TOKEN):
                words.append(word)
        return " ".join(words)

    def __len__(self):
        return len(self.word2idx)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — Dataset
# ─────────────────────────────────────────────────────────────────────────────

class PoetryDataset(Dataset):
    def __init__(self, df, src_vocab, tgt_vocab):
        self.pairs = []
        skipped = 0
        for _, row in df.iterrows():
            src = str(row["hi"]).strip()
            tgt = str(row["en_anthropic"]).strip()
            if not src or not tgt or tgt == "nan":
                skipped += 1
                continue
            self.pairs.append((
                self.encode(src, src_vocab),
                self.encode(tgt, tgt_vocab),
            ))
        if skipped:
            print(f"  Skipped {skipped} rows with missing translations")

    def encode(self, text, vocab):
        return vocab.encode(text)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]

def collate_fn(batch):
    """Pad sequences to same length within batch."""
    src_batch, tgt_batch = zip(*batch)

    src_lens = [len(s) for s in src_batch]
    tgt_lens = [len(t) for t in tgt_batch]

    src_max = max(src_lens)
    tgt_max = max(tgt_lens)

    src_padded = [s + [0] * (src_max - len(s)) for s in src_batch]
    tgt_padded = [t + [0] * (tgt_max - len(t)) for t in tgt_batch]

    return (
        torch.tensor(src_padded, dtype=torch.long),
        torch.tensor(tgt_padded, dtype=torch.long),
        torch.tensor(src_lens,   dtype=torch.long),
    )

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — Encoder
# ─────────────────────────────────────────────────────────────────────────────

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.LSTM(
            embed_dim, hidden_dim, n_layers,
            bidirectional=True, dropout=dropout if n_layers > 1 else 0,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        # Project bidirectional hidden/cell to decoder hidden_dim
        self.fc_hidden = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc_cell   = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, src, src_lens):
        # src: [batch, src_len]
        embedded = self.dropout(self.embedding(src))

        # Pack for efficiency
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, src_lens.cpu(), batch_first=True, enforce_sorted=False
        )
        outputs, (hidden, cell) = self.rnn(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        # outputs: [batch, src_len, hidden*2]

        # Concat forward and backward final hidden states
        # hidden: [n_layers*2, batch, hidden]
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)  # [batch, hidden*2]
        cell   = torch.cat([cell[-2],   cell[-1]],   dim=1)

        hidden = torch.tanh(self.fc_hidden(hidden))  # [batch, hidden]
        cell   = torch.tanh(self.fc_cell(cell))

        # Stack to match decoder n_layers
        hidden = hidden.unsqueeze(0).repeat(DEC_LAYERS, 1, 1)
        cell   = cell.unsqueeze(0).repeat(DEC_LAYERS, 1, 1)

        return outputs, hidden, cell

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — Bahdanau Attention
# ─────────────────────────────────────────────────────────────────────────────

class BahdanauAttention(nn.Module):
    """
    Bahdanau (additive) attention.
    score(s, h) = v^T * tanh(W_s * s + W_h * h)
    """
    def __init__(self, hidden_dim, enc_hidden_dim):
        super().__init__()
        self.W_s = nn.Linear(hidden_dim,     hidden_dim, bias=False)
        self.W_h = nn.Linear(enc_hidden_dim, hidden_dim, bias=False)
        self.v   = nn.Linear(hidden_dim, 1,              bias=False)

    def forward(self, decoder_hidden, encoder_outputs):
        # decoder_hidden:  [batch, hidden]
        # encoder_outputs: [batch, src_len, enc_hidden]

        src_len = encoder_outputs.size(1)

        # Expand decoder hidden to match src_len
        s = self.W_s(decoder_hidden).unsqueeze(1).repeat(1, src_len, 1)
        # s: [batch, src_len, hidden]

        h = self.W_h(encoder_outputs)
        # h: [batch, src_len, hidden]

        energy = self.v(torch.tanh(s + h)).squeeze(2)
        # energy: [batch, src_len]

        attention = torch.softmax(energy, dim=1)
        # attention: [batch, src_len]

        context = torch.bmm(attention.unsqueeze(1), encoder_outputs).squeeze(1)
        # context: [batch, enc_hidden]

        return context, attention

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — Decoder
# ─────────────────────────────────────────────────────────────────────────────

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, enc_hidden_dim, n_layers, dropout):
        super().__init__()
        self.embedding  = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.attention  = BahdanauAttention(hidden_dim, enc_hidden_dim)
        self.rnn        = nn.LSTM(
            embed_dim + enc_hidden_dim, hidden_dim, n_layers,
            dropout=dropout if n_layers > 1 else 0,
            batch_first=True
        )
        self.fc_out     = nn.Linear(hidden_dim + enc_hidden_dim + embed_dim, vocab_size)
        self.dropout    = nn.Dropout(dropout)

    def forward_step(self, input_token, hidden, cell, encoder_outputs):
        # input_token: [batch]
        embedded = self.dropout(self.embedding(input_token.unsqueeze(1)))
        # embedded: [batch, 1, embed]

        # Use top layer hidden for attention
        context, attn_weights = self.attention(hidden[-1], encoder_outputs)
        # context: [batch, enc_hidden]

        rnn_input = torch.cat([embedded, context.unsqueeze(1)], dim=2)
        # rnn_input: [batch, 1, embed+enc_hidden]

        output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
        # output: [batch, 1, hidden]

        output  = output.squeeze(1)   # [batch, hidden]
        embedded = embedded.squeeze(1) # [batch, embed]

        pred = self.fc_out(torch.cat([output, context, embedded], dim=1))
        # pred: [batch, vocab_size]

        return pred, hidden, cell, attn_weights

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — Seq2Seq Model
# ─────────────────────────────────────────────────────────────────────────────

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, tgt_vocab_size, device):
        super().__init__()
        self.encoder       = encoder
        self.decoder       = decoder
        self.tgt_vocab_size = tgt_vocab_size
        self.device        = device

    def forward(self, src, tgt, src_lens, teacher_forcing_ratio=TEACHER_FORCE):
        batch_size = src.size(0)
        tgt_len    = tgt.size(1)

        outputs     = torch.zeros(batch_size, tgt_len, self.tgt_vocab_size).to(self.device)
        enc_outputs, hidden, cell = self.encoder(src, src_lens)

        input_token = tgt[:, 0]   # <sos>

        for t in range(1, tgt_len):
            pred, hidden, cell, _ = self.decoder.forward_step(
                input_token, hidden, cell, enc_outputs
            )
            outputs[:, t] = pred
            teacher_force  = random.random() < teacher_forcing_ratio
            input_token    = tgt[:, t] if teacher_force else pred.argmax(1)

        return outputs

    def translate(self, src_tensor, src_len, tgt_vocab, max_len=MAX_LEN):
        """Greedy decode a single sentence. Returns tokens and attention weights."""
        self.eval()
        with torch.no_grad():
            enc_outputs, hidden, cell = self.encoder(
                src_tensor.unsqueeze(0),
                torch.tensor([src_len])
            )
            input_token = torch.tensor([tgt_vocab.word2idx[SOS_TOKEN]]).to(self.device)
            tokens      = []
            attentions  = []

            for _ in range(max_len):
                pred, hidden, cell, attn = self.decoder.forward_step(
                    input_token, hidden, cell, enc_outputs
                )
                top = pred.argmax(1).item()
                if top == tgt_vocab.word2idx[EOS_TOKEN]:
                    break
                tokens.append(top)
                attentions.append(attn.squeeze(0).cpu().numpy())
                input_token = torch.tensor([top]).to(self.device)

        return tokens, attentions

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — Training
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    for src, tgt, src_lens in tqdm(loader, desc="  Train", leave=False):
        src, tgt = src.to(DEVICE), tgt.to(DEVICE)
        optimizer.zero_grad()
        output = model(src, tgt, src_lens)
        # output: [batch, tgt_len, vocab]
        # tgt:    [batch, tgt_len]
        output = output[:, 1:].reshape(-1, output.size(-1))
        tgt    = tgt[:, 1:].reshape(-1)
        loss   = criterion(output, tgt)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(loader)

def evaluate_epoch(model, loader, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for src, tgt, src_lens in tqdm(loader, desc="  Val  ", leave=False):
            src, tgt = src.to(DEVICE), tgt.to(DEVICE)
            output = model(src, tgt, src_lens, teacher_forcing_ratio=0)
            output = output[:, 1:].reshape(-1, output.size(-1))
            tgt    = tgt[:, 1:].reshape(-1)
            loss   = criterion(output, tgt)
            epoch_loss += loss.item()
    return epoch_loss / len(loader)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — BLEU (same simple impl as Tier 1)
# ─────────────────────────────────────────────────────────────────────────────

def ngrams(tokens, n):
    return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

def simple_bleu(hypothesis, reference, max_n=4):
    from collections import Counter
    hyp_tokens = hypothesis.lower().split()
    ref_tokens = reference.lower().split()
    if not hyp_tokens:
        return 0.0
    bp = min(1.0, math.exp(1 - len(ref_tokens)/len(hyp_tokens)))
    precisions = []
    for n in range(1, max_n+1):
        hyp_ng = Counter(ngrams(hyp_tokens, n))
        ref_ng = Counter(ngrams(ref_tokens, n))
        clipped = sum(min(c, ref_ng[g]) for g, c in hyp_ng.items())
        total   = max(len(hyp_tokens)-n+1, 0)
        precisions.append(clipped/total if total > 0 else 0.0)
    if any(p == 0 for p in precisions):
        return 0.0
    return round(bp * math.exp(sum(math.log(p) for p in precisions)/max_n)*100, 2)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 — Evaluation on full poems
# ─────────────────────────────────────────────────────────────────────────────

def translate_poem(model, poem_hi, src_vocab, tgt_vocab):
    """Translate a full poem line by line, return joined translation."""
    lines = [l.strip() for l in poem_hi.split("\n") if l.strip()]
    translated = []
    for line in lines:
        src_enc = src_vocab.encode(line)
        src_len = len(src_enc)
        src_tensor = torch.tensor(src_enc, dtype=torch.long).to(DEVICE)
        tokens, _ = model.translate(src_tensor, src_len, tgt_vocab)
        translated.append(tgt_vocab.decode(tokens))
    return "\n".join(translated)

def rhyme_density(text):
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if not lines:
        return 0.0
    from collections import defaultdict
    endings = [l.split()[-1][-3:] if l.split() else "" for l in lines]
    counts = defaultdict(int)
    for e in endings:
        counts[e] += 1
    rhyming = sum(1 for e in endings if counts[e] > 1)
    return round(rhyming / len(endings), 3)

def evaluate_on_poems(model, src_vocab, tgt_vocab, max_poems=100):
    poems_df = pd.read_csv(DATA_DIR / "poetry_poems_test.csv")
    results  = []
    for i, row in poems_df.head(max_poems).iterrows():
        hi       = str(row["hi"]).strip()
        ref_lit  = str(row["en_t"]).strip()
        ref_style= str(row["en_anthropic"]).strip()

        hyp = translate_poem(model, hi, src_vocab, tgt_vocab)

        results.append({
            "id":                i,
            "poet":              row["poet"],
            "poem_slug":         row["poem_slug"],
            "source":            hi,
            "ref_literal":       ref_lit,
            "ref_style":         ref_style,
            "hypothesis":        hyp,
            "bleu_vs_literal":   simple_bleu(hyp, ref_lit),
            "bleu_vs_style":     simple_bleu(hyp, ref_style),
            "src_rhyme_density": rhyme_density(hi),
            "tgt_rhyme_density": rhyme_density(hyp),
        })
    return results

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10 — Plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_loss_curve(train_losses, val_losses):
    plt.figure(figsize=(10, 4))
    plt.plot(train_losses, label="Train Loss", color="steelblue")
    plt.plot(val_losses,   label="Val Loss",   color="darkorange")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Tier 2 Seq2Seq LSTM — Training Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "tier2_loss_curve.png", dpi=150)
    plt.close()
    print(f"  Saved → {PLOTS_DIR / 'tier2_loss_curve.png'}")

def plot_attention(src_tokens, tgt_tokens, attention, sample_idx=0):
    fig, ax = plt.subplots(figsize=(10, 6))
    attn_matrix = np.array(attention[:len(tgt_tokens)])
    if attn_matrix.ndim == 1:
        attn_matrix = attn_matrix.reshape(1, -1)
    im = ax.imshow(attn_matrix, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(src_tokens)))
    ax.set_yticks(range(len(tgt_tokens)))
    ax.set_xticklabels(src_tokens, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(tgt_tokens, fontsize=8)
    ax.set_xlabel("Source (Hindi)")
    ax.set_ylabel("Target (English)")
    ax.set_title("Bahdanau Attention Weights — Sample Translation")
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "tier2_attention_sample.png", dpi=150)
    plt.close()
    print(f"  Saved → {PLOTS_DIR / 'tier2_attention_sample.png'}")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("TIER 2 — SEQ2SEQ LSTM WITH BAHDANAU ATTENTION")
    print("=" * 60)

    # ── Load data ─────────────────────────────────────────────────────────────
    print("\nLoading data...")
    train_df = pd.read_csv(DATA_DIR / "poetry_train.csv")
    val_df   = pd.read_csv(DATA_DIR / "poetry_val.csv")

    # Drop rows with missing en_anthropic
    train_df = train_df.dropna(subset=["hi", "en_anthropic"])
    val_df   = val_df.dropna(subset=["hi", "en_anthropic"])
    train_df = train_df[train_df["en_anthropic"].str.strip() != ""]
    val_df   = val_df[val_df["en_anthropic"].str.strip() != ""]

    print(f"  Train: {len(train_df):,} rows")
    print(f"  Val:   {len(val_df):,} rows")

    # ── Build vocabularies ────────────────────────────────────────────────────
    print("\nBuilding vocabularies...")
    src_vocab = Vocabulary("hindi")
    tgt_vocab = Vocabulary("english")
    src_vocab.build(train_df["hi"].tolist())
    tgt_vocab.build(train_df["en_anthropic"].tolist())

    # Save vocabs for inference
    torch.save({
        "src_word2idx": src_vocab.word2idx,
        "src_idx2word": src_vocab.idx2word,
        "tgt_word2idx": tgt_vocab.word2idx,
        "tgt_idx2word": tgt_vocab.idx2word,
    }, MODELS_DIR / "vocabs.pt")
    print(f"  Saved vocabs → {MODELS_DIR / 'vocabs.pt'}")

    # ── Datasets & loaders ────────────────────────────────────────────────────
    print("\nBuilding datasets...")
    train_dataset = PoetryDataset(train_df, src_vocab, tgt_vocab)
    val_dataset   = PoetryDataset(val_df,   src_vocab, tgt_vocab)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate_fn, num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn, num_workers=2, pin_memory=True
    )

    # ── Build model ───────────────────────────────────────────────────────────
    print("\nBuilding model...")
    encoder = Encoder(
        vocab_size=len(src_vocab),
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        n_layers=ENC_LAYERS,
        dropout=DROPOUT
    ).to(DEVICE)

    decoder = Decoder(
        vocab_size=len(tgt_vocab),
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        enc_hidden_dim=HIDDEN_DIM * 2,   # bidirectional encoder
        n_layers=DEC_LAYERS,
        dropout=DROPOUT
    ).to(DEVICE)

    model = Seq2Seq(encoder, decoder, len(tgt_vocab), DEVICE).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {total_params:,}")

    # ── Optimizer & loss ──────────────────────────────────────────────────────
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # ignore PAD

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"\nTraining for {EPOCHS} epochs...")
    best_val_loss = float("inf")
    train_losses, val_losses = [], []
    no_improve = 0
    PATIENCE = 7   # early stopping

    for epoch in range(1, EPOCHS + 1):
        start = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, criterion, CLIP)
        val_loss   = evaluate_epoch(model, val_loader, criterion)
        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        elapsed = time.time() - start
        print(f"  Epoch {epoch:02d}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"PPL: {math.exp(val_loss):.2f} | "
              f"Time: {elapsed:.1f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch":      epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_loss":   val_loss,
                "src_vocab_size": len(src_vocab),
                "tgt_vocab_size": len(tgt_vocab),
                "hyperparams": {
                    "embed_dim": EMBED_DIM, "hidden_dim": HIDDEN_DIM,
                    "enc_layers": ENC_LAYERS, "dec_layers": DEC_LAYERS,
                }
            }, MODELS_DIR / "best_model.pt")
            print(f"    ✓ Saved best model (val_loss={val_loss:.4f})")
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"\n  Early stopping at epoch {epoch} (no improvement for {PATIENCE} epochs)")
                break

    # ── Plot loss curve ───────────────────────────────────────────────────────
    print("\nPlotting loss curve...")
    plot_loss_curve(train_losses, val_losses)

    # ── Load best model for evaluation ───────────────────────────────────────
    print("\nLoading best model for evaluation...")
    checkpoint = torch.load(MODELS_DIR / "best_model.pt", map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    print(f"  Best model from epoch {checkpoint['epoch']} (val_loss={checkpoint['val_loss']:.4f})")

    # ── Evaluate on full poems ────────────────────────────────────────────────
    print("\nEvaluating on full poems...")
    results = evaluate_on_poems(model, src_vocab, tgt_vocab, max_poems=100)

    # ── Print sample translations ─────────────────────────────────────────────
    print("\n── Sample Translations ──────────────────────────────────────")
    for r in results[:2]:
        print(f"\nSOURCE:\n{r['source'][:200]}")
        print(f"REF STYLE:\n{r['ref_style'][:200]}")
        print(f"HYPOTHESIS:\n{r['hypothesis'][:200]}")
        print(f"BLEU literal: {r['bleu_vs_literal']}  BLEU style: {r['bleu_vs_style']}")

    # ── Attention plot on one sample ──────────────────────────────────────────
    print("\nGenerating attention plot...")
    sample_line = train_df["hi"].iloc[0]
    src_enc     = src_vocab.encode(sample_line)
    src_tensor  = torch.tensor(src_enc, dtype=torch.long).to(DEVICE)
    tokens, attentions = model.translate(src_tensor, len(src_enc), tgt_vocab)
    if attentions:
        src_tokens = sample_line.split()
        tgt_tokens = [tgt_vocab.idx2word.get(t, UNK_TOKEN) for t in tokens]
        plot_attention(src_tokens, tgt_tokens, attentions)

    # ── Save results ──────────────────────────────────────────────────────────
    print("\nSaving results...")
    csv_path = RESULTS_DIR / "tier2_translations.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"  Saved → {csv_path}")

    bleu_lit   = [r["bleu_vs_literal"] for r in results]
    bleu_style = [r["bleu_vs_style"]   for r in results]
    src_rhyme  = [r["src_rhyme_density"] for r in results]
    tgt_rhyme  = [r["tgt_rhyme_density"] for r in results]

    metrics = {
        "model":                   "Tier2_Seq2Seq_LSTM_Bahdanau",
        "num_poems":               len(results),
        "best_epoch":              checkpoint["epoch"],
        "best_val_loss":           round(checkpoint["val_loss"], 4),
        "best_val_ppl":            round(math.exp(checkpoint["val_loss"]), 2),
        "avg_bleu_vs_literal":     round(sum(bleu_lit)  /len(bleu_lit),   2),
        "avg_bleu_vs_style":       round(sum(bleu_style)/len(bleu_style), 2),
        "avg_src_rhyme_density":   round(sum(src_rhyme) /len(src_rhyme),  3),
        "avg_tgt_rhyme_density":   round(sum(tgt_rhyme) /len(tgt_rhyme),  3),
        "hyperparams": {
            "embed_dim": EMBED_DIM, "hidden_dim": HIDDEN_DIM,
            "enc_layers": ENC_LAYERS, "dec_layers": DEC_LAYERS,
            "batch_size": BATCH_SIZE, "dropout": DROPOUT,
            "learning_rate": LEARNING_RATE,
        }
    }

    json_path = RESULTS_DIR / "tier2_metrics.json"
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved → {json_path}")

    # Report
    report_path = RESULTS_DIR / "tier2_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("TIER 2 — SEQ2SEQ LSTM WITH BAHDANAU ATTENTION — REPORT\n")
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
            f.write(f"BLEU literal: {r['bleu_vs_literal']}  BLEU style: {r['bleu_vs_style']}\n")
            f.write("-"*70 + "\n\n")
    print(f"  Saved → {report_path}")

    print("\n── Final Metrics ────────────────────────────────────────────")
    for k, v in metrics.items():
        if k != "hyperparams":
            print(f"  {k:<30} {v}")

    print("\n✅ TIER 2 COMPLETE")
    print(f"   Model saved  → {MODELS_DIR / 'best_model.pt'}")
    print(f"   Results saved → {RESULTS_DIR}")
    print(f"   Plots saved   → {PLOTS_DIR}")