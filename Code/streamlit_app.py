"""
=============================================================================
Streamlit App — Style-Preserving Hindi/Urdu → English Poetry Translation
DATS 6312 Final Project
=============================================================================
Runs a single input poem through 11 translators and reports per-model scores.

External APIs (network + key required):
    1. Google Translate         — deep-translator
    2. Anthropic Claude          — AWS Bedrock (claude-sonnet-4-5)
    3. Mistral Large             — Mistral AI API

Local models (loaded directly from disk, no shared loader):
    4. Tier 1 v1   — Models/tier1   (rule-based, pure python)
    5. Tier 1 v2   — Models/tier1_v2 (rule-based v2, pure python)
    6. Tier 2 v1   — Models/tier2_seq2seq/{best_model.pt, vocabs.pt}
    7. Tier 2 v2   — Models/tier2_seq2seq_v2/{best_model.pt, vocabs.pt}
    8. Tier 3 v1   — Models/tier3_opusmt/tier3_v1/   (HuggingFace format)
    9. Tier 3 v2   — Models/tier3_opusmt/tier3_v2/
   10. Tier 3 v3a  — Models/tier3_opusmt/tier3_v3a/
   11. Tier 3 v3b  — Models/tier3_opusmt/tier3_v3b/

PROJECT LAYOUT
    Final-Project-Group2/
    ├── Code/
    │   ├── streamlit_app.py
    │   ├── tier1_rule_based_baseline.py
    │   ├── tier1_rule_based_baseline_v2.py
    │   ├── tier2_seq2seq_lstm.py
    │   └── tier2_seq2seq_lstm_v2.py
    ├── Models/...
    └── .env

RUN
    pip install streamlit deep-translator boto3
    streamlit run streamlit_app.py
=============================================================================
"""

from __future__ import annotations

import json
import os
import re
import time
import unicodedata
import urllib.error
import urllib.request
from collections import defaultdict
from configparser import ConfigParser, ExtendedInterpolation
from pathlib import Path
from typing import Callable, Optional

import pandas as pd
import streamlit as st
import torch

# =============================================================================
# Path configuration — the user's exact layout
# =============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR   = SCRIPT_DIR.parent
MODELS_DIR = BASE_DIR / "Models"
ENV_FILE   = BASE_DIR / ".env"

# Tier 2 checkpoint locations
T2V1_CKPT  = MODELS_DIR / "tier2_seq2seq"     / "best_model.pt"
T2V1_VOCAB = MODELS_DIR / "tier2_seq2seq"     / "vocabs.pt"
T2V2_CKPT  = MODELS_DIR / "tier2_seq2seq_v2"  / "best_model.pt"
T2V2_VOCAB = MODELS_DIR / "tier2_seq2seq_v2"  / "vocabs.pt"

# Tier 3 model directories (HuggingFace from_pretrained format)
T3V1_DIR  = MODELS_DIR / "tier3_opusmt" / "tier3_v1"
T3V2_DIR  = MODELS_DIR / "tier3_opusmt" / "tier3_v2"
T3V3A_DIR = MODELS_DIR / "tier3_opusmt" / "tier3_v3a"
T3V3B_DIR = MODELS_DIR / "tier3_opusmt" / "tier3_v3b"

AWS_BEDROCK_SECTION = "240143401157_CCAS-DATS-NLP-Student"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================================================================
# .env loader — supports flat KEY=VALUE and INI-style [section] sections
# =============================================================================
def load_env_file(env_path: Path = ENV_FILE) -> dict:
    """
    Two-pass read:
      1. Push flat KEY=VALUE into os.environ (existing env vars win).
      2. Return INI sections so callers can pull AWS Bedrock creds.
    """
    sections: dict[str, dict[str, str]] = {}
    if not env_path.exists():
        return sections

    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("["):
                continue
            if "=" not in line:
                continue
            k, _, v = line.partition("=")
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            if k and v and k not in os.environ:
                os.environ[k] = v

    try:
        cfg = ConfigParser(interpolation=ExtendedInterpolation())
        cfg.read(env_path, encoding="utf-8")
        for section in cfg.sections():
            sections[section] = dict(cfg[section])
    except Exception:
        pass

    return sections


ENV_SECTIONS = load_env_file()


# =============================================================================
# Devanagari → Roman transliteration (used to romanize Anthropic prompts)
# =============================================================================
DEVANAGARI_TO_ROMAN = {
    'अ': 'a', 'आ': 'aa', 'इ': 'i', 'ई': 'ee', 'उ': 'u', 'ऊ': 'oo',
    'ए': 'e', 'ऐ': 'ai', 'ओ': 'o', 'औ': 'au',
    'क': 'k', 'ख': 'kh', 'ग': 'g', 'घ': 'gh', 'च': 'ch',
    'ज': 'j', 'झ': 'jh', 'ट': 't', 'ड': 'd', 'ढ': 'dh',
    'त': 't', 'थ': 'th', 'द': 'd', 'ध': 'dh', 'न': 'n',
    'प': 'p', 'फ': 'ph', 'ब': 'b', 'भ': 'bh', 'म': 'm',
    'य': 'y', 'र': 'r', 'ल': 'l', 'व': 'v', 'श': 'sh',
    'स': 's', 'ह': 'h', 'ं': 'n', '़': '',
    'ा': 'a', 'ि': 'i', 'ी': 'ee', 'ु': 'u', 'ू': 'oo',
    'े': 'e', 'ै': 'ai', 'ो': 'o', 'ौ': 'au', '्': '',
}


def auto_romanize(line: str) -> str:
    return "".join(DEVANAGARI_TO_ROMAN.get(ch, ch) for ch in line)


# =============================================================================
# Inline metric functions (no dependency on evaluate_all_tiers_final.py)
# =============================================================================
def compute_bleu(hypothesis: str, reference: str) -> float:
    if not hypothesis.strip() or not reference.strip():
        return 0.0
    try:
        import sacrebleu
        return round(sacrebleu.sentence_bleu(hypothesis, [reference]).score, 2)
    except ImportError:
        # Fallback: simple n-gram BLEU
        return _fallback_bleu(hypothesis, reference)


def _fallback_bleu(hyp: str, ref: str, max_n: int = 4) -> float:
    """Self-contained BLEU when sacrebleu isn't installed."""
    import math
    from collections import Counter
    hyp_t = hyp.lower().split()
    ref_t = ref.lower().split()
    if not hyp_t or not ref_t:
        return 0.0
    bp = min(1.0, math.exp(1 - len(ref_t) / len(hyp_t)))
    precisions = []
    for n in range(1, max_n + 1):
        hg = Counter(tuple(hyp_t[i:i+n]) for i in range(len(hyp_t)-n+1))
        rg = Counter(tuple(ref_t[i:i+n]) for i in range(len(ref_t)-n+1))
        clipped = sum(min(c, rg[g]) for g, c in hg.items())
        total = max(len(hyp_t) - n + 1, 0)
        precisions.append(clipped / total if total > 0 else 0.0)
    if any(p == 0 for p in precisions):
        return 0.0
    return round(bp * math.exp(sum(math.log(p) for p in precisions) / max_n) * 100, 2)


def rhyme_density(text: str) -> float:
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if not lines:
        return 0.0
    endings = []
    for l in lines:
        words = l.split()
        if words:
            endings.append(re.sub(r'[^\w]', '', words[-1].lower())[-3:])
        else:
            endings.append("")
    counts = defaultdict(int)
    for e in endings:
        counts[e] += 1
    return round(sum(1 for e in endings if counts[e] > 1) / len(endings), 3)


def rhyme_scheme(text: str) -> str:
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    endings = []
    for l in lines:
        words = l.split()
        if words:
            endings.append(re.sub(r'[^\w]', '', words[-1].lower())[-3:])
    labels: dict[str, str] = {}
    scheme = []
    counter = 0
    for end in endings:
        if end not in labels:
            labels[end] = chr(ord('A') + counter % 26)
            counter += 1
        scheme.append(labels[end])
    return "".join(scheme)


def _count_devanagari_syllables(text: str) -> int:
    count = 0
    for ch in text:
        cat  = unicodedata.category(ch)
        name = unicodedata.name(ch, "")
        if "DEVANAGARI VOWEL" in name or "DEVANAGARI LETTER" in name:
            if cat in ("Lo", "Mc", "Mn"):
                count += 1
    return max(count, 1)


def _count_english_syllables(text: str) -> int:
    text = text.lower().strip()
    if not text:
        return 1
    vowels = "aeiouy"
    count = 0
    prev = False
    for ch in text:
        is_v = ch in vowels
        if is_v and not prev:
            count += 1
        prev = is_v
    if text.endswith('e') and count > 1:
        count -= 1
    return max(count, 1)


def syllable_alignment_score(src_text: str, tgt_text: str) -> float:
    src_lines = [l.strip() for l in src_text.split("\n") if l.strip()]
    tgt_lines = [l.strip() for l in tgt_text.split("\n") if l.strip()]
    if not src_lines or not tgt_lines:
        return 0.0
    n = min(len(src_lines), len(tgt_lines))
    scores = []
    for i in range(n):
        sv = _count_devanagari_syllables(src_lines[i])
        tv = _count_english_syllables(tgt_lines[i])
        scores.append(min(sv, tv) / max(sv, tv))
    return round(sum(scores) / len(scores), 3)


# =============================================================================
# External API translators
# =============================================================================
def translate_google(poem: str) -> str:
    """Google Translate via deep-translator, line by line."""
    from deep_translator import GoogleTranslator
    translator = GoogleTranslator(source="hi", target="en")
    out = []
    for line in poem.split("\n"):
        line = line.strip()
        if not line:
            continue
        out.append(translator.translate(line))
    return "\n".join(out)


def _build_bedrock_client():
    import boto3
    section     = ENV_SECTIONS.get(AWS_BEDROCK_SECTION, {})
    aws_key     = section.get("aws_access_key_id")     or os.environ.get("AWS_ACCESS_KEY_ID")
    aws_secret  = section.get("aws_secret_access_key") or os.environ.get("AWS_SECRET_ACCESS_KEY")
    aws_session = section.get("aws_session_token")     or os.environ.get("AWS_SESSION_TOKEN")
    if not (aws_key and aws_secret):
        raise RuntimeError(
            f"AWS credentials not found in [{AWS_BEDROCK_SECTION}] of {ENV_FILE}"
        )
    sess = boto3.Session(
        aws_access_key_id     = aws_key,
        aws_secret_access_key = aws_secret,
        aws_session_token     = aws_session,
    )
    return sess.client("bedrock-runtime", region_name="us-east-1")


def translate_anthropic(poem: str) -> str:
    """Claude Sonnet 4.5 via AWS Bedrock — line-aligned poetic translation."""
    SYSTEM_PROMPT = (
        "You are an expert translator specializing in Hindi/Urdu classical poetry "
        "(ghazals, nazms). Translate poem lines into English while:\n"
        "1. Preserving the emotional depth and poetic meaning\n"
        "2. Maintaining the rhyme scheme where possible\n"
        "3. Keeping the same number of lines as the input\n"
        "4. Preserving cultural metaphors and imagery\n"
        "5. Capturing rhythm and cadence\n\n"
        "Return ONLY a valid JSON array of translated strings, one per input line."
    )
    MODEL_ID = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"

    hi_lines = [l.strip() for l in poem.split("\n") if l.strip()]
    en_lines = [auto_romanize(l) for l in hi_lines]

    user_prompt = (
        "Hindi (Devanagari):\n"
        + "\n".join(f"{i+1}. {l}" for i, l in enumerate(hi_lines))
        + "\n\nRomanization:\n"
        + "\n".join(f"{i+1}. {l}" for i, l in enumerate(en_lines))
        + f"\n\nTranslate all {len(hi_lines)} lines to English, "
          "preserving poetic style and rhyme scheme."
    )

    client = _build_bedrock_client()
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens":  1024,
        "temperature": 0,
        "system":      SYSTEM_PROMPT,
        "messages":    [{"role": "user", "content": user_prompt}],
    }
    resp = client.invoke_model(modelId=MODEL_ID, body=json.dumps(body))
    payload = json.loads(resp["body"].read().decode())
    raw = payload["content"][0]["text"].strip()

    if raw.startswith("```"):
        parts = raw.split("\n")
        raw = "\n".join(parts[1:-1] if parts[-1].strip() == "```" else parts[1:])

    translations = json.loads(raw.strip())
    while len(translations) < len(hi_lines):
        translations.append("")
    return "\n".join(translations[:len(hi_lines)])


def translate_mistral(poem: str) -> str:
    """Mistral Large via Mistral AI API."""
    api_key = os.environ.get("MISTRAL_API_KEY", "")
    if not api_key:
        raise RuntimeError("MISTRAL_API_KEY not set in .env")

    SYSTEM_PROMPT = (
        "You are a poetic translator specializing in Hindi/Urdu to English translation. "
        "Translate the given Urdu/Hindi poem lines into English while:\n"
        "1. Preserving the poetic meaning and emotional depth\n"
        "2. Maintaining the same number of lines as the input\n"
        "3. Keeping rhyme schemes and rhythm where possible\n"
        "4. Preserving cultural imagery, metaphors, and the ghazal's tone\n\n"
        "Return ONLY a JSON array of translated strings, one per input line."
    )
    hi_lines = [l.strip() for l in poem.split("\n") if l.strip()]
    payload = json.dumps({
        "model": "mistral-large-latest",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": json.dumps(hi_lines, ensure_ascii=False)},
        ],
        "temperature": 0.7,
        "max_tokens":  4000,
    }).encode("utf-8")

    req = urllib.request.Request(
        "https://api.mistral.ai/v1/chat/completions",
        data    = payload,
        headers = {
            "Content-Type":  "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method  = "POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            result = json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Mistral HTTP {e.code}: {e.read().decode()[:200]}")

    raw = result["choices"][0]["message"]["content"].strip()
    if raw.startswith("```"):
        parts = raw.split("\n")
        raw = "\n".join(parts[1:-1] if parts[-1].strip() == "```" else parts[1:])
    start, end = raw.find("["), raw.rfind("]")
    if start != -1 and end != -1:
        raw = raw[start:end+1]
    translations = json.loads(raw)
    while len(translations) < len(hi_lines):
        translations.append("")
    return "\n".join(translations[:len(hi_lines)])


# =============================================================================
# Tier 1 — import directly from the original tier scripts
# (these scripts are pure-python: dictionaries + string ops, no models)
# =============================================================================
def _safe_import(name: str):
    """Import a module by name; return (module, error_str) where exactly one is None."""
    try:
        return __import__(name), None
    except Exception as e:    # noqa: BLE001
        return None, f"{type(e).__name__}: {e}"


_t1v1, _t1v1_err = _safe_import("tier1_rule_based_baseline")
_t1v2, _t1v2_err = _safe_import("tier1_rule_based_baseline_v2")


def translate_tier1_v1(poem: str) -> str:
    """Tier 1 v1 — uses tier1_rule_based_baseline.translate_poem (returns dict)."""
    if _t1v1 is None:
        raise RuntimeError(f"tier1_rule_based_baseline not importable: {_t1v1_err}")
    result = _t1v1.translate_poem(poem)
    return result["translated_text"]


def translate_tier1_v2(poem: str) -> str:
    """Tier 1 v2 — uses tier1_rule_based_baseline_v2.translate_poem (returns string)."""
    if _t1v2 is None:
        raise RuntimeError(f"tier1_rule_based_baseline_v2 not importable: {_t1v2_err}")
    return _t1v2.translate_poem(poem)


# =============================================================================
# Tier 2 — load checkpoint from disk, reuse architecture from tier scripts
# =============================================================================
_t2v1_mod, _t2v1_err = _safe_import("tier2_seq2seq_lstm")
_t2v2_mod, _t2v2_err = _safe_import("tier2_seq2seq_lstm_v2")


@st.cache_resource(show_spinner=False)
def _load_tier2_v1():
    """Load Tier 2 v1: Models/tier2_seq2seq/{best_model.pt, vocabs.pt}."""
    if _t2v1_mod is None:
        raise RuntimeError(f"tier2_seq2seq_lstm not importable: {_t2v1_err}")
    if not (T2V1_CKPT.exists() and T2V1_VOCAB.exists()):
        return None
    ckpt  = torch.load(T2V1_CKPT,  map_location=DEVICE, weights_only=False)
    vocab = torch.load(T2V1_VOCAB, map_location=DEVICE, weights_only=False)

    hp = ckpt["hyperparams"]
    encoder = _t2v1_mod.Encoder(
        vocab_size = ckpt["src_vocab_size"],
        embed_dim  = hp["embed_dim"],
        hidden_dim = hp["hidden_dim"],
        n_layers   = hp["enc_layers"],
        dropout    = 0.0,
    ).to(DEVICE)
    decoder = _t2v1_mod.Decoder(
        vocab_size     = ckpt["tgt_vocab_size"],
        embed_dim      = hp["embed_dim"],
        hidden_dim     = hp["hidden_dim"],
        enc_hidden_dim = hp["hidden_dim"] * 2,
        n_layers       = hp["dec_layers"],
        dropout        = 0.0,
    ).to(DEVICE)
    model = _t2v1_mod.Seq2Seq(encoder, decoder, ckpt["tgt_vocab_size"], DEVICE).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    src_vocab = _t2v1_mod.Vocabulary("src")
    src_vocab.word2idx = vocab["src_word2idx"]
    src_vocab.idx2word = vocab["src_idx2word"]
    tgt_vocab = _t2v1_mod.Vocabulary("tgt")
    tgt_vocab.word2idx = vocab["tgt_word2idx"]
    tgt_vocab.idx2word = vocab["tgt_idx2word"]

    return model, src_vocab, tgt_vocab


def translate_tier2_v1(poem: str) -> str:
    bundle = _load_tier2_v1()
    if bundle is None:
        raise RuntimeError(f"Tier 2 v1 checkpoint not found at {T2V1_CKPT}")
    model, src_vocab, tgt_vocab = bundle
    # Inline equivalent of tier2_seq2seq_lstm.translate_poem (which uses module DEVICE)
    lines = [l.strip() for l in poem.split("\n") if l.strip()]
    out = []
    for line in lines:
        enc = src_vocab.encode(line)
        src_t = torch.tensor(enc, dtype=torch.long).to(DEVICE)
        tokens, _ = model.translate(src_t, len(enc), tgt_vocab)
        out.append(tgt_vocab.decode(tokens))
    return "\n".join(out)


@st.cache_resource(show_spinner=False)
def _load_tier2_v2():
    """Load Tier 2 v2: Models/tier2_seq2seq_v2/{best_model.pt, vocabs.pt}."""
    if _t2v2_mod is None:
        raise RuntimeError(f"tier2_seq2seq_lstm_v2 not importable: {_t2v2_err}")
    if not (T2V2_CKPT.exists() and T2V2_VOCAB.exists()):
        return None
    ckpt  = torch.load(T2V2_CKPT,  map_location=DEVICE, weights_only=False)
    vocab = torch.load(T2V2_VOCAB, map_location=DEVICE, weights_only=False)

    hp = ckpt["hyperparams"]
    encoder = _t2v2_mod.Encoder(
        vocab_size    = ckpt["src_vocab_size"],
        embed_dim     = hp["embed_dim"],
        hidden_dim    = hp["hidden_dim"],
        n_layers      = hp["enc_layers"],
        embed_dropout = 0.0,
        lstm_dropout  = 0.0,
    ).to(DEVICE)
    decoder = _t2v2_mod.Decoder(
        vocab_size     = ckpt["tgt_vocab_size"],
        embed_dim      = hp["embed_dim"],
        hidden_dim     = hp["hidden_dim"],
        enc_hidden_dim = hp["hidden_dim"] * 2,
        n_layers       = hp["dec_layers"],
        embed_dropout  = 0.0,
        lstm_dropout   = 0.0,
    ).to(DEVICE)
    model = _t2v2_mod.Seq2Seq(encoder, decoder, ckpt["tgt_vocab_size"], DEVICE).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    src_vocab = _t2v2_mod.Vocabulary("src")
    src_vocab.word2idx = vocab["src_word2idx"]
    src_vocab.idx2word = vocab["src_idx2word"]
    tgt_vocab = _t2v2_mod.Vocabulary("tgt")
    tgt_vocab.word2idx = vocab["tgt_word2idx"]
    tgt_vocab.idx2word = vocab["tgt_idx2word"]

    return model, src_vocab, tgt_vocab


def translate_tier2_v2(poem: str) -> str:
    bundle = _load_tier2_v2()
    if bundle is None:
        raise RuntimeError(f"Tier 2 v2 checkpoint not found at {T2V2_CKPT}")
    model, src_vocab, tgt_vocab = bundle
    lines = [l.strip() for l in poem.split("\n") if l.strip()]
    out = []
    for line in lines:
        enc = src_vocab.encode(line)
        src_t = torch.tensor(enc, dtype=torch.long).to(DEVICE)
        # v2 model.translate returns the decoded string directly
        out.append(model.translate(src_t, len(enc), tgt_vocab))
    return "\n".join(out)


# =============================================================================
# Tier 3 — load HuggingFace MarianMT checkpoints from disk
# =============================================================================
@st.cache_resource(show_spinner=False)
def _load_tier3(model_dir_str: str):
    """Load Tier 3 from a Models/tier3_opusmt/tier3_*/ directory."""
    from transformers import MarianMTModel, MarianTokenizer
    model_dir = Path(model_dir_str)
    if not model_dir.exists():
        return None
    tokenizer = MarianTokenizer.from_pretrained(str(model_dir))
    model     = MarianMTModel.from_pretrained(str(model_dir)).to(DEVICE)
    model.eval()
    return model, tokenizer


def _translate_tier3(poem: str, model_dir: Path) -> str:
    bundle = _load_tier3(str(model_dir))
    if bundle is None:
        raise RuntimeError(f"Tier 3 model directory not found at {model_dir}")
    model, tokenizer = bundle
    lines = [l.strip() for l in poem.split("\n") if l.strip()]
    out = []
    for line in lines:
        inputs = tokenizer(line, return_tensors="pt",
                           max_length=128, truncation=True).to(DEVICE)
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_length=128,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3,
            )
        out.append(tokenizer.decode(generated[0], skip_special_tokens=True))
    return "\n".join(out)


def translate_tier3_v1(poem: str)  -> str: return _translate_tier3(poem, T3V1_DIR)
def translate_tier3_v2(poem: str)  -> str: return _translate_tier3(poem, T3V2_DIR)
def translate_tier3_v3a(poem: str) -> str: return _translate_tier3(poem, T3V3A_DIR)
def translate_tier3_v3b(poem: str) -> str: return _translate_tier3(poem, T3V3B_DIR)


# =============================================================================
# Translator dispatch
# =============================================================================
DISPATCH: dict[str, Callable[[str], str]] = {
    "google":     translate_google,
    "anthropic":  translate_anthropic,
    "mistral":    translate_mistral,
    "tier1_v1":   translate_tier1_v1,
    "tier1_v2":   translate_tier1_v2,
    "tier2_v1":   translate_tier2_v1,
    "tier2_v2":   translate_tier2_v2,
    "tier3_v1":   translate_tier3_v1,
    "tier3_v2":   translate_tier3_v2,
    "tier3_v3a":  translate_tier3_v3a,
    "tier3_v3b":  translate_tier3_v3b,
}

TRANSLATORS = [
    # (short,        display,                 group,    description)
    ("google",       "Google Translate",      "api",    "deep-translator (literal baseline)"),
    ("anthropic",    "Anthropic Claude",      "api",    "Claude Sonnet 4.5 via AWS Bedrock"),
    ("mistral",      "Mistral Large",         "api",    "Mistral Large via Mistral AI API"),

    ("tier1_v1",     "Tier 1 v1",             "tier1",  "Rule-based — compact dictionary"),
    ("tier1_v2",     "Tier 1 v2",             "tier1",  "Rule-based — 950+ entries, transliteration fallback"),

    ("tier2_v1",     "Tier 2 v1",             "tier2",  "Seq2Seq LSTM, Bahdanau attention"),
    ("tier2_v2",     "Tier 2 v2",             "tier2",  "Seq2Seq LSTM, multi-target augmentation"),

    ("tier3_v1",     "Tier 3 v1",             "tier3",  "Opus-MT — IIT-Bombay → poetry fine-tune"),
    ("tier3_v2",     "Tier 3 v2",             "tier3",  "Opus-MT — improved hyperparameters"),
    ("tier3_v3a",    "Tier 3 v3a",            "tier3",  "Opus-MT — interleaved multi-target"),
    ("tier3_v3b",    "Tier 3 v3b",            "tier3",  "Opus-MT — curriculum learning"),
]

GROUP_LABELS = {
    "api":   "External APIs",
    "tier1": "Tier 1 — Rule-based",
    "tier2": "Tier 2 — Seq2Seq LSTM",
    "tier3": "Tier 3 — Opus-MT (Fine-tuned Transformer)",
}


# =============================================================================
# Scoring
# =============================================================================
def score_translation(
    source: str, hypothesis: str, reference: Optional[str] = None,
) -> dict:
    """
    Composite score for ranking:
        With reference:    0.4 × BLEU/100 + 0.3 × rhyme + 0.3 × syllable
        Without reference: 0.5 × rhyme            + 0.5 × syllable
    """
    tgt_rd = rhyme_density(hypothesis)
    syl    = syllable_alignment_score(source, hypothesis)
    src_rd = rhyme_density(source)
    src_sc = rhyme_scheme(source)
    tgt_sc = rhyme_scheme(hypothesis)

    bleu = None
    if reference and reference.strip() and reference.strip() != hypothesis.strip():
        bleu = compute_bleu(hypothesis, reference)
        composite = 0.4 * (bleu / 100.0) + 0.3 * tgt_rd + 0.3 * syl
    else:
        composite = 0.5 * tgt_rd + 0.5 * syl

    return {
        "bleu":               bleu,
        "src_rhyme_density":  src_rd,
        "tgt_rhyme_density":  tgt_rd,
        "syllable_alignment": syl,
        "src_rhyme_scheme":   src_sc,
        "tgt_rhyme_scheme":   tgt_sc,
        "composite_score":    round(composite, 3),
    }


# =============================================================================
# UI
# =============================================================================
st.set_page_config(
    page_title="Hindi/Urdu Poetry Translator",
    page_icon="🪶",
    layout="wide",
)

st.title("🪶 Style-Preserving Hindi/Urdu → English Poetry Translator")
st.caption(
    "DATS 6312 Final Project — runs an input poem through 3 external APIs "
    "and 8 local models (Tier 1/2/3), surfacing the best output by composite style score."
)

# ── Sidebar — translators + BLEU reference only ──────────────────────────────
with st.sidebar:
    st.markdown("**Active translators**")

    enabled: dict[str, bool] = {}
    for group_key, group_label in GROUP_LABELS.items():
        st.markdown(f"_{group_label}_")
        for short, display, grp, _desc in TRANSLATORS:
            if grp == group_key:
                enabled[short] = st.checkbox(display, value=True, key=f"chk_{short}")

    st.markdown("---")
    st.markdown("**BLEU reference**")
    ref_mode = st.radio(
        "Compute BLEU against:",
        options=[
            "None (style metrics only)",
            "Anthropic output (auto)",
            "Mistral output (auto)",
            "Google output (auto)",
            "Custom (paste below)",
        ],
        index=1,
        label_visibility="collapsed",
    )

    custom_reference = ""
    if ref_mode == "Custom (paste below)":
        custom_reference = st.text_area(
            "Reference English translation",
            value="",
            height=120,
            key="custom_ref",
        )

# ── Input ────────────────────────────────────────────────────────────────────
st.subheader("Input Poem (Hindi / Urdu in Devanagari)")
poem_input = st.text_area(
    "Enter the poem — one line per line:",
    value="",
    height=180,
    key="poem_input",
    label_visibility="collapsed",
    placeholder="दिल में दर्द है\nआँखों में आँसू हैं\nफिर भी मुस्कुराता हूँ",
)

translate_btn = st.button(
    "▶ Translate with all selected models",
    type="primary",
    disabled=not poem_input.strip(),
    use_container_width=True,
)

# =============================================================================
# Run
# =============================================================================
if translate_btn and poem_input.strip():

    selected = [t for t in TRANSLATORS if enabled.get(t[0])]
    if not selected:
        st.warning("Enable at least one translator in the sidebar.")
        st.stop()

    # ── Pass 1: translate ────────────────────────────────────────────────────
    progress = st.progress(0.0, text="Initializing…")
    results: list[dict] = []
    total = len(selected)

    for i, (short, display, group, desc) in enumerate(selected, start=1):
        progress.progress((i - 1) / total, text=f"Running {display}…")
        translator = DISPATCH.get(short)
        if translator is None:
            results.append({
                "short": short, "display": display, "group": group, "desc": desc,
                "ok": False, "hypothesis": None, "elapsed": 0.0,
                "error": "no dispatch entry",
            })
            continue

        try:
            t0 = time.time()
            hyp = translator(poem_input)
            results.append({
                "short": short, "display": display, "group": group, "desc": desc,
                "ok": True, "hypothesis": hyp, "elapsed": time.time() - t0,
                "error": None,
            })
        except Exception as e:    # noqa: BLE001 — surface any failure
            results.append({
                "short": short, "display": display, "group": group, "desc": desc,
                "ok": False, "hypothesis": None, "elapsed": 0.0,
                "error": f"{type(e).__name__}: {e}",
            })

    progress.progress(1.0, text="Scoring…")

    # ── Resolve BLEU reference ───────────────────────────────────────────────
    def output_of(short: str) -> Optional[str]:
        for r in results:
            if r["short"] == short and r["ok"]:
                return r["hypothesis"]
        return None

    reference: Optional[str] = None
    reference_label = "—"
    if ref_mode == "Anthropic output (auto)":
        reference = output_of("anthropic")
        reference_label = "Anthropic" if reference else "—"
    elif ref_mode == "Mistral output (auto)":
        reference = output_of("mistral")
        reference_label = "Mistral" if reference else "—"
    elif ref_mode == "Google output (auto)":
        reference = output_of("google")
        reference_label = "Google" if reference else "—"
    elif ref_mode == "Custom (paste below)" and custom_reference.strip():
        reference = custom_reference
        reference_label = "Custom"

    # ── Pass 2: score ────────────────────────────────────────────────────────
    for r in results:
        if not r["ok"]:
            r["metrics"] = None
            continue

        ref_for_this = reference
        # Don't BLEU-score a translator against itself
        if reference is not None:
            if (ref_mode == "Anthropic output (auto)" and r["short"] == "anthropic") or \
               (ref_mode == "Mistral output (auto)"   and r["short"] == "mistral")   or \
               (ref_mode == "Google output (auto)"    and r["short"] == "google"):
                ref_for_this = None

        r["metrics"] = score_translation(poem_input, r["hypothesis"], ref_for_this)

    progress.empty()

    successful = [r for r in results if r["ok"]]
    failed     = [r for r in results if not r["ok"]]

    if not successful:
        st.error("No translator produced output. See failure details below.")
        for r in failed:
            st.write(f"- **{r['display']}** — {r['error']}")
        st.stop()

    # ── Best output ──────────────────────────────────────────────────────────
    rankable = [r for r in successful
                if r["metrics"] and r["metrics"]["composite_score"] is not None]
    best = max(rankable, key=lambda r: r["metrics"]["composite_score"])

    st.markdown("---")
    st.subheader("🏆 Best translation")
    bl, br = st.columns([3, 1])
    with bl:
        st.success(f"**{best['display']}** — {best['desc']}")
        st.text(best["hypothesis"])
    with br:
        m = best["metrics"]
        st.metric("Composite",      f"{m['composite_score']:.3f}")
        st.metric("Rhyme density",  f"{m['tgt_rhyme_density']:.3f}")
        st.metric("Syllable align", f"{m['syllable_alignment']:.3f}")
        if m["bleu"] is not None:
            st.metric(f"BLEU vs {reference_label}", f"{m['bleu']:.2f}")

    if reference is not None:
        st.caption(
            f"BLEU reference: **{reference_label}** translation (selected from sidebar)."
        )

    # ── All outputs grouped by section ───────────────────────────────────────
    st.markdown("---")
    st.subheader("All translator outputs")

    by_group: dict[str, list[dict]] = {"api": [], "tier1": [], "tier2": [], "tier3": []}
    for r in results:
        by_group[r["group"]].append(r)

    def render_card(col, r):
        with col:
            badge = " ⭐" if (r["ok"] and r is best) else ""
            st.markdown(f"**{r['display']}**{badge}")
            st.caption(r["desc"])
            if not r["ok"]:
                st.warning(f"Not available — {r['error']}")
                return
            st.text_area(
                f"Output ({r['elapsed']:.1f}s)",
                value=r["hypothesis"],
                height=140,
                key=f"out_{r['short']}",
            )
            m = r["metrics"]
            mc = st.columns(4)
            mc[0].metric("Composite", f"{m['composite_score']:.3f}")
            mc[1].metric("Rhyme",     f"{m['tgt_rhyme_density']:.3f}")
            mc[2].metric("Syllable",  f"{m['syllable_alignment']:.3f}")
            mc[3].metric("BLEU",
                         "—" if m["bleu"] is None else f"{m['bleu']:.1f}")

    for group_key, label in GROUP_LABELS.items():
        items = by_group.get(group_key, [])
        if not items:
            continue
        st.markdown(f"#### {label}")
        n_cols = 3 if group_key == "api" else 2
        for row_start in range(0, len(items), n_cols):
            cols = st.columns(n_cols)
            for col, r in zip(cols, items[row_start:row_start + n_cols]):
                render_card(col, r)

    # ── Score table — sorted by composite desc, best row marked with 🏆 ──────
    st.markdown("---")
    st.subheader("All scores")

    rows = []
    for r in results:
        is_best = r["ok"] and r["metrics"] and r is best
        translator_label = f"🏆 {r['display']}" if is_best else r["display"]

        if r["ok"] and r["metrics"]:
            m = r["metrics"]
            rows.append({
                "Translator":       translator_label,
                "Group":            GROUP_LABELS[r["group"]],
                "Composite":        m["composite_score"],
                "BLEU":             "—" if m["bleu"] is None else round(m["bleu"], 2),
                "Rhyme density":    m["tgt_rhyme_density"],
                "Syllable align":   m["syllable_alignment"],
                "Src rhyme scheme": m["src_rhyme_scheme"],
                "Tgt rhyme scheme": m["tgt_rhyme_scheme"],
                "Time (s)":         round(r["elapsed"], 2),
            })
        else:
            rows.append({
                "Translator":       translator_label,
                "Group":            GROUP_LABELS[r["group"]],
                "Composite":        None,
                "BLEU":             None,
                "Rhyme density":    None,
                "Syllable align":   None,
                "Src rhyme scheme": None,
                "Tgt rhyme scheme": None,
                "Time (s)":         None,
            })

    df = pd.DataFrame(rows)
    df_sorted = df.sort_values(
        "Composite",
        ascending=False,
        na_position="last",
        kind="stable",
    ).reset_index(drop=True)

    st.dataframe(df_sorted, use_container_width=True, hide_index=True)

    csv_bytes = df_sorted.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download scores as CSV",
        data=csv_bytes,
        file_name="poem_translation_scores.csv",
        mime="text/csv",
    )

    # ── Failures ─────────────────────────────────────────────────────────────
    if failed:
        with st.expander(f"⚠️ {len(failed)} translator(s) skipped or failed"):
            for r in failed:
                st.write(f"- **{r['display']}** — {r['error']}")

else:
    st.info(
        "Enter a Hindi/Urdu poem above (one line per line) and click "
        "**Translate** to see results across all enabled translators."
    )