# Final-Project-Group2

# Style-Preserving Hindi/Urdu → English Poetic Translation

**DATS 6312 — Natural Language Processing — Spring 2026**
**Author:** Adarsh Singh
**Instructor:** Dr. Amir Jafari
**The George Washington University**

For my NLP class project I'm thinking of building a model using NLP's topics which translates poems, songs from one language(specifically Hindi/Urdu) to English while keeping the poetic rhythm and style intact.

A multi-tier neural machine translation system that translates Hindi/Urdu classical poetry (ghazals, nazms, dohas) into English while preserving poetic style — rhythm, end-rhyme, and cultural metaphor — beyond what literal tools like Google Translate can produce.

The system progresses through three tiers of increasing sophistication:

1. **Tier 1 — Rule-Based Dictionary Baseline.** Word-by-word lookup against an 8.5K-entry Urdu/Hindi–English lexicon, with ITRANS romanization fallback for OOV tokens.
2. **Tier 2 — Seq2Seq LSTM with Bahdanau Attention.** From-scratch PyTorch implementation, 2-layer bidirectional encoder + 2-layer decoder, trained on 14.6K (v1) or 43.7K (v2) parallel poem-line pairs.
3. **Tier 3 — Helsinki-NLP Opus-MT (MarianMT) Fine-Tuned.** Pretrained encoder-decoder Transformer fine-tuned on the multi-reference poetry corpus, with four version variants (v1 two-phase, v2 simplified, v3a interleaved styles, v3b curriculum learning).

A total of **8 model checkpoints** are trained, evaluated on a common 100-poem test set, and exposed through a Streamlit demo that compares them side-by-side with three external API translators (Google Translate, Anthropic Claude, Mistral Large).

---

## Headline Results

Final cross-version comparison from `evaluate_all_tiers_final.py`, evaluated on the same 100 held-out test poems:

| Metric | T1 v1 | T1 v2 | T2 v1 | T2 v2 | T3 v1 | T3 v2 | T3 v3a | T3 v3b |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BLEU vs Literal | 0.35 | 0.92 | 2.19 | 2.84 | **9.05** | 8.94 | 7.90 | 7.77 |
| BLEU vs Style | 0.29 | 0.73 | 3.52 | 2.12 | 6.69 | 6.84 | **7.07** | 6.69 |
| Rhyme Density (tgt) | 0.600 | 0.616 | **0.701** | 0.651 | 0.270 | 0.252 | 0.311 | 0.284 |
| Syllable Alignment | 0.144 | 0.469 | 0.388 | 0.327 | 0.503 | **0.520** | 0.494 | 0.455 |

- **Best BLEU vs Literal:** Tier 3 v1 (9.05)
- **Best BLEU vs Style:** Tier 3 v3a (7.07)
- **Best Rhyme Density:** Tier 2 v1 (0.701) — partly an artifact of repetitive line endings
- **Best Syllable Alignment:** Tier 3 v2 (0.520)

No single model wins on every axis — the right model depends on whether the user prioritizes literal fidelity, stylistic English, surface rhyme, or rhythmic length-matching.

---

## Repository Structure

```
Final-Project-Group2/
├── Code/                              # All Python scripts (see Pipeline section)
│   ├── translate_poems_google.py      # Step 2 — generate en_t (literal)
│   ├── translate_poems_anthropic.py   # Step 3 — generate en_anthropic (style)
│   ├── translate_poems_all.py         # Step 4 — multi-LLM (only en_mistral worked)
│   ├── test_indictrans2_ghazals.py    # Step 5 — IndicTrans2 (skipped path)
│   ├── iitb_data_prep.py              # Step 7 — prep IIT Bombay corpus
│   ├── poetry_json_to_csv.py          # Step 8 — JSON → train/val/test CSV
│   ├── tier1_rule_based_baseline.py   # Step 9  — Tier 1 v1
│   ├── tier1_rule_based_baseline_v2.py# Step 13 — Tier 1 v2
│   ├── tier2_seq2seq_lstm.py          # Step 10 — Tier 2 v1
│   ├── tier2_seq2seq_lstm_v2.py       # Step 14 — Tier 2 v2
│   ├── tier3_opus_mt.py               # Step 11 — Tier 3 v1
│   ├── tier3_opus_mt_v2.py            # Step 15 — Tier 3 v2
│   ├── tier3_opus_mt_v3a.py           # Step 17 — Tier 3 v3a
│   ├── tier3_opus_mt_v3b.py           # Step 18 — Tier 3 v3b
│   ├── evaluate_all_tiers.py          # Step 12 — evaluate v1 set
│   ├── evaluate_all_tiers_v2.py       # Step 16 — evaluate v2 set
│   ├── evaluate_all_tiers_final.py    # Step 19 — final 8-model comparison
│   └── streamlit_app.py               # Step 20 — interactive demo
├── Data/
│   ├── raw/                           # Kaggle Urdu Ghazal Dataset (gitignored)
│   └── processed/                     # CSVs and processed JSONs (gitignored)
│       ├── iitb_train_filtered.csv
│       ├── poetry_train.csv
│       ├── poetry_val.csv
│       ├── poetry_test.csv
│       ├── poetry_poems_test.csv
│       └── iitb_eda_report.txt
├── Models/                            # Trained checkpoints (gitignored)
│   ├── tier1_rule_based/
│   ├── tier2_seq2seq/{v1,v2}/
│   └── tier3_opusmt/
│       ├── tier3_v1/
│       ├── tier3_v2/
│       ├── tier3_v3a/
│       └── tier3_v3b/
├── Results/
│   ├── outputs/                       # Translation JSONs (raw LLM outputs)
│   ├── tier1/, tier2/, tier3/         # Per-tier metrics + sample translations
│   ├── evaluation/                    # v1 cross-tier comparison
│   ├── evaluation_v2/                 # v2 cross-tier comparison
│   └── evaluation_final/              # final 8-model comparison
├── Plots/                             # Loss curves, comparison bar charts
├── Individual-Final-Project-Report/
│   └── Adarsh-Final-Report.pdf
├── Final-Group-Project-Report/
│   └── Group-Final-Report.pdf
├── Final-Group-Presentation/
│   └── Group-Presentation.pdf
├── requirements.txt
├── .gitignore
└── README.md
```

Large binaries (model weights, raw datasets, generated CSVs and PNGs) are excluded by `.gitignore`. They are regenerated by re-running the pipeline below.

---

## Setup

### 1. Clone and create environment

```bash
git clone https://github.com/drsh0755/Final-Project-Group2.git
cd Final-Project-Group2
python3 -m venv nlp_env
source nlp_env/bin/activate
pip install -r requirements.txt
```

### 2. Set credentials (only needed for the data-generation steps 1–4)

Create a `.env` file in the repo root:

```
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_kaggle_api_key
ANTHROPIC_BEDROCK_REGION=us-east-1
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
MISTRAL_API_KEY=...
```

The Kaggle key is needed for Step 1 (dataset download). The AWS Bedrock credentials are needed for Step 3 (`en_anthropic`). The Mistral key is needed for Step 4 (`en_mistral`).

### 3. GPU

All training was done on an AWS EC2 `g5.xlarge` instance with one NVIDIA A10G (24 GB VRAM), Ubuntu 22.04, CUDA 12, PyTorch 2.x. Tier 1 runs on CPU. Tier 2 needs a GPU but fits on most consumer cards. Tier 3 fine-tuning uses mixed-precision (`fp16`) and runs comfortably in 24 GB.

---

## Pipeline (run order)

The full pipeline is 20 steps. Steps 1–8 build the dataset; steps 9–19 train and evaluate the model versions; step 20 launches the demo. Run from the `Code/` directory unless noted.

### Data acquisition

```bash
# Step 1 — download the Kaggle Urdu Ghazal Dataset (32 poets, en/hi/ur folders)
kaggle datasets download mabubakrsiddiq/urdu-ghazal-dataset-32-poets-and-their-ghazals
unzip urdu-ghazal-dataset-32-poets-and-their-ghazals.zip -d ../Data/raw/

# Step 2 — generate en_t (literal reference, deep-translator/Google)
python translate_poems_google.py --test
python translate_poems_google.py --all

# Step 3 — generate en_anthropic (style-preserving reference, Claude Sonnet 4.5)
python translate_poems_anthropic.py --test
python translate_poems_anthropic.py --all

# Step 4 — generate en_mistral (style-preserving reference, Mistral Large)
python3 translate_poems_all.py --test --only en_mistral
python3 translate_poems_all.py --all  --only en_mistral

# Step 5 — IndicTrans2 (skipped; scaffold left in for completeness)
# python test_indictrans2_ghazals.py
```

Steps 4's other targets (`en_groq`, `en_gemini`, `en_meta`, `en_grok`) were attempted but dropped because the corresponding APIs would not authenticate cleanly within the time window or hit free-tier daily limits.

### Data preparation

```bash
# Step 6 — sync translation JSONs into Data/processed/
cp ../Results/outputs/*.json ../Data/processed/

# Step 7 — download and prepare the IIT Bombay Hindi-English parallel corpus
python iitb_data_prep.py

# Step 8 — convert per-poem JSON to long-format line-level CSV with stratified split
python poetry_json_to_csv.py
```

The split is stratified by poet at the poem level (no two lines from the same poem land in different splits): 14,591 train / 2,898 val / 3,380 test line pairs.

### Train and evaluate v1 set

```bash
python3 tier1_rule_based_baseline.py    # Step 9  — T1 v1
python3 tier2_seq2seq_lstm.py           # Step 10 — T2 v1
python3 tier3_opus_mt.py                # Step 11 — T3 v1 (two-phase)
python3 evaluate_all_tiers.py           # Step 12 — v1 comparison
```

### Train and evaluate v2 set

```bash
python3 tier1_rule_based_baseline_v2.py # Step 13 — T1 v2 (expanded lexicon + ITRANS fallback)
python3 tier2_seq2seq_lstm_v2.py        # Step 14 — T2 v2 (multi-target augmentation)
python3 tier3_opus_mt_v2.py             # Step 15 — T3 v2 (Phase 1 dropped, lower LR)
python3 evaluate_all_tiers_v2.py        # Step 16 — v2 comparison
```

### Tier 3 v3 variants and final cross-version evaluation

```bash
python3 tier3_opus_mt_v3a.py            # Step 17 — T3 v3a (interleaved en_anthropic / en_mistral)
python3 tier3_opus_mt_v3b.py            # Step 18 — T3 v3b (curriculum learning)
python3 evaluate_all_tiers_final.py     # Step 19 — all 8 checkpoints side-by-side
```

### Streamlit demo

```bash
streamlit run streamlit_app.py          # Step 20
```

The app loads all 8 local checkpoints plus 3 external API translators and runs an input poem through 11 translators side-by-side with per-model BLEU, rhyme density, and syllable alignment scores.

---

## Dataset

| Component | Source | Size | Role |
|---|---|---|---|
| Hindi/Urdu poetry source | Kaggle: `mabubakrsiddiq/urdu-ghazal-dataset-32-poets-and-their-ghazals` | 32 poets | Devanagari source text |
| `en_t` literal reference | deep-translator (Google Translate) | All poems | Used for BLEU vs Literal |
| `en_anthropic` style reference | Anthropic Claude Sonnet 4.5 (Bedrock) | All poems | Primary training target |
| `en_mistral` style reference | Mistral Large | All poems | Secondary target (Tier 2 v2 augmentation) |
| IIT Bombay parallel corpus | HuggingFace `cfilt/iitb-english-hindi` | 50K filtered pairs | Tier 3 v1 Phase 1 only |

The original plan included scraping Rekhta.org and Kavitakosh.org for richer professionally-translated material, but neither site exposed structured plaintext usable in bulk, so the Kaggle dataset became the spine of the project.

---

## Evaluation Metrics

Two families of metrics are reported:

**Standard MT metrics**
- **BLEU vs Literal** — corpus BLEU against `en_t` (deep-translator output).
- **BLEU vs Style** — corpus BLEU against `en_anthropic` (Claude style-preserving reference).

**Custom style metrics**
- **Rhyme Density Score** — fraction of output lines whose final-word last-3-character ending matches at least one other line in the same output.
- **Syllable Alignment Score** — `1 − mean(|σ_src − σ_tgt| / max(σ_src, σ_tgt))` per line, where source syllables are counted on Devanagari (vowels + matras) and target syllables on English (vowel groups). Higher = better rhythmic length-matching.

Full mathematical formulations are in §1.3 of the report.

---

## Key Implementation Details

- **Bahdanau additive attention** (Tier 2): `e_{t,j} = vᵀ tanh(W_a s_{t-1} + U_a h_j)`, `α = softmax(e)`, `c_t = Σ α · h`.
- **Variable-length batching**: PyTorch `pack_padded_sequence` with sort-then-restore.
- **Teacher forcing**: probability 0.5 during Tier 2 training.
- **Mixed precision**: `fp16` enabled for all Tier 3 training, halving VRAM.
- **Beam search** (Tier 3): 4 beams, `no_repeat_ngram_size=2` for v2+, `length_penalty=0.8`.
- **Early stopping**: patience 10 (Tier 2), patience 5 on validation BLEU (Tier 3).
- **Multi-target augmentation** (Tier 2 v2): each `hi` line paired with `en_anthropic`, `en_mistral`, and `en_t` as three separate training rows, tripling the effective train set to 43,673 pairs.

---


---

## References

1. CFILT. `cfilt/iitb-english-hindi`. <https://huggingface.co/datasets/cfilt/iitb-english-hindi>
2. CFILT. `cfilt/HiNER-original`. <https://huggingface.co/datasets/cfilt/HiNER-original>
3. M. A. B. Siddiqui. Urdu Ghazal Dataset — 32 Poets and Their Ghazals. <https://www.kaggle.com/datasets/mabubakrsiddiq/urdu-ghazal-dataset-32-poets-and-their-ghazals>
4. AI4Bharat. IndicTrans2. <https://github.com/AI4Bharat/IndicTrans2>
5. AI4Bharat. `ai4bharat/indictrans2-indic-en-1B`. <https://huggingface.co/ai4bharat/indictrans2-indic-en-1B>
6. Google Research. Dakshina Dataset. <https://github.com/google-research-datasets/dakshina>
7. PyTorch. NLP From Scratch: Translation with a Sequence to Sequence Network and Attention. <https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html>
8. A. Kunchukuttan. IndicNLP Library. <https://github.com/anoopkunchukuttan/indic_nlp_library>
9. Streamlit Documentation. <https://docs.streamlit.io/>

---

## Generative AI Acknowledgement

Anthropic Claude (`claude-sonnet-4-5` via AWS Bedrock) was used to generate the `en_anthropic` style-preserving reference translations used as the primary training target, to iterate on data preparation, training, and evaluation scripts, and to draft and refine sections of the report. Mistral Large was used to generate the `en_mistral` reference translations. All architectural decisions, hyperparameter choices, debugging, and final code/text were reviewed and approved.