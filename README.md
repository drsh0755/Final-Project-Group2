# Hindi/Urdu → English Poetry Translation System

> NLP class project — Final Project Group 2

Translates poems and songs from **Hindi (Devanagari)** and **Urdu (Nastaliq)**
into English while preserving **poetic rhythm**, **rhyme scheme**, and **thematic style**.

---

## Features

| Feature | Description |
|---------|-------------|
| **Neural Translation** | Line-by-line translation using Helsinki-NLP's MarianMT (`opus-mt-hi-en` / `opus-mt-ur-en`) |
| **Topic Modelling** | LDA-based theme extraction to identify dominant motifs in a poem |
| **Style Analysis** | Rhyme-scheme detection (CMU Pronouncing Dictionary) + per-line syllable counting |
| **Style Preservation** | WordNet synonym substitution to restore rhyme in translated lines |
| **Side-by-Side View** | Original and translated lines displayed in aligned columns |
| **Language Detection** | Unicode script detection (Devanagari vs Arabic script) with `langdetect` fallback |

---

## Project Structure

```
Final-Project-Group2/
├── app.py                        # CLI entry point
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── poetry_utils.py           # Language detection, text cleaning, formatting
│   ├── translator.py             # MarianMT translation (Hindi/Urdu → English)
│   ├── style_analyzer.py         # Rhyme, syllable, style post-processing
│   └── topic_modeling.py         # LDA topic modelling
├── data/
│   ├── sample_poems_hindi.txt    # Sample Hindi poems (Kabir, Mirabai, Ghalib)
│   └── sample_poems_urdu.txt     # Sample Urdu poems (Ghalib, Mir, Faiz)
└── tests/
    ├── test_poetry_utils.py
    ├── test_style_analyzer.py
    ├── test_topic_modeling.py
    └── test_translator.py
```

---

## Installation

```bash
pip install -r requirements.txt
```

> **Note:** `torch` and `transformers` will download the MarianMT models
> (~300 MB each) on first use.  An internet connection is required for the
> initial download; subsequent runs use the local cache.

---

## Usage

### Translate a poem file

```bash
# Hindi poem
python app.py --file data/sample_poems_hindi.txt --lang hi

# Urdu poem with topic modelling and style report
python app.py --file data/sample_poems_urdu.txt --lang ur --topics --style
```

### Translate inline text

```bash
python app.py --text "माटी कहे कुम्हार से, तू क्या रौंदे मोय।" --lang hi --style
```

### All options

```
usage: app.py [-h] (--file PATH | --text TEXT) [--lang {hi,ur}]
              [--topics] [--style] [--verbose]

  --file PATH    Path to a text file containing the poem
  --text TEXT    Inline poem text (quoted string)
  --lang         Source language: 'hi' (Hindi) or 'ur' (Urdu)  [default: hi]
  --topics       Run LDA topic modelling on the source poem
  --style        Analyse and report poetic style metrics
  --verbose      Print progress information
```

---

## NLP Pipeline

```
Input poem (Hindi/Urdu)
        │
        ▼
1. Language Detection    ← Unicode script analysis + langdetect
        │
        ▼
2. Topic Modelling       ← LDA (gensim) on source text
        │
        ▼
3. Neural Translation    ← Helsinki-NLP MarianMT (line-by-line)
        │
        ▼
4. Style Analysis        ← CMU Pronouncing Dictionary rhyme detection
        │                   + syllable counting
        ▼
5. Style Post-processing ← WordNet synonym substitution for rhyme restoration
        │
        ▼
Output: Side-by-side translation + Style Report
```

---

## Running Tests

```bash
pytest tests/ -v
```

All tests are dependency-light: the transformer models and `torch` are mocked
so the suite runs without a GPU or internet connection.

---

## Dependencies

| Package | Purpose | Min version |
|---------|---------|-------------|
| `transformers` | MarianMT translation models | ≥ 4.48.0 |
| `torch` | PyTorch backend for inference | ≥ 2.6.0 |
| `sentencepiece` | Tokeniser for MarianMT | ≥ 0.2.1 |
| `gensim` | LDA topic modelling | ≥ 4.3.2 |
| `nltk` | Stopwords, WordNet synonyms | ≥ 3.9.3 |
| `pronouncing` | CMU rhyme/syllable analysis | ≥ 0.2.0 |
| `langdetect` | Fallback language detection | ≥ 1.0.9 |
| `sacrebleu` | BLEU score evaluation | ≥ 2.4.0 |

---

## Example Output

```
=== Discovered Themes ===
Topic 0 (42%): माटी, कहे, कुम्हार, ज्ञान, साधु
Topic 1 (35%): राम, रतन, धन, प्रभु, भक्ति
Topic 2 (23%): खजूर, पेड़, छाया, पंथी, दूर

=== Translation ===
Original                                      | Translation
------------------------------------------------------------------------------------------
माटी कहे कुम्हार से, तू क्या रौंदे मोय।    | The clay says to the potter, why do you knead me?
एक दिन ऐसा आएगा, मैं रौंदूँगी तोय॥         | One day such will come, I will knead you.

=== Style Analysis Report ===
Original rhyme scheme   : AB
Translated rhyme scheme : AB
Rhyme preserved         : No
Syllable similarity     : 68%
```
