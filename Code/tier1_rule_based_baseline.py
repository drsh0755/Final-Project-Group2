"""
=============================================================================
Tier 1 — Rule-Based Baseline Translation
DATS 6312 Final Project: Style-Preserving Hindi/Urdu to English Translation
=============================================================================
What this does:
    Word-by-word dictionary lookup (Hindi → English) using a curated
    Hindi/Urdu lexicon. Applies basic reordering rules to handle SOV→SVO
    word order. Also extracts rhyme scheme from source poem for analysis.

Why it exists:
    Lower-bound baseline. Demonstrates why naive translation destroys poetic
    structure — motivates the Seq2Seq and transformer models in the report.

Input:
    ../Data/processed/poetry_poems_test.csv   (primary — poetry test set)
    ../Data/processed/iitb_test_filtered.csv  (fallback — IIT Bombay)

Output:
    ../Results/tier1/tier1_translations.csv   — side-by-side results
    ../Results/tier1/tier1_metrics.json       — BLEU + style metrics per poem
    ../Results/tier1/tier1_report.txt         — human-readable summary

Run from: Code/ directory
Run time: < 1 minute, no GPU needed
=============================================================================
"""

import re
import json
import unicodedata
from pathlib import Path
from collections import defaultdict

# ── Paths (run from Code/ directory) ─────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent.parent   # Final-Project-Group2/
DATA_DIR    = BASE_DIR / "Data"  / "processed"
RESULTS_DIR = BASE_DIR / "Results" / "tier1"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — Hindi / Urdu → English Dictionary
# Covers common poetic vocabulary: emotions, nature, love, time, body, action
# ─────────────────────────────────────────────────────────────────────────────

HINDI_DICT = {
    # Pronouns
    "मैं": "I", "मुझे": "me", "मुझको": "me", "मेरा": "my", "मेरी": "my",
    "मेरे": "my", "हम": "we", "हमारा": "our", "हमारी": "our",
    "तू": "you", "तुम": "you", "तुझे": "you", "तेरा": "your", "तेरी": "your",
    "आप": "you", "आपका": "your", "वह": "he/she", "वो": "he/she",
    "उसका": "his/her", "उसकी": "his/her", "उन्हें": "them",
    "यह": "this", "ये": "these", "वे": "they", "कोई": "someone",
    "कुछ": "something", "सब": "all", "हर": "every",

    # Verbs (common forms)
    "है": "is", "हैं": "are", "था": "was", "थी": "was", "थे": "were",
    "हो": "be", "होना": "to be", "होता": "becomes", "होती": "becomes",
    "हूँ": "am", "हुआ": "became", "हुई": "became",
    "जाना": "to go", "जाता": "goes", "जाती": "goes", "जा": "go",
    "आना": "to come", "आता": "comes", "आती": "comes", "आ": "come",
    "करना": "to do", "करता": "does", "करती": "does", "कर": "do",
    "देना": "to give", "देता": "gives", "दे": "give", "दिया": "gave",
    "लेना": "to take", "लेता": "takes", "ले": "take",
    "कहना": "to say", "कहता": "says", "कहती": "says", "कहा": "said",
    "देखना": "to see", "देखता": "sees", "देख": "see", "देखा": "saw",
    "सुनना": "to hear", "सुनता": "hears", "सुन": "hear",
    "चलना": "to walk", "चलता": "walks", "चल": "walk",
    "रहना": "to stay", "रहता": "stays", "रही": "staying", "रहे": "staying",
    "मिलना": "to meet", "मिलता": "meets", "मिले": "met",
    "जीना": "to live", "जीता": "lives", "जी": "live",
    "मरना": "to die", "मरता": "dies", "मर": "die",
    "खोना": "to lose", "खो": "lose", "खोया": "lost",
    "पाना": "to find", "पाता": "finds", "पाई": "found",
    "भूलना": "to forget", "भूल": "forget", "भूला": "forgot",
    "याद": "memory/remember", "याद करना": "to remember",
    "समझना": "to understand", "समझ": "understand",
    "जानना": "to know", "जानता": "knows", "जाना": "to know",
    "लगना": "to seem", "लगता": "seems", "लगती": "seems",
    "उठना": "to rise", "उठता": "rises", "उठ": "rise",
    "बोलना": "to speak", "बोलता": "speaks", "बोल": "speak",
    "रोना": "to cry", "रोता": "cries", "रो": "cry",
    "हँसना": "to laugh", "हँसता": "laughs", "हँस": "laugh",

    # Nouns — emotions
    "दिल": "heart", "दिलों": "hearts",
    "प्यार": "love", "मुहब्बत": "love", "इश्क": "love",
    "प्रेम": "love", "प्रीत": "love",
    "दर्द": "pain", "तकलीफ": "suffering", "गम": "sorrow",
    "ख़ुशी": "happiness", "खुशी": "happiness",
    "आँसू": "tears", "आंसू": "tears",
    "उम्मीद": "hope", "आशा": "hope",
    "ख़्वाब": "dream", "ख्वाब": "dream", "सपना": "dream",
    "डर": "fear", "खौफ": "fear",
    "गुस्सा": "anger", "क्रोध": "anger",
    "शर्म": "shame", "हया": "modesty",
    "अकेला": "lonely", "अकेलापन": "loneliness",
    "तड़प": "longing", "तड़पन": "yearning",
    "जुनून": "passion", "जज्बा": "passion",
    "आरज़ू": "desire", "ख्वाहिश": "wish",
    "सुकून": "peace", "चैन": "peace",

    # Nouns — nature
    "रात": "night", "रातें": "nights",
    "दिन": "day", "दिनों": "days",
    "चाँद": "moon", "चंद्रमा": "moon",
    "सूरज": "sun", "सूर्य": "sun",
    "आसमान": "sky", "आकाश": "sky",
    "तारा": "star", "तारे": "stars", "सितारे": "stars",
    "बादल": "cloud", "बादलों": "clouds",
    "बारिश": "rain", "बरसात": "rain",
    "हवा": "wind/breeze", "पवन": "wind",
    "पानी": "water", "नदी": "river", "समुद्र": "ocean",
    "फूल": "flower", "फूलों": "flowers",
    "पत्ता": "leaf", "पत्ते": "leaves",
    "पेड़": "tree", "पेड़ों": "trees",
    "धरती": "earth", "जमीन": "ground",
    "रोशनी": "light", "उजाला": "light",
    "अँधेरा": "darkness", "अंधकार": "darkness",
    "आग": "fire", "लौ": "flame",
    "खुशबू": "fragrance", "महक": "scent",
    "शाम": "evening", "सुबह": "morning",
    "भोर": "dawn", "सवेरा": "dawn",

    # Nouns — people/body
    "आँखें": "eyes", "आंखें": "eyes", "नयन": "eyes",
    "होंठ": "lips", "अधर": "lips",
    "हाथ": "hand", "हाथों": "hands",
    "दिमाग": "mind", "मन": "mind/heart",
    "रूह": "soul", "आत्मा": "soul",
    "जिंदगी": "life", "ज़िंदगी": "life", "जीवन": "life",
    "मौत": "death", "मृत्यु": "death",
    "साँस": "breath", "श्वास": "breath",
    "जिस्म": "body", "तन": "body",
    "माँ": "mother", "बाप": "father", "यार": "friend/beloved",
    "दोस्त": "friend", "साथी": "companion",
    "महबूब": "beloved", "साजन": "beloved",

    # Adjectives
    "सुंदर": "beautiful", "खूबसूरत": "beautiful",
    "अच्छा": "good", "अच्छी": "good", "बुरा": "bad",
    "बड़ा": "big", "छोटा": "small",
    "नया": "new", "पुराना": "old",
    "गहरा": "deep", "ऊँचा": "high",
    "सच्चा": "true", "झूठा": "false",
    "तेज़": "fast/bright", "धीमा": "slow/soft",
    "ठंडा": "cold", "गर्म": "warm/hot",

    # Prepositions / particles
    "में": "in", "पर": "on/at", "से": "from/with",
    "को": "to", "के": "of", "की": "of", "का": "of",
    "तक": "until/till", "बिना": "without",
    "साथ": "with", "लिए": "for",
    "अब": "now", "फिर": "again/then", "कभी": "ever/sometimes",
    "कभी-कभी": "sometimes", "हमेशा": "always",
    "आज": "today", "कल": "yesterday/tomorrow",
    "यहाँ": "here", "वहाँ": "there",
    "जब": "when", "तब": "then",
    "क्यों": "why", "कैसे": "how", "क्या": "what",
    "नहीं": "not", "ना": "no/not", "मत": "don't",
    "भी": "also/too", "ही": "only/just", "तो": "then/so",
    "और": "and", "या": "or", "लेकिन": "but", "पर": "but",

    # Urdu-specific (common in ghazals)
    "शायर": "poet", "शायरी": "poetry",
    "ग़ज़ल": "ghazal", "नज़्म": "poem",
    "मिसरा": "verse", "शेर": "couplet",
    "रदीफ": "radif", "काफ़िया": "rhyme",
    "मतला": "opening couplet", "मक़ता": "closing couplet",
    "दीवान": "collection", "दीवाना": "madman/lover",
    "मयकदा": "tavern", "मय": "wine",
    "साक़ी": "wine-bearer", "जाम": "cup",
    "परदा": "veil/curtain", "पर्दा": "veil",
    "मंजिल": "destination", "राह": "path",
    "मुसाफिर": "traveler", "सफर": "journey",
    "वक्त": "time", "ज़माना": "era/world",
    "दुनिया": "world", "जहाँ": "world/where",
    "खुदा": "God", "रब": "God",
    "इबादत": "worship", "दुआ": "prayer",
}

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — Reordering Rules (SOV → SVO)
# Hindi is Subject-Object-Verb; English is Subject-Verb-Object.
# We apply simple heuristic: if last token looks like a verb, move it forward.
# ─────────────────────────────────────────────────────────────────────────────

VERB_ENDINGS = ["ता", "ती", "ते", "ना", "ना", "या", "ई", "है", "हैं", "था", "थे", "थी"]

def looks_like_verb(hindi_word):
    return any(hindi_word.endswith(e) for e in VERB_ENDINGS)

def reorder_sov_to_svo(tokens):
    """
    Heuristic: if the last token is a verb, move it to position 1 (after subject).
    Handles the most common SOV→SVO case.
    """
    if len(tokens) >= 3 and looks_like_verb(tokens[-1]):
        return [tokens[0], tokens[-1]] + tokens[1:-1]
    return tokens

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — Rhyme Scheme Extractor
# ─────────────────────────────────────────────────────────────────────────────

def get_line_ending(line: str) -> str:
    """Extract the last 2–3 characters of the last word as a rhyme key."""
    words = line.strip().split()
    if not words:
        return ""
    last = words[-1]
    # strip punctuation
    last = re.sub(r'[।\.!?,;:\-]+$', '', last)
    return last[-3:] if len(last) >= 3 else last

def detect_rhyme_scheme(lines: list) -> str:
    """
    Assign rhyme labels (A, B, C...) to each line based on ending similarity.
    Returns string like 'AABB' or 'ABAB'.
    """
    endings = [get_line_ending(l) for l in lines if l.strip()]
    labels = {}
    scheme = []
    counter = 0
    for end in endings:
        if end not in labels:
            labels[end] = chr(ord('A') + counter)
            counter += 1
        scheme.append(labels[end])
    return "".join(scheme)

def rhyme_density(lines: list) -> float:
    """Fraction of lines that share a rhyme with at least one other line."""
    endings = [get_line_ending(l) for l in lines if l.strip()]
    counts = defaultdict(int)
    for e in endings:
        counts[e] += 1
    rhyming = sum(1 for e in endings if counts[e] > 1)
    return round(rhyming / len(endings), 3) if endings else 0.0

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — Core Translation Function
# ─────────────────────────────────────────────────────────────────────────────

def tokenize_hindi(text: str) -> list:
    """Simple whitespace + punctuation tokenizer for Hindi/Devanagari."""
    text = re.sub(r'[।\.!?,;:\-"\']+', ' ', text)
    return [t for t in text.split() if t]

def translate_token(token: str) -> str:
    """Look up a single token; return [token] if not found."""
    if token in HINDI_DICT:
        return HINDI_DICT[token]
    # Try lowercase
    if token.lower() in HINDI_DICT:
        return HINDI_DICT[token.lower()]
    return f"[{token}]"   # unknown token marker

def translate_line(line: str, reorder: bool = True) -> str:
    """Translate one line of Hindi poetry."""
    tokens = tokenize_hindi(line)
    if reorder:
        tokens = reorder_sov_to_svo(tokens)
    translated = [translate_token(t) for t in tokens]
    return " ".join(translated)

def translate_poem(poem_text: str) -> dict:
    """
    Translate a full poem (multi-line string).
    Returns dict with translated text and style analysis.
    """
    lines = [l for l in poem_text.strip().split('\n') if l.strip()]
    translated_lines = [translate_line(l) for l in lines]

    # Style analysis on source
    src_scheme  = detect_rhyme_scheme(lines)
    src_density = rhyme_density(lines)

    # Style analysis on translation
    tgt_scheme  = detect_rhyme_scheme(translated_lines)
    tgt_density = rhyme_density(translated_lines)

    # Coverage: fraction of tokens found in dict
    all_tokens = [t for l in lines for t in tokenize_hindi(l)]
    known = sum(1 for t in all_tokens if t in HINDI_DICT)
    coverage = round(known / len(all_tokens), 3) if all_tokens else 0.0

    return {
        "source_lines":       lines,
        "translated_lines":   translated_lines,
        "translated_text":    "\n".join(translated_lines),
        "src_rhyme_scheme":   src_scheme,
        "tgt_rhyme_scheme":   tgt_scheme,
        "src_rhyme_density":  src_density,
        "tgt_rhyme_density":  tgt_density,
        "dict_coverage":      coverage,
        "num_lines":          len(lines),
    }

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — BLEU Score (simple corpus-level, no sacrebleu needed)
# ─────────────────────────────────────────────────────────────────────────────

from collections import Counter
import math

def ngrams(tokens, n):
    return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

def simple_bleu(hypothesis: str, reference: str, max_n: int = 4) -> float:
    """Simplified sentence-level BLEU score."""
    hyp_tokens = hypothesis.lower().split()
    ref_tokens = reference.lower().split()

    if not hyp_tokens:
        return 0.0

    # Brevity penalty
    bp = min(1.0, math.exp(1 - len(ref_tokens) / len(hyp_tokens))) if hyp_tokens else 0.0

    precisions = []
    for n in range(1, max_n + 1):
        hyp_ng = Counter(ngrams(hyp_tokens, n))
        ref_ng = Counter(ngrams(ref_tokens, n))
        clipped = sum(min(c, ref_ng[g]) for g, c in hyp_ng.items())
        total   = max(len(hyp_tokens) - n + 1, 0)
        precisions.append(clipped / total if total > 0 else 0.0)

    if any(p == 0 for p in precisions):
        return 0.0

    log_avg = sum(math.log(p) for p in precisions) / max_n
    return round(bp * math.exp(log_avg) * 100, 2)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — Run on Test Data
# ─────────────────────────────────────────────────────────────────────────────

def run_on_csv(csv_path: str, max_rows: int = 200) -> list:
    """Translate rows from a CSV with 'hindi' and 'english' columns."""
    import csv
    results = []
    with open(csv_path, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= max_rows:
                break
            src      = row["hi"].strip()
            ref_lit  = row.get("en_t", "").strip()
            ref_style = row.get("en_anthropic", "").strip()
            result   = translate_poem(src)
            hyp      = result["translated_text"]
            bleu_lit   = simple_bleu(hyp, ref_lit)   if ref_lit   else None
            bleu_style = simple_bleu(hyp, ref_style) if ref_style else None
            results.append({
                "id":                i,
                "source":            src,
                "ref_literal":       ref_lit,
                "ref_style":         ref_style,
                "hypothesis":        hyp,
                "bleu_vs_literal":   bleu_lit,
                "bleu_vs_style":     bleu_style,
                "src_rhyme_scheme":  result["src_rhyme_scheme"],
                "tgt_rhyme_scheme":  result["tgt_rhyme_scheme"],
                "src_rhyme_density": result["src_rhyme_density"],
                "tgt_rhyme_density": result["tgt_rhyme_density"],
                "dict_coverage":     result["dict_coverage"],
            })
    return results

def run_on_sample_poems() -> list:
    """
    Run on a built-in set of sample Hindi/Urdu couplets when no CSV is available.
    These are generic illustrative examples — replace with real test data on GCP.
    """
    samples = [
        {
            "hindi": "दिल में दर्द है\nआँखों में आँसू हैं\nफिर भी मुस्कुराता हूँ",
            "english": "There is pain in the heart\nThere are tears in the eyes\nStill I smile"
        },
        {
            "hindi": "रात के तारे चमकते हैं\nचाँद आसमान में है\nसब कुछ सुंदर है",
            "english": "Stars shine in the night\nThe moon is in the sky\nEverything is beautiful"
        },
        {
            "hindi": "तेरा इंतजार है\nमेरी उम्मीद तू है\nतू आएगा कभी",
            "english": "I wait for you\nYou are my hope\nYou will come someday"
        },
        {
            "hindi": "प्यार की राह में\nदर्द मिलता है\nफिर भी दिल मानता नहीं",
            "english": "On the path of love\nPain is found\nStill the heart does not listen"
        },
        {
            "hindi": "ज़िंदगी एक सफर है\nहम मुसाफिर हैं\nमंजिल दूर है",
            "english": "Life is a journey\nWe are travelers\nThe destination is far"
        },
    ]
    results = []
    for i, s in enumerate(samples):
        result = translate_poem(s["hindi"])
        bleu = simple_bleu(result["translated_text"], s["english"])
        results.append({
            "id":               i,
            "source":           s["hindi"],
            "reference":        s["english"],
            "hypothesis":       result["translated_text"],
            "bleu":             bleu,
            "src_rhyme_scheme": result["src_rhyme_scheme"],
            "tgt_rhyme_scheme": result["tgt_rhyme_scheme"],
            "src_rhyme_density":result["src_rhyme_density"],
            "tgt_rhyme_density":result["tgt_rhyme_density"],
            "dict_coverage":    result["dict_coverage"],
        })
    return results

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — Save Outputs
# ─────────────────────────────────────────────────────────────────────────────

def save_outputs(results: list):
    import csv

    # CSV
    csv_path = RESULTS_DIR / "tier1_translations.csv"
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"  Saved CSV → {csv_path}")

    # Metrics JSON
    bleu_lit_scores   = [r["bleu_vs_literal"]  for r in results if r["bleu_vs_literal"]  is not None]
    bleu_style_scores = [r["bleu_vs_style"]    for r in results if r["bleu_vs_style"]    is not None]
    metrics = {
        "model":                   "Tier1_RuleBasedBaseline",
        "num_samples":             len(results),
        "avg_bleu_vs_literal":     round(sum(bleu_lit_scores)  /len(bleu_lit_scores),   2) if bleu_lit_scores   else None,
        "avg_bleu_vs_style":       round(sum(bleu_style_scores)/len(bleu_style_scores), 2) if bleu_style_scores else None,
        "avg_dict_coverage":       round(sum(r["dict_coverage"] for r in results)/len(results), 3),
        "avg_src_rhyme_density":   round(sum(r["src_rhyme_density"] for r in results)/len(results), 3),
        "avg_tgt_rhyme_density":   round(sum(r["tgt_rhyme_density"] for r in results)/len(results), 3),
        "rhyme_preservation_rate": round(
            sum(1 for r in results if r["src_rhyme_scheme"] == r["tgt_rhyme_scheme"]) / len(results), 3
        ),
    }
    json_path = RESULTS_DIR / "tier1_metrics.json"
    with open(json_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved metrics → {json_path}")

    # Human-readable report
    report_path = RESULTS_DIR / "tier1_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("TIER 1 — RULE-BASED BASELINE TRANSLATION REPORT\n")
        f.write("DATS 6312 Final Project\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Samples evaluated : {metrics['num_samples']}\n")
        f.write(f"Avg BLEU vs literal: {metrics['avg_bleu_vs_literal']}\n")
        f.write(f"Avg BLEU vs style  : {metrics['avg_bleu_vs_style']}\n")
        f.write(f"Dict coverage      : {metrics['avg_dict_coverage']*100:.1f}%\n")
        f.write(f"Rhyme density src  : {metrics['avg_src_rhyme_density']}\n")
        f.write(f"Rhyme density tgt  : {metrics['avg_tgt_rhyme_density']}\n")
        f.write(f"Rhyme scheme match : {metrics['rhyme_preservation_rate']*100:.1f}%\n\n")
        f.write("-" * 70 + "\n\n")
        for r in results:
            f.write(f"[{r['id']}] SOURCE:\n{r['source']}\n\n")
            f.write(f"REF LITERAL:\n{r['ref_literal']}\n\n")
            f.write(f"REF STYLE:\n{r['ref_style']}\n\n")
            f.write(f"HYPOTHESIS:\n{r['hypothesis']}\n\n")
            f.write(f"BLEU vs literal: {r['bleu_vs_literal']}  |  BLEU vs style: {r['bleu_vs_style']}  "
                    f"|  Coverage: {r['dict_coverage']}  "
                    f"|  Src Scheme: {r['src_rhyme_scheme']}  |  Tgt Scheme: {r['tgt_rhyme_scheme']}\n")
            f.write("-" * 70 + "\n\n")
    print(f"  Saved report → {report_path}")

    return metrics

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("TIER 1 — RULE-BASED BASELINE")
    print("=" * 60)

    # Try to load real test data; fall back to built-in samples
    poetry_csv = DATA_DIR / "poetry_poems_test.csv"
    test_csv   = DATA_DIR / "iitb_test_filtered.csv"

    if poetry_csv.exists():
        print(f"\nUsing Kaggle poetry test set: {poetry_csv}")
        results = run_on_csv(str(poetry_csv), max_rows=200)
    elif test_csv.exists():
        print(f"\nUsing IIT Bombay test set: {test_csv}")
        results = run_on_csv(str(test_csv), max_rows=200)
    else:
        print("\nNo CSV found — running on built-in sample poems.")
        print("(Run on GCP with real test CSVs for actual evaluation)\n")
        results = run_on_sample_poems()

    print(f"\nTranslated {len(results)} samples.")

    # Print a few examples
    print("\n── Sample Translations ──────────────────────────────────────")
    for r in results[:3]:
        print(f"\nSOURCE:          {r['source'][:80]}")
        print(f"REF LITERAL:     {r['ref_literal'][:80]}")
        print(f"REF STYLE:       {r['ref_style'][:80]}")
        print(f"HYPOTHESIS:      {r['hypothesis'][:80]}")
        print(f"BLEU literal: {r['bleu_vs_literal']}  BLEU style: {r['bleu_vs_style']}  "
              f"Coverage: {r['dict_coverage']}  Scheme: {r['src_rhyme_scheme']} → {r['tgt_rhyme_scheme']}")

    print("\n── Saving outputs ───────────────────────────────────────────")
    metrics = save_outputs(results)

    print("\n── Final Metrics ────────────────────────────────────────────")
    for k, v in metrics.items():
        print(f"  {k:<30} {v}")

    print("\n✅ TIER 1 COMPLETE")
    print(f"   Results saved to {RESULTS_DIR}")
    print("   These baseline numbers go in your report Section 6 (Results).")
    print("   Tier 2 (Seq2Seq LSTM) and Tier 3 (Opus-MT) should outperform these.")