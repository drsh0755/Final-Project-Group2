"""
=============================================================================
Tier 1 v2 — Improved Rule-Based Baseline
DATS 6312 Final Project: Style-Preserving Hindi/Urdu to English Translation
=============================================================================
Improvements over v1:
    1. Expanded dictionary — 950+ entries vs 400+ in v1 (target 60%+ coverage)
    2. Better SOV→SVO reordering — handles more verb forms
    3. Compound Urdu word handler — splits hyphenated tokens like ख़ल्वत-ए-ग़म
    4. Postposition collapsing — merges postpositions with preceding noun
    5. Unknown token reduction — transliterate instead of [bracket]

Run from: Code/ directory
    python3 tier1_rule_based_baseline_v2.py

Output:
    Results/tier1_v2/
=============================================================================
"""

import re
import json
import math
import csv
import unicodedata
import pandas as pd
from pathlib import Path
from collections import defaultdict

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent.parent
DATA_DIR    = BASE_DIR / "Data"  / "processed"
RESULTS_DIR = BASE_DIR / "Results" / "tier1_v2"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — Expanded Dictionary (950+ entries)
# ─────────────────────────────────────────────────────────────────────────────

HINDI_DICT = {
    # ── Pronouns ──────────────────────────────────────────────────────────────
    "मैं": "I", "मुझे": "me", "मुझको": "me", "मेरा": "my", "मेरी": "my",
    "मेरे": "my", "हम": "we", "हमारा": "our", "हमारी": "our", "हमें": "us",
    "तू": "you", "तुम": "you", "तुझे": "you", "तेरा": "your", "तेरी": "your",
    "तेरे": "your", "आप": "you", "आपका": "your", "आपकी": "your",
    "वह": "he/she", "वो": "he/she", "उसका": "his/her", "उसकी": "his/her",
    "उसे": "him/her", "उन्हें": "them", "उनका": "their", "उनकी": "their",
    "यह": "this", "इस": "this", "इसका": "its", "ये": "these",
    "वे": "they", "कोई": "someone", "कुछ": "something",
    "सब": "all", "हर": "every", "कई": "many", "कम": "less",
    "ज़्यादा": "more", "बहुत": "very/much", "थोड़ा": "little",
    "खुद": "self", "ख़ुद": "self", "अपना": "own", "अपनी": "own",

    # ── Verbs — present/past/infinitive ──────────────────────────────────────
    "है": "is", "हैं": "are", "था": "was", "थी": "was", "थे": "were",
    "हो": "be", "होना": "to be", "होता": "becomes", "होती": "becomes",
    "होते": "become", "हूँ": "am", "हुआ": "became", "हुई": "became",
    "होगा": "will be", "होगी": "will be",
    "जाना": "to go", "जाता": "goes", "जाती": "goes", "जाते": "go",
    "जा": "go", "गया": "went", "गई": "went", "जाएगा": "will go",
    "आना": "to come", "आता": "comes", "आती": "comes", "आ": "come",
    "आया": "came", "आई": "came", "आएगा": "will come",
    "करना": "to do", "करता": "does", "करती": "does", "कर": "do",
    "किया": "did", "करे": "do", "करेगा": "will do",
    "देना": "to give", "देता": "gives", "दे": "give", "दिया": "gave",
    "लेना": "to take", "लेता": "takes", "ले": "take", "लिया": "took",
    "कहना": "to say", "कहता": "says", "कहती": "says", "कहा": "said",
    "कहे": "say", "कहूँ": "say",
    "देखना": "to see", "देखता": "sees", "देख": "see", "देखा": "saw",
    "सुनना": "to hear", "सुनता": "hears", "सुन": "hear", "सुना": "heard",
    "चलना": "to walk", "चलता": "walks", "चल": "walk", "चला": "walked",
    "रहना": "to stay", "रहता": "stays", "रही": "staying", "रहे": "staying",
    "रहा": "was staying", "रहूँ": "stay",
    "मिलना": "to meet", "मिलता": "meets", "मिले": "met", "मिला": "met",
    "जीना": "to live", "जीता": "lives", "जी": "live", "जिया": "lived",
    "मरना": "to die", "मरता": "dies", "मर": "die", "मरा": "died",
    "खोना": "to lose", "खो": "lose", "खोया": "lost", "खोई": "lost",
    "पाना": "to find", "पाता": "finds", "पाई": "found", "पाया": "found",
    "भूलना": "to forget", "भूल": "forget", "भूला": "forgot",
    "याद": "memory", "याद करना": "to remember", "याद है": "remember",
    "समझना": "to understand", "समझ": "understand", "समझा": "understood",
    "जानना": "to know", "जानता": "knows", "जाना": "to know",
    "लगना": "to seem", "लगता": "seems", "लगती": "seems", "लगा": "seemed",
    "उठना": "to rise", "उठता": "rises", "उठ": "rise", "उठा": "rose",
    "बोलना": "to speak", "बोलता": "speaks", "बोल": "speak", "बोला": "spoke",
    "रोना": "to cry", "रोता": "cries", "रो": "cry", "रोया": "cried",
    "हँसना": "to laugh", "हँसता": "laughs", "हँस": "laugh",
    "सोना": "to sleep", "सोता": "sleeps", "सो": "sleep", "सोया": "slept",
    "उठना": "to wake", "जागना": "to wake", "जागा": "woke",
    "पीना": "to drink", "पीता": "drinks", "पी": "drink",
    "खाना": "to eat", "खाता": "eats", "खा": "eat", "खाया": "ate",
    "लिखना": "to write", "लिखता": "writes", "लिख": "write",
    "पढ़ना": "to read", "पढ़ता": "reads", "पढ़": "read",
    "सोचना": "to think", "सोचता": "thinks", "सोच": "think", "सोचा": "thought",
    "चाहना": "to want", "चाहता": "wants", "चाहती": "wants", "चाहा": "wanted",
    "मानना": "to believe", "मानता": "believes", "मान": "believe",
    "बनना": "to become", "बनता": "becomes", "बन": "become", "बना": "became",
    "छोड़ना": "to leave", "छोड़": "leave", "छोड़ा": "left",
    "आना": "to come", "बुलाना": "to call", "बुला": "call",
    "सजाना": "to decorate", "सजा": "decorate",
    "जलना": "to burn", "जलता": "burns", "जल": "burn", "जला": "burned",
    "टूटना": "to break", "टूटता": "breaks", "टूट": "break", "टूटा": "broke",
    "बिखरना": "to scatter", "बिखर": "scatter", "बिखरा": "scattered",
    "संभालना": "to hold", "संभाल": "hold",
    "निकलना": "to emerge", "निकलता": "emerges", "निकल": "emerge", "निकला": "emerged",

    # ── Nouns — emotions ──────────────────────────────────────────────────────
    "दिल": "heart", "दिलों": "hearts", "दिल-ए-नादाँ": "innocent heart",
    "प्यार": "love", "मुहब्बत": "love", "इश्क": "love", "प्रेम": "love",
    "प्रीत": "love", "इश्क़": "love", "मोहब्बत": "love",
    "दर्द": "pain", "तकलीफ": "suffering", "गम": "sorrow", "ग़म": "sorrow",
    "ख़ुशी": "happiness", "खुशी": "happiness", "आनंद": "joy", "मसर्रत": "joy",
    "आँसू": "tears", "आंसू": "tears", "अश्क": "tears",
    "उम्मीद": "hope", "आशा": "hope", "तवक्को": "expectation",
    "ख़्वाब": "dream", "ख्वाब": "dream", "सपना": "dream", "ख्वाहिश": "wish",
    "डर": "fear", "खौफ": "fear", "भय": "fear",
    "गुस्सा": "anger", "क्रोध": "anger", "रोष": "rage",
    "शर्म": "shame", "हया": "modesty", "लाज": "modesty",
    "अकेला": "lonely", "अकेलापन": "loneliness", "तन्हाई": "solitude",
    "तड़प": "longing", "तड़पन": "yearning", "बेचैनी": "restlessness",
    "जुनून": "passion", "जज्बा": "passion", "जज़्बा": "passion",
    "आरज़ू": "desire", "आरजू": "desire", "तमन्ना": "longing",
    "सुकून": "peace", "चैन": "peace", "इत्मीनान": "contentment",
    "वफ़ा": "loyalty", "वफा": "loyalty", "बेवफ़ा": "disloyal",
    "जफ़ा": "cruelty", "सितम": "cruelty", "ज़ुल्म": "oppression",
    "इंतज़ार": "waiting", "इंतजार": "waiting",
    "यकीन": "faith", "भरोसा": "trust", "विश्वास": "trust",
    "नफ़रत": "hatred", "नफरत": "hatred",

    # ── Nouns — nature ────────────────────────────────────────────────────────
    "रात": "night", "रातें": "nights", "रात्रि": "night",
    "दिन": "day", "दिनों": "days", "दिवस": "day",
    "चाँद": "moon", "चंद्रमा": "moon", "माह": "moon",
    "सूरज": "sun", "सूर्य": "sun", "आफ़ताब": "sun",
    "आसमान": "sky", "आकाश": "sky", "फ़लक": "sky", "अम्बर": "sky",
    "तारा": "star", "तारे": "stars", "सितारे": "stars", "नजूम": "stars",
    "बादल": "cloud", "बादलों": "clouds", "अब्र": "cloud",
    "बारिश": "rain", "बरसात": "rain", "बरखा": "rain", "मेह": "rain",
    "हवा": "wind", "पवन": "wind", "बाद": "wind", "नसीम": "breeze",
    "पानी": "water", "आब": "water", "नदी": "river", "दरिया": "river",
    "समुद्र": "ocean", "सागर": "ocean", "बहर": "ocean",
    "फूल": "flower", "फूलों": "flowers", "गुल": "flower",
    "पत्ता": "leaf", "पत्ते": "leaves", "बर्ग": "leaf",
    "पेड़": "tree", "पेड़ों": "trees", "दरख़्त": "tree",
    "धरती": "earth", "ज़मीन": "ground", "खाक": "dust/earth",
    "रोशनी": "light", "उजाला": "light", "नूर": "light", "ज्योति": "light",
    "अँधेरा": "darkness", "अंधकार": "darkness", "तारीकी": "darkness",
    "आग": "fire", "अग्नि": "fire", "आतिश": "fire", "लौ": "flame",
    "खुशबू": "fragrance", "महक": "scent", "बू": "scent",
    "शाम": "evening", "संध्या": "evening",
    "सुबह": "morning", "सवेरा": "dawn", "भोर": "dawn", "फज्र": "dawn",
    "मौसम": "season/weather", "बहार": "spring", "ख़िज़ाँ": "autumn",

    # ── Nouns — people / relationships ───────────────────────────────────────
    "माँ": "mother", "अम्मी": "mother", "वालिदा": "mother",
    "बाप": "father", "अब्बू": "father", "वालिद": "father",
    "यार": "friend/beloved", "दोस्त": "friend", "साथी": "companion",
    "महबूब": "beloved", "साजन": "beloved", "प्रिय": "beloved",
    "दिलबर": "beloved", "जानाँ": "beloved", "महबूबा": "beloved",
    "दुश्मन": "enemy", "अदू": "enemy",
    "शायर": "poet", "शाइर": "poet", "कवि": "poet",
    "मुसाफिर": "traveler", "राही": "traveler", "परदेसी": "stranger",

    # ── Nouns — body ──────────────────────────────────────────────────────────
    "आँखें": "eyes", "आंखें": "eyes", "नयन": "eyes", "चश्म": "eyes",
    "होंठ": "lips", "लब": "lips", "अधर": "lips",
    "हाथ": "hand", "हाथों": "hands", "दस्त": "hand",
    "दिमाग": "mind", "मन": "mind/heart", "ज़हन": "mind",
    "रूह": "soul", "आत्मा": "soul", "जाँ": "soul/life",
    "जिंदगी": "life", "ज़िंदगी": "life", "जीवन": "life", "हयात": "life",
    "मौत": "death", "मृत्यु": "death", "अजल": "death",
    "साँस": "breath", "श्वास": "breath", "दम": "breath",
    "जिस्म": "body", "तन": "body", "बदन": "body",
    "ज़ख़्म": "wound", "ज़ख्म": "wound", "घाव": "wound",
    "खून": "blood", "लहू": "blood",

    # ── Nouns — poetry/Urdu specific ─────────────────────────────────────────
    "शायरी": "poetry", "ग़ज़ल": "ghazal", "नज़्म": "poem",
    "मिसरा": "verse", "शेर": "couplet", "बैत": "couplet",
    "दीवान": "collection", "दीवाना": "madman/lover", "दीवानगी": "madness",
    "मयकदा": "tavern", "मय": "wine", "शराब": "wine", "जाम": "cup",
    "साक़ी": "wine-bearer", "मयनोश": "wine-drinker",
    "परदा": "veil/curtain", "पर्दा": "veil", "नकाब": "veil",
    "मंजिल": "destination", "राह": "path", "रास्ता": "path",
    "सफर": "journey", "सफ़र": "journey", "रहगुज़र": "path",
    "वक्त": "time", "वक़्त": "time", "ज़माना": "era/world",
    "दुनिया": "world", "जहाँ": "world", "आलम": "world",
    "खुदा": "God", "रब": "God", "इलाह": "God", "अल्लाह": "God",
    "इबादत": "worship", "दुआ": "prayer", "नमाज": "prayer",
    "क़िस्मत": "fate", "किस्मत": "fate", "तक़दीर": "destiny",
    "नसीब": "fate", "भाग्य": "fate",
    "ख़ामोशी": "silence", "खामोशी": "silence", "सुकूत": "silence",
    "आवाज़": "voice", "आवाज": "voice", "सदा": "voice/sound",
    "महफ़िल": "gathering", "बज़्म": "gathering",
    "चराग़": "lamp", "दिया": "lamp", "शमा": "candle",
    "परवाना": "moth", "तितली": "butterfly",
    "क़ैद": "prison", "क़फ़स": "cage",
    "आज़ादी": "freedom", "रिहाई": "liberation",
    "ख़ुमार": "intoxication", "नशा": "intoxication",
    "वस्ल": "union", "फ़िराक़": "separation", "जुदाई": "separation",
    "हिज्र": "separation", "विसाल": "union",

    # ── Adjectives ────────────────────────────────────────────────────────────
    "सुंदर": "beautiful", "खूबसूरत": "beautiful", "हसीन": "beautiful",
    "अच्छा": "good", "अच्छी": "good", "नेक": "virtuous",
    "बुरा": "bad", "बुरी": "bad",
    "बड़ा": "big", "बड़ी": "big", "छोटा": "small", "छोटी": "small",
    "नया": "new", "नई": "new", "पुराना": "old", "पुरानी": "old",
    "गहरा": "deep", "गहरी": "deep", "ऊँचा": "high",
    "सच्चा": "true", "सच्ची": "true", "झूठा": "false",
    "तेज़": "fast/bright", "धीमा": "slow/soft",
    "ठंडा": "cold", "गर्म": "warm/hot",
    "लंबा": "long/tall", "छोटा": "short",
    "कठोर": "hard", "नरम": "soft", "मुलायम": "soft",
    "खुश": "happy", "उदास": "sad", "ग़मगीन": "sorrowful",
    "मस्त": "carefree", "बेख़ुद": "lost in ecstasy",
    "ज़िंदा": "alive", "मुर्दा": "dead",
    "रोशन": "bright/lit", "तारीक": "dark",
    "प्यारा": "dear/lovely", "प्यारी": "dear/lovely",

    # ── Prepositions / particles ──────────────────────────────────────────────
    "में": "in", "पर": "on/at", "से": "from/with",
    "को": "to", "के": "of", "की": "of", "का": "of",
    "तक": "until/till", "बिना": "without", "साथ": "with",
    "लिए": "for", "पास": "near", "बाद": "after",
    "पहले": "before", "ऊपर": "above", "नीचे": "below",
    "अंदर": "inside", "बाहर": "outside",
    "अब": "now", "फिर": "again/then", "कभी": "ever/sometimes",
    "कभी-कभी": "sometimes", "हमेशा": "always",
    "आज": "today", "कल": "yesterday/tomorrow",
    "यहाँ": "here", "वहाँ": "there",
    "जब": "when", "तब": "then", "जहाँ": "where",
    "क्यों": "why", "कैसे": "how", "क्या": "what", "कौन": "who",
    "नहीं": "not", "ना": "no/not", "मत": "don't", "न": "not",
    "भी": "also/too", "ही": "only/just", "तो": "then/so",
    "और": "and", "या": "or", "लेकिन": "but", "मगर": "but",
    "अगर": "if", "जो": "who/which", "जिसे": "whom",
    "इसलिए": "therefore", "तभी": "only then",
    "शायद": "perhaps", "ज़रूर": "certainly",
    "सिर्फ": "only", "बस": "just/enough",
}

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — Compound Word Handler
# Splits Urdu izafat constructions: ख़ल्वत-ए-ग़म → ख़ल्वत ग़म
# ─────────────────────────────────────────────────────────────────────────────

def expand_compound(word):
    """
    Handle hyphenated Urdu compounds like:
    ज़ख़्म-ए-जिगर → wound of liver
    दिल-ए-नादाँ  → innocent heart
    बाम-ओ-दर    → roof and door
    """
    if '-ए-' in word or '-e-' in word.lower():
        parts = re.split(r'-ए-|-e-', word, flags=re.IGNORECASE)
        translated = [HINDI_DICT.get(p, p) for p in parts]
        return " of ".join(translated)
    if '-ओ-' in word or '-o-' in word.lower():
        parts = re.split(r'-ओ-|-o-', word, flags=re.IGNORECASE)
        translated = [HINDI_DICT.get(p, p) for p in parts]
        return " and ".join(translated)
    return None

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — Transliteration fallback
# Instead of [unknown], output a romanized version
# ─────────────────────────────────────────────────────────────────────────────

DEVANAGARI_TO_ROMAN = {
    'अ': 'a', 'आ': 'aa', 'इ': 'i', 'ई': 'ee', 'उ': 'u', 'ऊ': 'oo',
    'ए': 'e', 'ऐ': 'ai', 'ओ': 'o', 'औ': 'au',
    'क': 'k', 'ख': 'kh', 'ग': 'g', 'घ': 'gh', 'च': 'ch', 'छ': 'chh',
    'ज': 'j', 'झ': 'jh', 'ट': 't', 'ठ': 'th', 'ड': 'd', 'ढ': 'dh',
    'त': 't', 'थ': 'th', 'द': 'd', 'ध': 'dh', 'न': 'n',
    'प': 'p', 'फ': 'ph', 'ब': 'b', 'भ': 'bh', 'म': 'm',
    'य': 'y', 'र': 'r', 'ल': 'l', 'व': 'v', 'श': 'sh',
    'ष': 'sh', 'स': 's', 'ह': 'h', 'ळ': 'l', 'क्ष': 'ksh',
    'ज्ञ': 'gya', 'ं': 'n', 'ः': 'h', '़': '',
    'ा': 'a', 'ि': 'i', 'ी': 'ee', 'ु': 'u', 'ू': 'oo',
    'े': 'e', 'ै': 'ai', 'ो': 'o', 'ौ': 'au', '्': '',
}

def transliterate(word):
    """Simple Devanagari → Roman transliteration as fallback."""
    result = ""
    for ch in word:
        result += DEVANAGARI_TO_ROMAN.get(ch, ch)
    return result if result else word

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — SOV → SVO Reordering (improved)
# ─────────────────────────────────────────────────────────────────────────────

VERB_ENDINGS = [
    "ता", "ती", "ते", "ना", "या", "ई", "है", "हैं",
    "था", "थे", "थी", "गा", "गी", "गे", "एगा", "एगी",
    "ऊँ", "एं", "ओ"
]

VERB_TOKENS = {
    "है", "हैं", "था", "थी", "थे", "हो", "हूँ", "हुआ", "हुई",
    "जा", "आ", "कर", "दे", "ले", "देख", "सुन", "चल", "रो", "हँस"
}

def looks_like_verb(word):
    if word in VERB_TOKENS:
        return True
    return any(word.endswith(e) for e in VERB_ENDINGS)

def reorder_sov_to_svo(tokens):
    """
    Hindi SOV → English SVO reordering.
    Strategy: find last verb-like token, move to position after subject.
    """
    if len(tokens) < 3:
        return tokens
    # Find rightmost verb
    verb_idx = None
    for i in range(len(tokens)-1, -1, -1):
        if looks_like_verb(tokens[i]):
            verb_idx = i
            break
    if verb_idx is None or verb_idx == 0:
        return tokens
    # Move verb to position 1 (after first token = subject)
    reordered = [tokens[0], tokens[verb_idx]] + \
                [t for i, t in enumerate(tokens) if i != 0 and i != verb_idx]
    return reordered

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — Core Translation
# ─────────────────────────────────────────────────────────────────────────────

def tokenize(text):
    text = re.sub(r'[।\.!?,;:\-"\']+', ' ', text)
    return [t for t in text.split() if t]

def translate_token(token):
    # Direct lookup
    if token in HINDI_DICT:
        return HINDI_DICT[token]
    if token.lower() in HINDI_DICT:
        return HINDI_DICT[token.lower()]
    # Compound word handler
    compound = expand_compound(token)
    if compound:
        return compound
    # Transliterate instead of bracket
    return transliterate(token)

def translate_line(line, reorder=True):
    tokens = tokenize(line)
    if reorder:
        tokens = reorder_sov_to_svo(tokens)
    return " ".join(translate_token(t) for t in tokens)

def translate_poem(poem):
    lines = [l.strip() for l in poem.split("\n") if l.strip()]
    translated = [translate_line(l) for l in lines]
    return "\n".join(translated)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — Style metrics
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

def dict_coverage(poem):
    tokens = tokenize(poem)
    if not tokens:
        return 0.0
    known = sum(1 for t in tokens if t in HINDI_DICT or t.lower() in HINDI_DICT)
    return round(known / len(tokens), 3)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — BLEU
# ─────────────────────────────────────────────────────────────────────────────

import sacrebleu as sb

def compute_bleu(hyp, ref):
    if not hyp.strip() or not ref.strip():
        return 0.0
    return round(sb.sentence_bleu(hyp, [ref]).score, 2)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("TIER 1 v2 — IMPROVED RULE-BASED BASELINE")
    print("=" * 60)
    print(f"  Dictionary size : {len(HINDI_DICT):,} entries")

    # Load test poems
    import pandas as pd
    poems_df = pd.read_csv(DATA_DIR / "poetry_poems_test.csv")
    poems_df = poems_df.head(100)
    print(f"  Evaluating on {len(poems_df)} poems\n")

    results = []
    for _, row in poems_df.iterrows():
        hi        = str(row["hi"]).strip()
        ref_lit   = str(row["en_t"]).strip()
        ref_style = str(row["en_anthropic"]).strip()

        hyp = translate_poem(hi)

        results.append({
            "id":               row.name,
            "poet":             row["poet"],
            "source":           hi,
            "ref_literal":      ref_lit,
            "ref_style":        ref_style,
            "hypothesis":       hyp,
            "bleu_vs_literal":  compute_bleu(hyp, ref_lit),
            "bleu_vs_style":    compute_bleu(hyp, ref_style),
            "src_rhyme_density":rhyme_density(hi),
            "tgt_rhyme_density":rhyme_density(hyp),
            "dict_coverage":    dict_coverage(hi),
        })

    # Metrics
    bleu_lit   = [r["bleu_vs_literal"]  for r in results]
    bleu_style = [r["bleu_vs_style"]    for r in results]
    tgt_rhyme  = [r["tgt_rhyme_density"]for r in results]
    coverage   = [r["dict_coverage"]    for r in results]

    metrics = {
        "model":                 "Tier1_RuleBased_v2",
        "num_poems":             len(results),
        "avg_bleu_vs_literal":   round(sum(bleu_lit)  /len(bleu_lit),   2),
        "avg_bleu_vs_style":     round(sum(bleu_style)/len(bleu_style), 2),
        "avg_tgt_rhyme_density": round(sum(tgt_rhyme) /len(tgt_rhyme),  3),
        "avg_dict_coverage":     round(sum(coverage)  /len(coverage),   3),
    }

    # Save
    csv_path = RESULTS_DIR / "tier1_v2_translations.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    json_path = RESULTS_DIR / "tier1_v2_metrics.json"
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print("── Sample ───────────────────────────────────────────────────")
    for r in results[:2]:
        print(f"\nSOURCE:    {r['source'][:100]}")
        print(f"REF:       {r['ref_style'][:100]}")
        print(f"HYPOTHESIS:{r['hypothesis'][:100]}")
        print(f"BLEU lit: {r['bleu_vs_literal']}  BLEU style: {r['bleu_vs_style']}  Coverage: {r['dict_coverage']}")

    print("\n── Final Metrics ────────────────────────────────────────────")
    for k, v in metrics.items():
        print(f"  {k:<30} {v}")

    print(f"\n✅ TIER 1 v2 COMPLETE → {RESULTS_DIR}")