"""
Preprocessing pipeline (Section III / Fig. 1 preprocessing block):
    - noise removal
    - Unicode / case normalisation
    - tokenisation
    - stop-word removal
    - emoji extraction (emojis are pulled OUT of the text before the text
      is fed to mBERT, and mapped separately to an emotion vector by
      src/emoji_features.py, mirroring "Emoji and Emotion extraction" in Fig. 1)
"""

import re
import unicodedata
import emoji as emoji_lib
import nltk

for pkg in ("punkt", "punkt_tab", "stopwords"):
    try:
        nltk.data.find(f"tokenizers/{pkg}" if "punkt" in pkg else f"corpora/{pkg}")
    except LookupError:
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass

from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

URL_RE = re.compile(r"https?://\S+|www\.\S+")
HTML_RE = re.compile(r"<.*?>")
MULTISPACE_RE = re.compile(r"\s+")
REPEATED_PUNCT_RE = re.compile(r"([!?.]){2,}")

try:
    _EN_STOPWORDS = set(stopwords.words("english"))
except Exception:
    _EN_STOPWORDS = set()

# A small, extensible Hindi stop-word list (NLTK does not ship one).
_HI_STOPWORDS = {
    "और", "है", "में", "की", "का", "को", "पर", "यह", "से", "हैं",
    "थी", "था", "कि", "जो", "एक", "भी", "ने", "तो", "हो", "कर",
}


def extract_emojis(text: str):
    """Return (text_without_emojis, list_of_emoji_chars)."""
    found = [c["emoji"] for c in emoji_lib.emoji_list(text)]
    cleaned = emoji_lib.replace_emoji(text, replace="")
    return cleaned, found


def normalize_unicode(text: str) -> str:
    """NFC normalisation; required for Devanagari script consistency."""
    return unicodedata.normalize("NFC", text)


def clean_text(text: str) -> str:
    text = str(text)
    text = URL_RE.sub(" ", text)
    text = HTML_RE.sub(" ", text)
    text = REPEATED_PUNCT_RE.sub(r"\1", text)
    text = normalize_unicode(text)
    text = MULTISPACE_RE.sub(" ", text).strip()
    return text


def remove_stopwords(tokens, lang: str = "en"):
    sw = _EN_STOPWORDS if lang == "en" else _HI_STOPWORDS
    return [t for t in tokens if t.lower() not in sw]


def tokenize_sentences(text: str, lang: str = "en"):
    """Sentence-level split, used for sentence-level emotion scoring
    (Section III.D, aggregated later via attention-weighted mean)."""
    try:
        if lang == "en":
            return sent_tokenize(text)
    except Exception:
        pass
    # simple Devanagari-aware fallback splitter (period, danda, exclaim, question)
    parts = re.split(r"(?<=[।.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def preprocess_review(raw_text: str, lang: str = "en"):
    """
    Full preprocessing pipeline for one review. Returns a dict with:
        clean_text   : text with emojis stripped, normalised, cleaned
        emojis       : list of emoji characters found (-> emoji_features.py)
        sentences    : sentence-level split of clean_text (-> emotion_features.py)
        tokens       : word tokens with stop-words removed (-> linguistic_features.py)
    """
    text_no_emoji, emojis = extract_emojis(raw_text)
    cleaned = clean_text(text_no_emoji)
    sentences = tokenize_sentences(cleaned, lang=lang)
    try:
        tokens = word_tokenize(cleaned) if lang == "en" else cleaned.split()
    except Exception:
        tokens = cleaned.split()
    tokens_ns = remove_stopwords(tokens, lang=lang)
    return {
        "clean_text": cleaned,
        "emojis": emojis,
        "sentences": sentences if sentences else [cleaned],
        "tokens": tokens,
        "tokens_no_stopwords": tokens_ns,
    }
