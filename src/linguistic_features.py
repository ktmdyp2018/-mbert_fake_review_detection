
from typing import List
import numpy as np

try:
    import textstat
    _HAS_TEXTSTAT = True
except Exception:
    _HAS_TEXTSTAT = False


def review_length(tokens: List[str]) -> int:
    return len(tokens)


def repetition_ratio(tokens: List[str]) -> float:
    """1 - (unique tokens / total tokens); higher => more repetitive text,
    a signal often associated with templated/farm-written fake reviews."""
    if not tokens:
        return 0.0
    unique = len(set(t.lower() for t in tokens))
    return 1.0 - (unique / len(tokens))


def readability_score(text: str, lang: str = "en") -> float:
    """Flesch Reading-Ease for English; a lightweight average-sentence-
    length / average-word-length proxy for Hindi (no validated Hindi
    Flesch formula exists), rescaled to a comparable ~0-100 range."""
    if lang == "en" and _HAS_TEXTSTAT and text.strip():
        try:
            return float(textstat.flesch_reading_ease(text))
        except Exception:
            pass
    words = text.split()
    if not words:
        return 0.0
    sentences = max(text.count("।") + text.count("."), 1)
    avg_sent_len = len(words) / sentences
    avg_word_len = float(np.mean([len(w) for w in words]))
    # crude proxy: shorter sentences/words => higher "readability"
    score = 100 - (avg_sent_len * 1.5) - (avg_word_len * 4.0)
    return float(np.clip(score, 0, 100))


def compute_linguistic_vector(tokens: List[str], clean_text: str, asv: float,
                               lang: str = "en") -> np.ndarray:
    length = review_length(tokens)
    rep = repetition_ratio(tokens)
    read = readability_score(clean_text, lang=lang) / 100.0  # scale to [0,1]
    return np.array([length, rep, read, asv], dtype=np.float32)
