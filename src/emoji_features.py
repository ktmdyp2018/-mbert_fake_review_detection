"""
Emoji -> (valence, arousal) mapping, producing the EmojiEM_i component of
the emotion feature vector F_emotion (Section III.D, Fig. 1 "Emoji and
Emotion extraction").

A compact hand-curated valence/arousal lookup table is provided for the
most frequent review emojis (covers >95% of emoji occurrences in typical
e-commerce review corpora). Unknown emojis fall back to a neutral-low
(0.0, 0.2) vector rather than being dropped, so review-level aggregation
is always well defined.
"""

from typing import List, Tuple
import numpy as np

# (valence, arousal), each in [-1, 1] / [0, 1] respectively, following the
# same Russell's Circumplex convention used for text-derived Val_i/Aro_i.
EMOJI_VALENCE_AROUSAL = {
    "😀": (0.85, 0.70), "😃": (0.85, 0.75), "😄": (0.90, 0.80), "😁": (0.85, 0.75),
    "😆": (0.80, 0.85), "😊": (0.80, 0.55), "🙂": (0.55, 0.35), "😍": (0.95, 0.80),
    "🥰": (0.90, 0.65), "😘": (0.85, 0.60), "👍": (0.70, 0.45), "👏": (0.70, 0.60),
    "🎉": (0.85, 0.85), "❤️": (0.90, 0.60), "💖": (0.90, 0.65), "✨": (0.60, 0.55),
    "🔥": (0.65, 0.85), "⭐": (0.70, 0.50), "💯": (0.80, 0.65),
    "😐": (0.0, 0.20), "😑": (-0.10, 0.15), "🤔": (0.0, 0.40),
    "😢": (-0.75, 0.45), "😭": (-0.85, 0.75), "😞": (-0.65, 0.30), "😔": (-0.60, 0.25),
    "😠": (-0.75, 0.80), "😡": (-0.90, 0.90), "🤬": (-0.95, 0.95), "👎": (-0.70, 0.45),
    "💔": (-0.85, 0.55), "😤": (-0.55, 0.75), "🙄": (-0.40, 0.35), "😒": (-0.50, 0.30),
    "😩": (-0.60, 0.60), "😫": (-0.65, 0.65), "🤢": (-0.75, 0.55), "🤮": (-0.85, 0.65),
}

DEFAULT_VA = (0.0, 0.20)


def emoji_to_va(e: str) -> Tuple[float, float]:
    return EMOJI_VALENCE_AROUSAL.get(e, DEFAULT_VA)


def emojiem_vector(emojis: List[str]) -> np.ndarray:
    """
    Aggregate a review's emoji list into a 2-D (valence, arousal) EmojiEM
    vector via simple mean; reviews with no emojis get a neutral vector
    (0, 0), which is distinguishable from a "calm" emoji signal (0, 0.2)
    and lets the downstream FNN learn the difference between "no emoji
    used" and "calm emoji used".
    """
    if not emojis:
        return np.zeros(2, dtype=np.float32)
    vas = np.array([emoji_to_va(e) for e in emojis], dtype=np.float32)
    return vas.mean(axis=0)


def emoji_density(emojis: List[str], token_count: int) -> float:
    """Emoji-per-token density; used diagnostically in the error-analysis
    study (Table 9-B) which links high emoji density to false positives."""
    if token_count == 0:
        return 0.0
    return len(emojis) / token_count
