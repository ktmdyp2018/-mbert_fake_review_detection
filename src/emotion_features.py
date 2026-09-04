"""
Russell's Circumplex Model of emotion detection (Section III.D).

Pipeline per review i:
  1. Each sentence k is scored for (Val_i^(k), Aro_i^(k)) using a VAD
     (valence-arousal-dominance) lexicon [34].
  2. Sentence-level (Val, Aro) pairs are aggregated into review-level
     Em_i = (Val_i, Aro_i) via an ATTENTION-WEIGHTED mean (Eq. 4, revised):
         Val_i = sum_k beta_k * Val_i^(k),   Aro_i = sum_k beta_k * Aro_i^(k)
         beta_k = softmax( w^T tanh(W h_k) )
     where h_k is a pooled mBERT representation of sentence k. A
     simple-mean aggregator is also provided for ablation comparison.
  3. Emotion Intensity        EI_i  = sqrt(Val_i^2 + Aro_i^2)                  (Eq. 5)
     Emotion Polarity         EP_i  = atan2(Aro_i, Val_i) / pi                (Eq. 6)
     Sentiment_i (Eq. 7a)     = P(positive) - P(negative) in [-1, +1]
     Emotion Inconsistency    EIC_i = |Sentiment_i - Val_i|                   (Eq. 7)
  4. The complete emotion feature vector (Eq. 8) concatenates
     [Val_i, Aro_i, EI_i, EP_i, EIC_i, EmojiEM_i] -> 7-dim vector, where
     EmojiEM_i = (emoji_valence, emoji_arousal) from emoji_features.py.
"""

from typing import List, Sequence
import math
import numpy as np
import torch
import torch.nn as nn

from src.emoji_features import emojiem_vector

# --------------------------------------------------------------------------
# Compact built-in VAD lexicon (fallback). In production this should be
# replaced by the full NRC-VAD lexicon [34] (loadable via
# `load_external_lexicon`), but a small in-repo lexicon guarantees the
# pipeline runs fully offline / without external downloads.
# --------------------------------------------------------------------------
_EN_VAD = {
    "good": (0.7, 0.4), "great": (0.85, 0.6), "excellent": (0.9, 0.6),
    "amazing": (0.9, 0.7), "love": (0.85, 0.6), "best": (0.85, 0.55),
    "perfect": (0.9, 0.55), "happy": (0.8, 0.55), "satisfied": (0.65, 0.35),
    "bad": (-0.7, 0.4), "worst": (-0.9, 0.6), "terrible": (-0.85, 0.65),
    "poor": (-0.6, 0.35), "hate": (-0.85, 0.65), "disappointed": (-0.7, 0.45),
    "broken": (-0.6, 0.5), "defective": (-0.65, 0.5), "waste": (-0.7, 0.5),
    "fake": (-0.6, 0.55), "scam": (-0.8, 0.7), "cheap": (-0.35, 0.3),
    "awesome": (0.85, 0.65), "fantastic": (0.85, 0.65), "horrible": (-0.85, 0.7),
    "disgusting": (-0.85, 0.7), "recommend": (0.6, 0.35), "return": (-0.3, 0.4),
    "refund": (-0.35, 0.4), "damaged": (-0.65, 0.5), "excited": (0.75, 0.75),
    "angry": (-0.8, 0.8), "disappointing": (-0.65, 0.45), "wonderful": (0.85, 0.55),
    "useless": (-0.7, 0.4), "quality": (0.3, 0.2), "durable": (0.55, 0.25),
    "flimsy": (-0.55, 0.35), "sturdy": (0.55, 0.25), "comfortable": (0.6, 0.25),
    "fast": (0.4, 0.5), "slow": (-0.35, 0.3), "delay": (-0.4, 0.35),
    "delayed": (-0.4, 0.35), "ok": (0.1, 0.15), "okay": (0.1, 0.15),
    "fine": (0.2, 0.15), "average": (0.0, 0.15),
}

_HI_VAD = {
    "अच्छा": (0.7, 0.4), "बढ़िया": (0.75, 0.45), "शानदार": (0.85, 0.6),
    "बेहतरीन": (0.85, 0.55), "पसंद": (0.7, 0.5), "खराब": (-0.7, 0.4),
    "बेकार": (-0.7, 0.4), "घटिया": (-0.75, 0.5), "नकली": (-0.6, 0.55),
    "टूटा": (-0.6, 0.5), "धीमा": (-0.3, 0.25), "तेज": (0.35, 0.5),
    "संतुष्ट": (0.65, 0.35), "निराश": (-0.7, 0.45), "गुस्सा": (-0.8, 0.8),
    "बढ़िया": (0.75, 0.45), "मजबूत": (0.5, 0.25), "टिकाऊ": (0.55, 0.25),
}

_POS_WORDS = {w for w, va in {**_EN_VAD, **_HI_VAD}.items() if va[0] > 0.15}
_NEG_WORDS = {w for w, va in {**_EN_VAD, **_HI_VAD}.items() if va[0] < -0.15}


def load_external_lexicon(path: str, lang: str = "en"):
    """
    Optional hook: load a full VAD lexicon (e.g. NRC-VAD [34]) from a TSV
    file with columns `word, valence, arousal` (valence/arousal in [0,1]
    or [-1,1] - values are auto-rescaled to [-1,1]). Extends the in-repo
    lexicon in place.
    """
    import csv
    table = _EN_VAD if lang == "en" else _HI_VAD
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) < 3:
                continue
            word, val, aro = row[0], float(row[1]), float(row[2])
            if 0.0 <= val <= 1.0:      # rescale [0,1] -> [-1,1]
                val = 2 * val - 1
            if 0.0 <= aro <= 1.0:
                aro = 2 * aro - 1
            table[word.lower()] = (val, aro)


def sentence_valence_arousal(sentence: str, lang: str = "en") -> Sequence[float]:
    table = _EN_VAD if lang == "en" else _HI_VAD
    toks = sentence.lower().split()
    hits = [table[t] for t in toks if t in table]
    if not hits:
        return (0.0, 0.1)
    arr = np.array(hits, dtype=np.float32)
    return tuple(arr.mean(axis=0))


def sentence_sentiment_score(sentence: str, lang: str = "en") -> float:
    """
    Sentiment_i (Eq. 7a): continuous polarity in [-1, +1], defined as
    P(positive) - P(negative) using lexicon-based token counts as a
    lightweight, dependency-free proxy for a full 3-class softmax
    classifier (a fine-tuned mBERT sentiment head is a drop-in
    replacement in production - see docstring below).
    """
    toks = sentence.lower().split()
    pos = sum(1 for t in toks if t in _POS_WORDS)
    neg = sum(1 for t in toks if t in _NEG_WORDS)
    total = pos + neg
    if total == 0:
        return 0.0
    p_pos = pos / total
    p_neg = neg / total
    return float(p_pos - p_neg)  # in [-1, +1] by construction


class SentenceAttentionAggregator(nn.Module):
    """
    Single-head self-attention aggregator implementing beta_k =
    softmax(w^T tanh(W h_k)) used to combine sentence-level (Val, Aro)
    scores into a review-level Em_i (revised Eq. 4). `hidden_dim` should
    match the pooled sentence representation dimensionality (e.g. 768
    for an mBERT [CLS] pooled sentence embedding).
    """

    def __init__(self, hidden_dim: int = 768, attn_dim: int = 128):
        super().__init__()
        self.W = nn.Linear(hidden_dim, attn_dim, bias=True)
        self.w = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, sentence_hidden: torch.Tensor, mask: torch.Tensor = None):
        """
        sentence_hidden: (B, K, H) pooled per-sentence hidden states
        mask:            (B, K) 1 for real sentence, 0 for padding
        returns beta:    (B, K) attention weights summing to 1 per row
        """
        scores = self.w(torch.tanh(self.W(sentence_hidden))).squeeze(-1)  # (B, K)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        beta = torch.softmax(scores, dim=-1)
        return beta

    @staticmethod
    def aggregate(values: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        """values: (B, K) scalar per sentence (e.g. Val or Aro); returns (B,)."""
        return (values * beta).sum(dim=-1)


def simple_mean_aggregate(values: List[float]) -> float:
    """Baseline aggregator used only for the mean-vs-attention ablation."""
    return float(np.mean(values)) if values else 0.0


def compute_emotion_vector(sentences: List[str], emojis: List[str], lang: str = "en",
                            sentence_hidden: torch.Tensor = None,
                            aggregator: SentenceAttentionAggregator = None) -> np.ndarray:
    """
    End-to-end emotion feature extraction for one review.

    If `sentence_hidden` (pooled per-sentence mBERT vectors) and
    `aggregator` are supplied, attention-weighted aggregation is used;
    otherwise this falls back to simple mean-pooling (e.g. for fast
    CPU-only feature caching prior to training the attention module
    jointly with the FNN classifier).

    Returns a 7-dim np.float32 vector:
        [Val_i, Aro_i, EI_i, EP_i, EIC_i, EmojiVal_i, EmojiAro_i]
    """
    per_sentence_va = [sentence_valence_arousal(s, lang=lang) for s in sentences]
    per_sentence_sent = [sentence_sentiment_score(s, lang=lang) for s in sentences]

    vals = [v for v, a in per_sentence_va]
    aros = [a for v, a in per_sentence_va]

    if sentence_hidden is not None and aggregator is not None:
        with torch.no_grad():
            beta = aggregator(sentence_hidden.unsqueeze(0))  # (1, K)
            val_t = torch.tensor(vals, dtype=torch.float32).unsqueeze(0)
            aro_t = torch.tensor(aros, dtype=torch.float32).unsqueeze(0)
            val_i = aggregator.aggregate(val_t, beta).item()
            aro_i = aggregator.aggregate(aro_t, beta).item()
    else:
        val_i = simple_mean_aggregate(vals)
        aro_i = simple_mean_aggregate(aros)

    sentiment_i = simple_mean_aggregate(per_sentence_sent)   # Eq. 7a

    ei_i = math.sqrt(val_i ** 2 + aro_i ** 2)                # Eq. 5
    ep_i = math.atan2(aro_i, val_i) / math.pi                # Eq. 6, in [-1, 1]
    eic_i = abs(sentiment_i - val_i)                          # Eq. 7

    emoji_va = emojiem_vector(emojis)                        # (EmojiVal_i, EmojiAro_i)

    vec = np.array([val_i, aro_i, ei_i, ep_i, eic_i, emoji_va[0], emoji_va[1]],
                    dtype=np.float32)
    return vec
