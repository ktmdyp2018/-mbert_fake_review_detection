"""
Aspect-Based Sentiment Analysis (Section III.E).

  1. A fixed taxonomy of six product aspects is used:
     {price, quality, delivery, packaging, service, durability}.
  2. Each sentence in a review is matched to an aspect using cosine
     similarity (>= 0.6) between the sentence embedding and an
     aspect-anchor phrase embedding, both computed with the multilingual
     Sentence-BERT model 'paraphrase-multilingual-MiniLM-L12-v2' [ABSA
     reproducibility revision].
  3. Aspect-level sentiment is scored with a lexicon-based polarity
     function (a fine-tuned mBERT 3-class sentiment head is a drop-in
     replacement in production; see `AspectSentimentScorer.set_classifier`).
  4. Aspect Sentiment Variance (ASV_i) = variance of the aspect-level
     scores across only the aspects actually DETECTED in review i
     (undetected aspects are excluded, not imputed as neutral).
"""

from typing import Dict, List
import numpy as np

from config import ABSA
from src.emotion_features import sentence_sentiment_score

_ASPECT_ANCHORS = {
    "price": "The price and value for money of this product.",
    "quality": "The build quality and material quality of this product.",
    "delivery": "The shipping speed and delivery experience.",
    "packaging": "The packaging and box condition on arrival.",
    "service": "The customer service and seller support experience.",
    "durability": "How durable and long-lasting this product is.",
}


class AspectMatcher:
    """
    Wraps a multilingual Sentence-BERT encoder for aspect matching. The
    actual model download requires network access to huggingface.co; a
    lightweight bag-of-words cosine-similarity fallback is used
    automatically if the SentenceTransformer package/model is
    unavailable, so the pipeline still runs end-to-end offline.
    """

    def __init__(self, model_name: str = None, threshold: float = None):
        self.threshold = threshold or ABSA.similarity_threshold
        self.model_name = model_name or ABSA.sentence_transformer_ckpt
        self._st_model = None
        self._aspect_embeds = None
        self._use_fallback = False
        self._init_backend()

    def _init_backend(self):
        try:
            from sentence_transformers import SentenceTransformer
            self._st_model = SentenceTransformer(self.model_name)
            anchors = list(_ASPECT_ANCHORS.values())
            self._aspect_embeds = self._st_model.encode(anchors, normalize_embeddings=True)
        except Exception:
            # Offline / no-network fallback: bag-of-words Jaccard-like matcher
            self._use_fallback = True
            self._aspect_keywords = {
                "price": {"price", "cost", "money", "value", "expensive", "cheap",
                          "कीमत", "दाम", "सस्ता", "महंगा"},
                "quality": {"quality", "material", "build", "गुणवत्ता", "मटेरियल"},
                "delivery": {"delivery", "shipping", "arrived", "late", "डिलीवरी", "समय"},
                "packaging": {"packaging", "box", "package", "पैकिंग", "डिब्बा"},
                "service": {"service", "support", "seller", "customer", "सेवा", "सर्विस"},
                "durability": {"durable", "durability", "broke", "lasted", "टिकाऊ", "मजबूत"},
            }

    def match_sentence(self, sentence: str) -> List[str]:
        """Return the list of aspects (0 or more) detected in a sentence."""
        if self._use_fallback:
            toks = set(sentence.lower().split())
            return [a for a, kws in self._aspect_keywords.items() if toks & kws]

        emb = self._st_model.encode([sentence], normalize_embeddings=True)[0]
        sims = self._aspect_embeds @ emb
        aspects = list(_ASPECT_ANCHORS.keys())
        return [aspects[i] for i, s in enumerate(sims) if s >= self.threshold]


class AspectSentimentScorer:
    """Aspect-level sentiment scoring; defaults to the lexicon-based
    `sentence_sentiment_score` (Eq. 7a) but a fine-tuned mBERT classifier
    can be injected via `set_classifier`."""

    def __init__(self):
        self._classifier_fn = None

    def set_classifier(self, fn):
        """fn(text: str) -> float in {-1, 0, +1} or continuous polarity."""
        self._classifier_fn = fn

    def score(self, sentence: str, lang: str = "en") -> float:
        if self._classifier_fn is not None:
            return self._classifier_fn(sentence)
        return sentence_sentiment_score(sentence, lang=lang)


def compute_absa(sentences: List[str], lang: str,
                  matcher: AspectMatcher, scorer: AspectSentimentScorer) -> Dict:
    """
    Returns:
        aspect_scores : {aspect: mean_sentiment} for aspects actually
                         detected in this review
        asv           : Aspect Sentiment Variance (float). 0.0 if fewer
                         than 2 distinct aspects are detected.
    """
    aspect_hits: Dict[str, List[float]] = {}
    for sent in sentences:
        aspects = matcher.match_sentence(sent)
        if not aspects:
            continue
        sent_score = scorer.score(sent, lang=lang)
        for a in aspects:
            aspect_hits.setdefault(a, []).append(sent_score)

    aspect_scores = {a: float(np.mean(v)) for a, v in aspect_hits.items()}
    if len(aspect_scores) >= 2:
        asv = float(np.var(list(aspect_scores.values())))
    else:
        asv = 0.0

    return {"aspect_scores": aspect_scores, "asv": asv}
