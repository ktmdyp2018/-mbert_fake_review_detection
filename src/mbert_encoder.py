"""
Contextual embedding module (Section III.C, Table 7-A).

Wraps a Hugging Face multilingual encoder (default: bert-base-multilingual-
cased -> 104 languages, 119,547 WordPiece vocab, 12 layers / 12 heads /
hidden 768). The first `num_frozen_layers` transformer blocks are frozen;
only the remaining top layers are fine-tuned jointly with the FNN
classifier (Table 5-B). Swappable with XLM-R / IndicBERT / MuRIL for the
backbone-substitution study (Table 7-A).
"""

import torch
import torch.nn as nn

from config import BACKBONE_CHECKPOINTS, TRAIN


class _MockTokenizer:
    """Whitespace/character-level tokenizer used ONLY when the real
    checkpoint cannot be downloaded (e.g. offline CI / sandboxed smoke
    tests). NEVER use this for real experiments - it exists purely so
    the rest of the pipeline (training loop, fusion, ablation,
    statistics) can be exercised end-to-end without network access."""

    def __init__(self, vocab_size: int = 30000):
        self.vocab_size = vocab_size
        self.pad_token_id = 0

    def __call__(self, texts, padding=True, truncation=True, max_length=256,
                 return_tensors="pt"):
        import torch as _torch
        seqs = []
        for t in texts:
            ids = [abs(hash(tok)) % (self.vocab_size - 2) + 2 for tok in str(t).split()][:max_length]
            if not ids:
                ids = [2]
            seqs.append(ids)
        max_len = max(len(s) for s in seqs)
        input_ids = _torch.zeros(len(seqs), max_len, dtype=_torch.long)
        attn = _torch.zeros(len(seqs), max_len, dtype=_torch.long)
        for i, s in enumerate(seqs):
            input_ids[i, :len(s)] = _torch.tensor(s, dtype=_torch.long)
            attn[i, :len(s)] = 1
        return {"input_ids": input_ids, "attention_mask": attn}


class _MockEncoderBackbone(nn.Module):
    """Tiny randomly-initialised transformer standing in for mBERT when
    offline. Produces embeddings of the correct shape (hidden_size=768)
    so every downstream module (fusion, FNN, GRL, ablation) can be
    smoke-tested without internet access."""

    def __init__(self, hidden_size: int = 768, vocab_size: int = 30000, num_layers: int = 2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        enc_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=8,
                                                dim_feedforward=1024, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.config = type("cfg", (), {"hidden_size": hidden_size})()

        class _Layers(list):
            pass
        self.layer = _Layers(self.transformer.layers)

        class _Embeddings(nn.Module):
            def __init__(self, embed):
                super().__init__()
                self.word_embeddings = embed

        self.embeddings = _Embeddings(self.embed)

    class _Encoder(nn.Module):
        def __init__(self, outer):
            super().__init__()
            self.layer = outer.layer

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None):
        x = self.embed(input_ids)
        pad_mask = (attention_mask == 0) if attention_mask is not None else None
        h = self.transformer(x, src_key_padding_mask=pad_mask)

        class _Out:
            pass
        out = _Out()
        out.last_hidden_state = h
        return out

    @property
    def encoder(self):
        return self._Encoder(self)


class ContextualEncoder(nn.Module):
    def __init__(self, backbone: str = None, num_frozen_layers: int = None,
                 max_seq_len: int = None, allow_mock_fallback: bool = True):
        super().__init__()
        backbone = backbone or TRAIN.backbone
        ckpt = BACKBONE_CHECKPOINTS[backbone]
        self.backbone_name = backbone
        self.max_seq_len = max_seq_len or TRAIN.max_seq_len
        self.is_mock = False

        try:
            from transformers import AutoModel, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(ckpt)
            self.encoder = AutoModel.from_pretrained(ckpt)
            self.hidden_size = self.encoder.config.hidden_size
        except Exception as e:
            if not allow_mock_fallback:
                raise
            print(f"[mbert_encoder] WARNING: could not download '{ckpt}' "
                  f"({type(e).__name__}: {e}). Falling back to a randomly-"
                  f"initialised MOCK encoder for offline smoke-testing only. "
                  f"Real experiments MUST run with network access to "
                  f"huggingface.co so the true pretrained weights are used.")
            self.is_mock = True
            self.hidden_size = 768
            self.tokenizer = _MockTokenizer()
            self.encoder = _MockEncoderBackbone(hidden_size=self.hidden_size)

        self._freeze_layers(num_frozen_layers if num_frozen_layers is not None
                             else TRAIN.num_frozen_layers)

    def _freeze_layers(self, num_frozen_layers: int):
        # Embeddings are always frozen along with the bottom N transformer
        # layers; only the top (num_layers - N) layers are fine-tuned.
        for p in self.encoder.embeddings.parameters():
            p.requires_grad = False

        layer_module = None
        for attr in ("encoder.layer", "encoder.layers"):
            obj = self.encoder
            try:
                for part in attr.split("."):
                    obj = getattr(obj, part)
                layer_module = obj
                break
            except AttributeError:
                continue
        if layer_module is None:
            return  # unknown architecture: skip selective freezing gracefully

        for i, layer in enumerate(layer_module):
            requires_grad = i >= num_frozen_layers
            for p in layer.parameters():
                p.requires_grad = requires_grad

    def tokenize(self, texts, device="cpu"):
        enc = self.tokenizer(
            texts, padding=True, truncation=True,
            max_length=self.max_seq_len, return_tensors="pt",
        )
        return {k: v.to(device) for k, v in enc.items()}

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        out = self.encoder(**kwargs)
        cls = out.last_hidden_state[:, 0, :]     # [CLS] pooled representation, C_i
        return cls, out.last_hidden_state

    def encode_texts(self, texts, device="cpu"):
        """Convenience: tokenize + forward -> (B, hidden_size) CLS embeddings."""
        enc = self.tokenize(texts, device=device)
        with torch.no_grad():
            cls, _ = self.forward(**enc)
        return cls
