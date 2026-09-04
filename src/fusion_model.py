"""
Feature Fusion Layer + Feedforward Neural Network classifier
(Section III.G/H, Table 5-A).

F_i = C_i (768) ++ F_emotion (7) ++ F_linguistic (4) ++ F_cross (768)   -> 1547-dim

FNN:
    FC1: 1547 -> 768, ReLU, BatchNorm, Dropout 0.3
    FC2:  768 -> 256, ReLU, BatchNorm, Dropout 0.3
    FC3:  256 ->  64, ReLU, Dropout 0.2
    Out:   64 ->   2, Softmax
"""

import torch
import torch.nn as nn

from config import FEATURE_DIMS, FNN, TRAIN
from src.mbert_encoder import ContextualEncoder
from src.cross_dataset_adaptation import CrossDatasetModule, adversarial_lambda


class FusionFNN(nn.Module):
    """The classifier head only (useful when features are pre-extracted
    and cached, i.e. the common fast-training path where mBERT embeddings
    are computed once and reused across epochs)."""

    def __init__(self, fused_dim: int = None):
        super().__init__()
        fused_dim = fused_dim or FEATURE_DIMS.fused_dim

        self.fc1 = nn.Linear(fused_dim, FNN.fc1_out)
        self.bn1 = nn.BatchNorm1d(FNN.fc1_out)
        self.drop1 = nn.Dropout(FNN.dropout_fc1)

        self.fc2 = nn.Linear(FNN.fc1_out, FNN.fc2_out)
        self.bn2 = nn.BatchNorm1d(FNN.fc2_out)
        self.drop2 = nn.Dropout(FNN.dropout_fc2)

        self.fc3 = nn.Linear(FNN.fc2_out, FNN.fc3_out)
        self.drop3 = nn.Dropout(FNN.dropout_fc3)

        self.out = nn.Linear(FNN.fc3_out, FNN.num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in (self.fc1, self.fc2, self.fc3, self.out):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, fused_vec: torch.Tensor) -> torch.Tensor:
        x = self.drop1(self.bn1(torch.relu(self.fc1(fused_vec))))
        x = self.drop2(self.bn2(torch.relu(self.fc2(x))))
        x = self.drop3(torch.relu(self.fc3(x)))
        logits = self.out(x)          # softmax applied by the loss (CrossEntropyLoss)
        return logits


class FullPipelineModel(nn.Module):
    """
    End-to-end model: ContextualEncoder (mBERT) -> [C_i] -> concatenated
    with pre-computed F_emotion / F_linguistic / F_cross -> FusionFNN.

    Handcrafted features (emotion, linguistic, ABSA-derived, cross-dataset
    interpolation) are computed OUTSIDE this module (src/dataset.py,
    src/emotion_features.py, src/absa_features.py) and passed in as a
    tensor, since they do not require gradient-based fine-tuning; only
    C_i (mBERT) and the FNN head are trained end-to-end. This mirrors the
    paper's Fig. 1 architecture and keeps training tractable on a single
    24GB GPU (Table 5-B).
    """

    def __init__(self, backbone: str = None, class_weights: torch.Tensor = None):
        super().__init__()
        self.encoder = ContextualEncoder(backbone=backbone)
        self.cross_module = CrossDatasetModule(hidden_dim=self.encoder.hidden_size,
                                                alpha=TRAIN.cross_dataset_alpha)
        self.classifier = FusionFNN()
        self.class_weights = class_weights

    def forward(self, input_ids, attention_mask, handcrafted_feats: torch.Tensor,
                token_type_ids=None, paired_input_ids=None, paired_attention_mask=None):
        """
        handcrafted_feats: (B, 11) = F_emotion(7) ++ F_linguistic(4), already
                            computed upstream and moved to the same device.
        Returns: logits (B, 2), C_i (B, 768) [for the domain-adversarial loss]
        """
        C_i, _ = self.encoder(input_ids, attention_mask, token_type_ids)

        C_hi = None
        if paired_input_ids is not None:
            C_hi, _ = self.encoder(paired_input_ids, paired_attention_mask)
        F_cross = self.cross_module.interpolate(C_i, C_hi)

        fused = torch.cat([C_i, handcrafted_feats, F_cross], dim=-1)
        logits = self.classifier(fused)
        return logits, C_i

    def compute_loss(self, logits, labels, C_i=None, domain_labels=None,
                      progress: float = 0.0):
        cls_loss = nn.functional.cross_entropy(logits, labels, weight=self.class_weights)
        total = cls_loss
        domain_loss_val = torch.tensor(0.0, device=logits.device)
        if domain_labels is not None and C_i is not None:
            lambd = adversarial_lambda(progress, gamma=TRAIN.grl_gamma)
            domain_loss_val = self.cross_module.domain_loss(C_i, domain_labels, lambd)
            total = cls_loss - lambd * (-domain_loss_val)  # equivalent to L_cls - λ·L_domain
            # (GRL already negates the gradient; we ADD the domain loss here so the
            #  optimizer minimizes it directly, while GRL flips the sign of the
            #  gradient flowing back into the encoder - this is the standard
            #  Ganin & Lempitsky implementation pattern.)
            total = cls_loss + domain_loss_val
        return total, cls_loss, domain_loss_val
