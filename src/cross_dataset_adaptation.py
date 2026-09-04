"""
Cross-dataset / cross-language feature-learning module (Section III.F).

Two complementary mechanisms:

  (a) Weighted interpolation (Eq. 9a) - auxiliary shared-representation
      regulariser applied to paired English/Hindi review representations
      during training:
          F_cross = alpha * C_i^EN + (1 - alpha) * C_i^HI

  (b) Domain-adversarial training (Eq. 9b) via a Gradient Reversal Layer
      (GRL - Ganin & Lempitsky, 2015): a domain classifier D(.) is trained
      to predict the language label l_i from C_i, while the encoder is
      trained adversarially to fool D(.):
          L_total = L_cls - lambda * L_domain
          lambda(p) = 2 / (1 + exp(-gamma * p)) - 1        (p = training progress in [0,1])

For single-language inference (i.e. deploying on one language only, no
paired cross-lingual batch available), F_cross defaults to C_i itself
(alpha=1, no paired partner) so the fused vector dimensionality (Table
5-A) stays constant at every stage of training/inference.
"""

import torch
import torch.nn as nn
from torch.autograd import Function


class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def grad_reverse(x, lambd: float = 1.0):
    return GradientReversalFunction.apply(x, lambd)


def adversarial_lambda(progress: float, gamma: float = 10.0) -> float:
    """progress in [0, 1] = current_step / total_steps."""
    import math
    return 2.0 / (1.0 + math.exp(-gamma * progress)) - 1.0


class DomainClassifier(nn.Module):
    """D(.) in Eq. 9b: predicts the language/domain label from C_i."""

    def __init__(self, hidden_dim: int = 768, num_domains: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_domains),
        )

    def forward(self, C_i: torch.Tensor, lambd: float):
        reversed_feat = grad_reverse(C_i, lambd)
        return self.net(reversed_feat)


class CrossDatasetModule(nn.Module):
    """
    Bundles the interpolation regulariser (Eq. 9a) and the domain
    classifier used for adversarial training (Eq. 9b).
    """

    def __init__(self, hidden_dim: int = 768, alpha: float = 0.5, num_domains: int = 2):
        super().__init__()
        self.alpha = alpha
        self.domain_classifier = DomainClassifier(hidden_dim, num_domains)

    def interpolate(self, C_en: torch.Tensor, C_hi: torch.Tensor = None) -> torch.Tensor:
        """Eq. 9a. If no paired-language batch is available (single-language
        deployment), returns C_en unchanged so F_cross keeps a stable
        dimensionality throughout training and inference."""
        if C_hi is None:
            return C_en
        return self.alpha * C_en + (1 - self.alpha) * C_hi

    def domain_loss(self, C_i: torch.Tensor, domain_labels: torch.Tensor,
                     lambd: float) -> torch.Tensor:
        logits = self.domain_classifier(C_i, lambd)
        return nn.functional.cross_entropy(logits, domain_labels)
