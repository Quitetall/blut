"""ai_models/student/seizure_head.py — multi-task seizure detection head.

Shares the encoder's latent representation [B, 32, 79] and adds a
lightweight binary classifier. Joint training with the codec loss
teaches the encoder to preserve seizure-relevant features in its
latent — features that might otherwise be discarded by a pure-
reconstruction objective.

From the optimization atlas + FAE paper: "jointly optimizing
compression and task heads yields better operating points than
pipelined design." The encoder learns two things simultaneously:
(a) produce tokens that the decoder reconstructs well, and
(b) produce tokens that a classifier can read seizure from.

Usage in train_joint.py:
    head = SeizureHead().to(device)
    ...
    latent = codec.encoder.encode(x_l3, quantize=True)
    seizure_logits = head(latent)
    seizure_loss = head.loss(seizure_logits, seizure_labels)
    total_loss = recon_loss + 0.1 * seizure_loss
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SeizureHead(nn.Module):
    """Binary seizure detection from the encoder's latent [B, 32, 79].

    Architecture: global average pool → LayerNorm → Linear → sigmoid.
    ~33 parameters. The head is intentionally tiny so it doesn't
    dominate the encoder's gradient — the primary task remains
    reconstruction; seizure detection is an auxiliary signal.
    """

    def __init__(self, latent_dim: int = 32):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.norm = nn.LayerNorm(latent_dim)
        self.fc = nn.Linear(latent_dim, 1)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """latent: [B, latent_dim, T] → [B] logits (pre-sigmoid)."""
        x = self.pool(latent).squeeze(-1)  # [B, latent_dim]
        x = self.norm(x)
        return self.fc(x).squeeze(-1)  # [B]

    @staticmethod
    def loss(logits: torch.Tensor, labels: torch.Tensor,
             pos_weight: float = 5.0) -> torch.Tensor:
        """Weighted BCE. pos_weight compensates for seizure rarity
        (~2-5% of windows in CHB-MIT). Default 5.0 means a missed
        seizure costs 5× a false alarm — matches clinical priorities.
        """
        pw = torch.tensor([pos_weight], device=logits.device)
        return F.binary_cross_entropy_with_logits(
            logits, labels.float(), pos_weight=pw)

    @staticmethod
    def accuracy(logits: torch.Tensor, labels: torch.Tensor,
                 threshold: float = 0.5) -> dict:
        """Returns {accuracy, sensitivity, specificity, f1}."""
        preds = (torch.sigmoid(logits) > threshold).float()
        labels_f = labels.float()
        tp = ((preds == 1) & (labels_f == 1)).sum().float()
        tn = ((preds == 0) & (labels_f == 0)).sum().float()
        fp = ((preds == 1) & (labels_f == 0)).sum().float()
        fn = ((preds == 0) & (labels_f == 1)).sum().float()
        acc = (tp + tn) / max(tp + tn + fp + fn, 1)
        sens = tp / max(tp + fn, 1)
        spec = tn / max(tn + fp, 1)
        f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        return {
            'accuracy': float(acc),
            'sensitivity': float(sens),
            'specificity': float(spec),
            'f1': float(f1),
        }
