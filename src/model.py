"""
Attention-Weighted Late Fusion Network v2 for EEG + Eye-Tracking.

Architecture:
  - Deeper per-modality encoders (in -> 64 -> 32 -> embed) with BatchNorm,
    GELU and dropout.
  - Gating network that sees BOTH modality embeddings plus the modality
    embeddings' own summary statistics, and outputs a per-sample trust
    weight w in [0, 1] for EEG vs eye.
  - Residual fusion: fused = w*eeg + (1-w)*eye, then a small MLP that also
    receives the raw concat [e_eeg, e_eye] (skip path), so the classifier can
    recover information the scalar gate squashed away.
  - Auxiliary single-modality heads trained with (optionally) small weights:
    they regularize each encoder to be predictive on its own, which keeps the
    gate honest instead of letting one modality lazily dominate.

Also provides StandardizationNet: a differentiable per-feature affine layer
used to apply session-adaptive baseline correction at inference time (see
realtime_pipeline.py) without touching the sklearn scalers.
"""
import torch
import torch.nn as nn


class ModalityEncoder(nn.Module):
    def __init__(self, in_dim, hidden=64, out_dim=24, dropout=0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.BatchNorm1d(hidden // 2), nn.GELU(), nn.Dropout(dropout * 0.5),
            nn.Linear(hidden // 2, out_dim), nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class AttentionFusionNet(nn.Module):
    def __init__(self, eeg_dim, eye_dim, embed_dim=24, n_classes=3,
                 aux_heads=True, gate_type="vector"):
        super().__init__()
        self.embed_dim = embed_dim
        self.gate_type = gate_type
        self.eeg_enc = ModalityEncoder(eeg_dim, out_dim=embed_dim)
        self.eye_enc = ModalityEncoder(eye_dim, out_dim=embed_dim)

        # gate input: both embeddings + their per-feature means/stds (4 extra)
        gate_in = embed_dim * 2 + 4
        gate_out = embed_dim if gate_type == "vector" else 1
        self.gate = nn.Sequential(
            nn.Linear(gate_in, 32), nn.GELU(),
            nn.Linear(32, 16), nn.GELU(),
            nn.Linear(16, gate_out), nn.Sigmoid(),
        )

        # classifier sees fused vector + skip concat of both embeddings
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim + embed_dim * 2, 64), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(64, 32), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(32, n_classes),
        )

        # auxiliary single-modality heads (regularizers)
        self.aux_heads = aux_heads
        if aux_heads:
            self.eeg_head = nn.Linear(embed_dim, n_classes)
            self.eye_head = nn.Linear(embed_dim, n_classes)

    @staticmethod
    def _moments(x):
        return torch.cat([x.mean(dim=1, keepdim=True), x.std(dim=1, keepdim=True)], dim=1)

    def forward(self, eeg_x, eye_x, return_weight=False, modality_dropout=0.0):
        e_eeg = self.eeg_enc(eeg_x)
        e_eye = self.eye_enc(eye_x)

        if self.training and modality_dropout > 0:
            # zero out one whole modality per sample with prob modality_dropout/2
            n = e_eeg.shape[0]
            drop_eeg = torch.rand(n, 1, device=e_eeg.device) < (modality_dropout / 2)
            drop_eye = (torch.rand(n, 1, device=e_eye.device) < (modality_dropout / 2)) & ~drop_eeg
            e_eeg = e_eeg * (~drop_eeg).float()
            e_eye = e_eye * (~drop_eye).float()

        w = self.gate(torch.cat([e_eeg, e_eye, self._moments(e_eeg), self._moments(e_eye)], dim=1))
        fused = w * e_eeg + (1.0 - w) * e_eye
        logits = self.classifier(torch.cat([fused, e_eeg, e_eye], dim=1))

        # Scalar trust weight for logging/backward-compat
        w_summary = w.mean(dim=1, keepdim=True) if self.gate_type == "vector" else w

        if self.training and self.aux_heads:
            aux = (self.eeg_head(e_eeg), self.eye_head(e_eye))
            if return_weight:
                return logits, w_summary, aux
            return logits, aux
        if return_weight:
            return logits, w_summary
        return logits


class StandardizationNet(nn.Module):
    """Per-feature affine transform: (x - shift) * scale, applied to the raw
    (unscaled) feature vector. Used for session-adaptive baseline correction:
    shift/scale come from a short calibration window instead of the training
    population, bridging the distribution shift between recording sessions."""

    def __init__(self, dim, shift=None, scale=None):
        super().__init__()
        self.shift = nn.Parameter(torch.zeros(dim) if shift is None else torch.as_tensor(shift, dtype=torch.float32))
        self.scale = nn.Parameter(torch.ones(dim) if scale is None else torch.as_tensor(scale, dtype=torch.float32))

    def forward(self, x):
        return (x - self.shift) * self.scale
