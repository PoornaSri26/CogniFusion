"""
CogniFusion High-Accuracy Fusion Model (v3)

Architecture improvements for 90%+ accuracy:
  - ModalityEncoder: wider (128→64→embed_dim) with residual connections + LayerNorm
  - Temporal GRU layer: 3 consecutive 4s epochs→GRU→classification (sequence-aware)
  - Vector-gated cross-modal attention: per-dimension trust weights not a scalar
  - Deeper classifier with residual skip from both encoders
  - MixStyle feature augmentation during training (inter-subject domain shift)
  - Auxiliary heads on BOTH modalities + a GRU hidden state head
  - StandardizationNet: session-adaptive affine baseline correction
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ModalityEncoder(nn.Module):
    """Wider encoder with an internal residual connection."""
    def __init__(self, in_dim, hidden=128, out_dim=48, dropout=0.25):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden), nn.GELU(), nn.Dropout(dropout),
        )
        self.res = nn.Sequential(
            nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.GELU(), nn.Dropout(dropout * 0.6),
            nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden),
        )
        self.out = nn.Sequential(
            nn.GELU(),
            nn.Linear(hidden, out_dim), nn.GELU(),
        )
        self.dropout = nn.Dropout(dropout * 0.5)

    def forward(self, x):
        h = self.proj(x)
        h = h + self.res(h)          # residual: same dim
        h = self.dropout(h)
        return self.out(h)


class AttentionFusionNet(nn.Module):
    """
    High-accuracy attention fusion network.

    Key upgrades vs v2:
      - embed_dim=48 (was 24) — richer latent space
      - Per-dimension vector gate: w in R^embed_dim instead of a scalar
      - Deeper 3-layer classifier with skip from both raw embeddings
      - Optional sequential GRU path: when seq_len>1 epochs are batched together
        (the training loop passes windows of 3 epochs), the GRU provides
        temporal context — the single biggest gain on real data.
    """
    def __init__(self, eeg_dim, eye_dim, embed_dim=48, n_classes=3,
                 aux_heads=True, gate_type="vector", seq_len=1):
        super().__init__()
        self.embed_dim = embed_dim
        self.gate_type = gate_type
        self.seq_len = seq_len

        # --- Modality encoders (wider, with residual) ---
        self.eeg_enc = ModalityEncoder(eeg_dim, hidden=128, out_dim=embed_dim)
        self.eye_enc = ModalityEncoder(eye_dim, hidden=64,  out_dim=embed_dim)

        # --- Per-dimension vector gate ---
        gate_in = embed_dim * 2 + 4           # concat + moments
        gate_out = embed_dim if gate_type == "vector" else 1
        self.gate = nn.Sequential(
            nn.Linear(gate_in, 64), nn.GELU(),
            nn.Linear(64, 32), nn.GELU(),
            nn.Linear(32, gate_out), nn.Sigmoid(),
        )

        # --- Optional GRU for temporal sequence (seq_len > 1) ---
        self.use_gru = seq_len > 1
        gru_hidden = embed_dim if self.use_gru else 0
        if self.use_gru:
            self.gru = nn.GRU(input_size=embed_dim, hidden_size=embed_dim,
                              num_layers=2, batch_first=True, dropout=0.2)

        # --- Classifier: fused + skip-concat of both + (optional) GRU context ---
        clf_in = embed_dim + embed_dim * 2 + gru_hidden
        self.classifier = nn.Sequential(
            nn.Linear(clf_in, 128), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(128, 64),  nn.GELU(), nn.Dropout(0.15),
            nn.Linear(64, 32),   nn.GELU(),
            nn.Linear(32, n_classes),
        )

        # --- Auxiliary per-modality heads (regularizers) ---
        self.aux_heads = aux_heads
        if aux_heads:
            self.eeg_head = nn.Sequential(nn.Linear(embed_dim, 32), nn.GELU(),
                                          nn.Linear(32, n_classes))
            self.eye_head = nn.Sequential(nn.Linear(embed_dim, 32), nn.GELU(),
                                          nn.Linear(32, n_classes))

    @staticmethod
    def _moments(x):
        return torch.cat([x.mean(dim=1, keepdim=True), x.std(dim=1, keepdim=True)], dim=1)

    @staticmethod
    def mixstyle(x, alpha=0.4, eps=1e-6):
        """MixStyle: mix per-sample feature statistics across the batch.
        Forces the encoder to learn instance-normalized representations that
        are robust to inter-subject domain shift without needing subject labels."""
        if not x.requires_grad:   # eval mode: skip
            return x
        B = x.size(0)
        if B < 2:
            return x
        perm = torch.randperm(B, device=x.device)
        mu = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True) + eps
        x_norm = (x - mu) / var.sqrt()
        # Convex mix of style statistics
        lam = torch.distributions.Beta(alpha, alpha).sample((B, 1)).to(x.device)
        mu_mix  = lam * mu  + (1 - lam) * mu[perm]
        var_mix = lam * var + (1 - lam) * var[perm]
        return x_norm * var_mix.sqrt() + mu_mix

    def _forward_seq(self, eeg_x, eye_x, return_weight=False, modality_dropout=0.0):
        # Batched sequence path: eeg_x/eye_x are (B, T, F). Encode every
        # timestep, run the vector gate per step, then GRU over time and
        # classify from the final timestep's representation.
        B, T = eeg_x.shape[0], eeg_x.shape[1]
        n = B * T
        flat_e = eeg_x.reshape(n, -1)
        flat_y = eye_x.reshape(n, -1)

        if self.training:
            flat_e = self.mixstyle(flat_e)
            flat_y = self.mixstyle(flat_y)

        e_eeg = self.eeg_enc(flat_e)                     # (n, E)
        e_eye = self.eye_enc(flat_y)                     # (n, E)

        if self.training and modality_dropout > 0:
            drop_eeg = torch.rand(n, 1, device=e_eeg.device) < (modality_dropout / 2)
            drop_eye = (torch.rand(n, 1, device=e_eye.device) < (modality_dropout / 2)) & ~drop_eeg
            e_eeg = e_eeg * (~drop_eeg).float()
            e_eye = e_eye * (~drop_eye).float()

        gate_input = torch.cat([e_eeg, e_eye, self._moments(e_eeg), self._moments(e_eye)], dim=1)
        w = self.gate(gate_input)                        # (n, E)
        fused = w * e_eeg + (1.0 - w) * e_eye            # (n, E)

        E = fused.shape[1]
        fused3 = fused.view(B, T, E)
        gru_out, _ = self.gru(fused3)                    # (B, T, E)
        gru_ctx = gru_out[:, -1]                         # last timestep

        # last-timestep representations for the classifier + aux heads
        e_eeg_t = e_eeg.view(B, T, E)[:, -1]
        e_eye_t = e_eye.view(B, T, E)[:, -1]
        fused_t = fused.view(B, T, E)[:, -1]
        w_t = w.view(B, T, E)[:, -1]

        clf_inp = torch.cat([fused_t, e_eeg_t, e_eye_t, gru_ctx], dim=1)
        logits = self.classifier(clf_inp)

        w_summary = w_t.mean(dim=1, keepdim=True) if self.gate_type == "vector" else w_t
        if self.training and self.aux_heads:
            aux = (self.eeg_head(e_eeg_t), self.eye_head(e_eye_t))
            if return_weight:
                return logits, w_summary, aux
            return logits, aux
        if return_weight:
            return logits, w_summary
        return logits

    def forward(self, eeg_x, eye_x, return_weight=False, modality_dropout=0.0,
                seq_hidden=None):
        if eeg_x.dim() == 3 and self.use_gru:
            return self._forward_seq(eeg_x, eye_x, return_weight, modality_dropout)
        # MixStyle domain augmentation during training
        if self.training:
            eeg_x = self.mixstyle(eeg_x)
            eye_x = self.mixstyle(eye_x)

        e_eeg = self.eeg_enc(eeg_x)
        e_eye = self.eye_enc(eye_x)

        if self.training and modality_dropout > 0:
            n = e_eeg.shape[0]
            drop_eeg = torch.rand(n, 1, device=e_eeg.device) < (modality_dropout / 2)
            drop_eye = (torch.rand(n, 1, device=e_eye.device) < (modality_dropout / 2)) & ~drop_eeg
            e_eeg = e_eeg * (~drop_eeg).float()
            e_eye = e_eye * (~drop_eye).float()

        # Per-dimension vector gate
        gate_input = torch.cat([e_eeg, e_eye, self._moments(e_eeg), self._moments(e_eye)], dim=1)
        w = self.gate(gate_input)
        fused = w * e_eeg + (1.0 - w) * e_eye

        # Optional GRU temporal context
        gru_ctx = None
        new_hidden = None
        if self.use_gru:
            # fused shape: (B, embed_dim) → unsqueeze seq dim
            gru_in = fused.unsqueeze(1)   # (B, 1, embed_dim)
            gru_out, new_hidden = self.gru(gru_in, seq_hidden)
            gru_ctx = gru_out.squeeze(1)  # (B, embed_dim)
            clf_inp = torch.cat([fused, e_eeg, e_eye, gru_ctx], dim=1)
        else:
            clf_inp = torch.cat([fused, e_eeg, e_eye], dim=1)

        logits = self.classifier(clf_inp)

        # Scalar trust weight for telemetry / backward-compat
        w_summary = w.mean(dim=1, keepdim=True) if self.gate_type == "vector" else w

        if self.training and self.aux_heads:
            aux = (self.eeg_head(e_eeg), self.eye_head(e_eye))
            if return_weight:
                return logits, w_summary, aux
            return logits, aux
        if return_weight:
            if self.use_gru:
                return logits, w_summary, new_hidden
            return logits, w_summary
        return logits


class StandardizationNet(nn.Module):
    """Per-feature affine transform: (x - shift) * scale — session-adaptive
    baseline correction so live inference can re-center to the current user
    without touching the training-time sklearn scalers."""

    def __init__(self, dim, shift=None, scale=None):
        super().__init__()
        self.shift = nn.Parameter(
            torch.zeros(dim) if shift is None else torch.as_tensor(shift, dtype=torch.float32))
        self.scale = nn.Parameter(
            torch.ones(dim)  if scale is None else torch.as_tensor(scale, dtype=torch.float32))

    def forward(self, x):
        return (x - self.shift) * self.scale
