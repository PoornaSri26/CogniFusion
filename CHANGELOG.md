# Changelog

## Unreleased — research toolkit + docs

### Training flags (all verified end-to-end)
- `--seq-len N`: GRU temporal sequence windows (non-overlapping, per-subject)
- `--ablate eeg|eye`: single-modality neural baselines
- `--grl/--grl-lambda/--grl-weight`: domain-adversarial subject invariance
- `--swa`: stochastic weight averaging (kept only if it beats best val F1)
- `--ema-decay`: weight EMA during training
- `--teacher/--kd-alpha/--kd-temp`: knowledge distillation
- `--feat-noise`, `--feat-dropout`, `--cutmix`: feature robustness augmentation
- `--gce-q`: generalized cross-entropy for noisy labels
- `--sam-rho`: sharpness-aware minimization
- `--rdrop`: R-Drop consistency regularization
- `--cv 30`: leave-one-subject-out evaluation

### Evaluation & interpretability (notebook, 100 cells)
- Split-conformal + per-class conformal prediction sets
- Risk-coverage curves, bootstrap CIs, ordinal metrics (QWK/MAE)
- Per-fold ROC-AUC, permutation importance, SHAP (TreeSHAP on GBM)
- MC-dropout uncertainty, few-shot subject adaptation, deep ensembles
- TTA, per-class temperature scaling, magnitude pruning, INT8 quantization
- Deployment benchmarks (params, checkpoint size, ONNX Runtime latency)

### Documentation
- `docs/` framework: PRD, TRD, APP_FLOW, IMPLEMENTATION_PLAN, BACKEND_SCHEMA,
  TESTING, DESIGN_SYSTEM, ARCHITECTURE, SECURITY, CODE_STYLE — each with
  AI-agent instruction blocks
- `AGENTS.md` agent operating manual; README documentation index
- BACKEND_SCHEMA verified against actual SQLite schema

### Fixes
- Eval loader passes `seq_len` for sequence checkpoints
- `train.py` F-namespace shadowing in KD path; EMA dtype handling
- Preprocessing-cell f-string corruption repaired

## v1.0.0 — 2026-11-22
- Initial public release: EEG+eye cognitive-load fusion (low/medium/high)
- AttentionFusionNet with vector-gated cross-modal fusion, focal loss, mixup,
  per-subject median/MAD normalization, temperature scaling, EMA smoothing
- Optional LightGBM stacking (--use-gbm)
- 5-fold subject-wise CV training, ensemble checkpoint format v3
- Real-time pipeline (synthetic + Muse/webcam), Streamlit dashboard
- Colab notebook: full pipeline + real Zenodo dataset path
- CI: compile check + import smoke test
