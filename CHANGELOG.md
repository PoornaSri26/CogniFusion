# Changelog

## v1.0.0 — 2026-11-22
- Initial public release: EEG+eye cognitive-load fusion (low/medium/high)
- AttentionFusionNet with vector-gated cross-modal fusion, focal loss, mixup,
  per-subject median/MAD normalization, temperature scaling, EMA smoothing
- Optional LightGBM stacking (--use-gbm)
- 5-fold subject-wise CV training, ensemble checkpoint format v3
- Real-time pipeline (synthetic + Muse/webcam), Streamlit dashboard
- Colab notebook: full pipeline + real Zenodo dataset path
- CI: compile check + import smoke test
