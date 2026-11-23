# TRD - CogniFusion

## Stack
- Language: Python 3.10+
- DL: PyTorch (AttentionFusionNet: per-modality MLP encoders, vector gate, optional GRU)
- GBM: LightGBM (stacked with NN probabilities, 60/40 blend)
- Features: scipy/pandas/numpy; EEG band powers + gaze/pupil/scanpath features (168 dims)
- Eval: scikit-learn metrics, bootstrap CIs, conformal prediction, SHAP
- Data: pandas Parquet; SQLite (data/logs.db) + JSON session recordings
- UI: Streamlit dashboard; ONNX export for edge
- Repro: Colab notebook, requirements.txt, environment.yml, Dockerfile, CI (compile+import smoke)

## Non-functional
- Inference latency target <50 ms/epoch CPU
- Checkpoint self-contained (scalers+temperature embedded per fold)
- No subject leakage: all splits group by subject

## Constraints
- Synthetic dataset committed as features.parquet (621 KB); raw epochs and 1.5 GB real CSV never committed
- Muse EEG + webcam paths unverified without hardware - documented honestly
- Real-data feature build is hours-scale; parquet reuse path provided
