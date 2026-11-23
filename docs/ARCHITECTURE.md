# ARCHITECTURE - CogniFusion

## Layers
- `data/`: simulate_data.py (synthetic epochs), build_dataset_real.py (Zenodo CSV -> features)
- `src/eeg_features.py`, `src/eye_features.py`: raw epoch -> feature vectors (137 + 31)
- `src/model.py`: AttentionFusionNet - EEG/eye MLP encoders -> vector gate -> fused embed -> classifier (+aux heads, optional GRU seq path, optional domain_head for GRL)
- `src/train.py`: subject-wise splits, adaptive_subject_norm, scalers, focal/GCE loss, mixup/cutmix/noise/dropout aug, SWA/EMA candidates, temperature scaling, optional LightGBM stacking -> self-contained checkpoint (folds + scalers + temp + label_classes)
- `src/realtime_pipeline.py`: checkpoint loader -> per-epoch preprocess/scale/predict -> EMA smoothing -> DataLogger + SessionManager
- `src/data_management.py`: DataLogger (SQLite), SessionManager (JSON), analytics helpers
- `src/advanced_analytics.py`, `src/dashboard_enhancements.py`, `src/testing_framework.py`: analytics, dashboard widgets, self-test harness
- `src/eeg_stream.py`, `src/webcam_eyetracker.py`, `src/realtime_features.py`: hardware/streaming adapters (BrainFlow EEG, MediaPipe gaze)
- `src/model_improvements.py`: extended model utilities
- `data/simulate_data.py`: synthetic epoch generator; `src/build_dataset.py`: feature table builder; `src/build_dataset_real.py`: Zenodo CSV -> features_real.parquet
- `dashboard.py`: Streamlit UI reading logs.db + dashboard_state.json
- `CogniFusion_Colab.ipynb`: end-to-end research notebook

## Data flow
raw epoch -> band-power/gaze features -> adaptive subject norm -> StandardScaler -> NN probs + GBM probs (0.6/0.4) -> temperature -> EMA smooth -> log/predict

## Decisions
- Checkpoint is self-contained so Colab/pipeline need only the .pt
- GBM stays per-epoch even when NN uses seq windows (blend guarded)
- All splits group by subject - no leakage
- Third-party calls: none at inference; dataset download only in notebook

---
## For AI agents
Put new code in the layer matching its job. Keep the checkpoint self-contained and the non-seq inference path backward compatible. Update this doc when structure changes.
