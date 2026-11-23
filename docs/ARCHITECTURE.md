# ARCHITECTURE - CogniFusion

## Layers
- `data/`: simulate_data.py (synthetic epochs), build_dataset_real.py (Zenodo CSV -> features)
- `src/eeg_features.py`, `src/eye_features.py`: raw epoch -> feature vectors (137 + 31)
- `src/model.py`: AttentionFusionNet - EEG/eye MLP encoders -> vector gate -> fused embed -> classifier (+aux heads, optional GRU seq path, optional domain_head for GRL)
- `src/train.py`: subject-wise splits, adaptive_subject_norm, scalers, focal/GCE loss, mixup/cutmix/noise/dropout aug, SWA/EMA candidates, temperature scaling, optional LightGBM stacking -> self-contained checkpoint (folds + scalers + temp + label_classes)
- `src/realtime_pipeline.py`: checkpoint loader -> per-epoch preprocess/scale/predict -> EMA smoothing -> DataLogger + SessionManager
- `src/data_management.py`: DataLogger (SQLite), SessionManager (JSON), analytics helpers
- `dashboard.py`: Streamlit UI reading logs.db + dashboard_state.json
- `CogniFusion_Colab.ipynb`: end-to-end research notebook

## Data flow
raw epoch -> band-power/gaze features -> adaptive subject norm -> StandardScaler -> NN probs + GBM probs (0.6/0.4) -> temperature -> EMA smooth -> log/predict

## Decisions
- Checkpoint is self-contained so Colab/pipeline need only the .pt
- GBM stays per-epoch even when NN uses seq windows (blend guarded)
- All splits group by subject - no leakage
- Third-party calls: none at inference; dataset download only in notebook
