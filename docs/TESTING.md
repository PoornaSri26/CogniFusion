# TESTING - CogniFusion

## Types
- Unit: feature extractors (eeg_features, eye_features), scaler paths, window builder
- Integration: train_fold end-to-end, checkpoint save/load, pipeline predict
- Smoke: `python -m compileall src data dashboard.py` (CI does this)
- Manual: 15-epoch training runs per new flag; realtime_pipeline synthetic mode

## Key flows that must not break
1. `train.py` -> checkpoint loads in `load_ensemble` + `realtime_pipeline`
2. Subject-wise split -> no subject appears in both train and test
3. seq_len>1 checkpoints -> pipeline threads seq_hidden correctly
4. DataLogger -> epochs/sessions/alerts queryable; SessionManager JSON round-trip

## Test cases
| ID | Scenario | Expected | Status |
|----|----------|----------|--------|
| TC-01 | train --epochs 15 --cv 0 | checkpoint saved, cv_summary printed | Pass |
| TC-02 | pipeline load fusion_model.pt | predictions + probs emitted | Pass |
| TC-03 | pipeline load seq checkpoint | seq_hidden threaded, 3-tuple handled | Pass |
| TC-04 | --ablate eye | eye features zeroed, GBM disabled | Pass |
| TC-05 | --grl/--swa/--ema/--teacher/--gce/--sam/--rdrop/--feat-*/--cutmix | trains without error | Pass |
| TC-06 | notebook cells compile (non-magic lines) | no SyntaxError | Pass |

## Commands
`python -m compileall src data dashboard.py` ; `python src/train.py --features data/processed/features.parquet --epochs 15 --cv 0`
