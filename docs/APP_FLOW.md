# APP_FLOW - CogniFusion

CogniFusion is primarily a library + notebook; "screens" map to pipeline stages and dashboard views.

## Primary journey (Colab)
Clone -> Open in Colab -> GPU runtime -> Run all:
Setup -> Data (prebuilt parquet skip) -> Features (skip if exists) -> EDA -> Train (5-fold CV + GBM) -> Curves -> Eval -> Diagnostics -> Ablations/Experiments -> Exports -> DB persistence -> Real-data optional -> Summary checklist

## Real-time journey
`python src/realtime_pipeline.py` -> hardware or synthetic mode -> per-epoch: preprocess -> scale -> NN+GBM probs -> EMA smooth -> log to SQLite/JSON -> alert if sustained high load -> dashboard reads state.

## States
- Loading: feature extraction progress, training fold progress
- Empty: no sessions logged -> dashboard shows empty state
- Error: hardware disconnect -> graceful fallback message; missing checkpoint -> clear error
- Success: prediction + confidence + gate weights displayed/logged

## Edge cases
- Missing modality at runtime -> pipeline requires both; document failure mode
- Subject drift -> adaptive_subject_norm per session
- Low confidence -> abstain flag available via threshold sweep results
