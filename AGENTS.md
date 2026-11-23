# AGENTS.md - CogniFusion

## Project
Multimodal EEG+eye cognitive-load classifier. PyTorch + LightGBM, SQLite/JSON logging, Streamlit dashboard, Colab notebook. Read `docs/` before coding.

## Commands
- Train: `python src/train.py --features data/processed/features.parquet --epochs 200 --cv 5 --use-gbm`
- Variants: `--seq-len N --ablate eeg|eye --grl --swa --ema-decay D --teacher CKPT --feat-noise S --feat-dropout F --cutmix F --gce-q Q --sam-rho R --rdrop W --cv 30`
- Pipeline: `python src/realtime_pipeline.py` (synthetic or hardware sources)
- Verify: `python -m compileall src data dashboard.py`

## Rules (always)
- Read docs/ARCHITECTURE.md + docs/CODE_STYLE.md before editing
- Subject-wise splits only - never random row splits
- Checkpoints must stay self-contained (scalers+temp per fold)
- Verify every change: compile + short training run + notebook compile check

## Ask first
- Adding dependencies, changing checkpoint schema, touching data/raw or committed artifacts

## Never
- Commit raw data, sessions/, logs.db, secrets, or >5 MB binaries without asking
- Claim hardware validation without hardware
- Break the non-seq inference path (backward compat)

## Doc map
PRD=scope - TRD=stack - APP_FLOW=journeys - IMPLEMENTATION_PLAN=phases - BACKEND_SCHEMA=DB - TESTING=cases - DESIGN_SYSTEM=UI - ARCHITECTURE=code map - SECURITY=rules - CODE_STYLE=conventions
