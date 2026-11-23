# Contributing to CogniFusion

Thanks for your interest! Please read this before opening a PR.

## Setup
```bash
pip install -r requirements.txt
python data/simulate_data.py
python src/build_dataset.py
python src/train.py --epochs 60 --cv 0
python src/realtime_pipeline.py --eeg synthetic --eye synthetic --cycles 10
```

## Guidelines
- Keep commits focused; conventional-commit style preferred (`feat:`, `fix:`, `docs:`)
- All new features need a short note in README or the Colab notebook
- Run `python -m compileall src data dashboard.py` before pushing
- Never commit `data/raw/`, `data/processed/` (except `features.parquet`), `models/` (except `fusion_model.pt`), or secrets

## Reporting issues
Use the issue templates. Include OS, Python version, and the failing command output.
