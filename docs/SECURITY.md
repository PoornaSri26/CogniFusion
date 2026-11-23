# SECURITY - CogniFusion

## Data
- No secrets in repo: `.env*` ignored; no API keys required anywhere
- EEG/eye data is biometric-adjacent: raw epochs and real CSV stay gitignored; only derived features.parquet committed
- sessions/ and data/logs.db contain user_id optionally - do not commit real user data

## Code
- No auth layer (local research tool) - do not expose pipeline/dashboard to untrusted networks without adding auth
- torch.load uses weights_only=False for legacy checkpoints - acceptable locally; never load untrusted .pt files
- Downloaded dataset verified by header/schema before processing

## Secrets handling
- Never commit .env, tokens, or credentials; use env vars if integrations added
- `git credential` used only for pushes, never stored in files

## Pre-launch checklist
- [ ] .gitignore covers data/raw, sessions, logs.db
- [ ] No hardcoded paths with usernames in committed code (use Path/relative)
- [ ] Checkpoint files scanned for embedded absolute paths before sharing
