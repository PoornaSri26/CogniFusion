# BACKEND_SCHEMA - CogniFusion

Two stores: SQLite `data/logs.db` (DataLogger) and JSON files under `sessions/` (SessionManager).

## SQLite tables (verified against `src/data_management.py`)

### `epochs` — one row per classified epoch
| Column | Type | Notes |
|--------|------|-------|
| id | INTEGER PK AUTOINCREMENT | |
| timestamp | TEXT NOT NULL | ISO timestamp |
| prediction | TEXT NOT NULL | predicted class label |
| confidence | REAL NOT NULL | max class probability |
| eeg_trust | REAL NOT NULL | gate weight for EEG modality |
| eye_trust | REAL NOT NULL | gate weight for eye modality |
| processing_time | REAL NOT NULL | inference seconds |
| features_json | TEXT | optional feature dump |
| metadata_json | TEXT | optional extras |

### `sessions`
| Column | Type | Notes |
|--------|------|-------|
| id | INTEGER PK AUTOINCREMENT | |
| session_id | TEXT UNIQUE NOT NULL | |
| user_id | TEXT NOT NULL | |
| start_time | TEXT NOT NULL | |
| end_time | TEXT | nullable until close |
| epoch_count | INTEGER DEFAULT 0 | |
| metadata_json | TEXT | |

### `alerts`
| Column | Type | Notes |
|--------|------|-------|
| id | INTEGER PK AUTOINCREMENT | |
| timestamp | TEXT NOT NULL | |
| alert_type | TEXT NOT NULL | |
| severity | TEXT NOT NULL | |
| message | TEXT NOT NULL | |
| session_id | TEXT | FK-ish link to sessions.session_id |
| metadata_json | TEXT | |

## JSON sessions/ (SessionManager)
`<session_id>.json`: `{session_id, user_id, metadata, epochs: [{timestamp, probs, prediction, gate_weights}], summary}`

## Relationships
sessions 1-N epochs (via session_id in metadata/alert rows); sessions 1-N alerts.

## Rules
- All writes via DataLogger/SessionManager APIs — never raw SQL elsewhere
- probabilities live in SessionManager JSON, not the epochs table
- No user PII beyond user_id; raw signals never persisted (features only)

---
## For AI agents
Generate DB access only through DataLogger/SessionManager. Never raw SQL elsewhere. Keep checkpoint format and JSON session schema backward compatible. If the schema changes, update this table first.
