# BACKEND_SCHEMA - CogniFusion

Two stores: SQLite `data/logs.db` (DataLogger) and JSON files under `sessions/` (SessionManager).

## SQLite tables (src/data_management.py)
- `epochs`: id, session_id, timestamp, predicted_class, confidence, probabilities (JSON), eeg_trust, eye_trust, features_hash - one row per classified epoch
- `sessions`: session_id PK, user_id, start/end timestamps, metadata JSON, status
- `alerts`: id, session_id, timestamp, alert_type, severity, message, confidence

## JSON sessions/
`<session_id>.json`: {session_id, user_id, metadata, epochs: [{timestamp, probs, prediction, gate_weights}], summary}

## Relationships
sessions 1-N epochs; sessions 1-N alerts; epochs/alerts reference session_id.

## Rules
- All writes via DataLogger/SessionManager APIs - never raw SQL elsewhere
- probabilities stored as JSON arrays aligned to `label_classes`
- No user PII beyond optional user_id; raw signals never persisted (features only)
