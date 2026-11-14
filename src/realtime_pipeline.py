"""
Real-time inference loop (v3): pulls a 4s EEG epoch + a 4s eye-tracking epoch
in parallel, extracts the same features used at training time, and runs them
through the trained AttentionFusionNet ensemble, writing the result to a
small JSON state file that the Streamlit dashboard polls.

v3 improvements:
  - INTEGRATED ADVANCED FEATURES: Includes connectivity, entropy, microsaccades,
    pupillometry analytics from enhanced feature extractors
  - REAL-TIME MONITORING: Break recommendations, stress detection, performance metrics
  - DATA LOGGING: Comprehensive logging to database and file system
  - ALERT SYSTEM: Real-time alerting for high cognitive load and stress states
  - SESSION MANAGEMENT: Enhanced session recording and user profile integration

v2 improvements:
  - SESSION-ADAPTIVE CALIBRATION: a rolling window of recent raw feature
    vectors is used to re-center/scale features to the CURRENT session
    (median/MAD, exactly matching training's adaptive_subject_norm). This
    bridges the distribution shift between training recordings and the live
    session — the single biggest cause of "model works offline, fails live".
  - ENSEMBLE INFERENCE: the checkpoint holds K fold models; probabilities are
    the average across folds (lower variance, better calibration).
  - Warmup: until CALIB_MIN_EPOCHS epochs are seen, predictions are shown but
    flagged low-confidence via "calibrated": false in the state file.

Run modes:
  --eeg synthetic   -> load-modulated synthetic EEG (works with zero hardware, default)
  --eeg muse        -> BrainFlow MUSE_S_BOARD (needs a real Muse headset)
  --eye synthetic   -> reuses data/simulate_data.py's eye-epoch generator (default)
  --eye webcam      -> live MediaPipe webcam eye tracking (needs a real camera)

Default run (`python src/realtime_pipeline.py`) needs NO hardware at all --
both streams are synthetic, only the fusion model + feature pipeline are real.
"""
import sys
import json
import time
import argparse
from collections import deque
from pathlib import Path
from datetime import datetime, timezone

sys.path.append(str(Path(__file__).parent))
import numpy as np
import pandas as pd
import torch

from eeg_features import extract_eeg_features
from eye_features import extract_eye_features
from model import AttentionFusionNet
from eeg_stream import EEGStreamer

# Import new advanced features
from advanced_analytics import AdvancedAnalytics
from realtime_features import RealTimeFeaturesManager
from data_management import DataLogger, SessionManager, convert_numpy_types
from dashboard_enhancements import AlertSystem

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATE_PATH = PROJECT_ROOT / "dashboard_state.json"
CHANNEL_NAMES = ["AF7", "AF8", "TP9", "TP10"]
FS_EEG = 256
CALIB_WINDOW = 30      # rolling epochs used for session-adaptive calibration
CALIB_MIN_EPOCHS = 3   # need at least this many before calibration kicks in


def project_path(path):
    """Resolve relative paths consistently, regardless of the launch directory."""
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_model(model_path="models/fusion_model.pt"):
    """Load the self-contained ensemble checkpoint (v2). Falls back to the
    legacy v1 layout (separate scalers.pkl) for old checkpoints."""
    model_path = project_path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)

    if ckpt.get("version") == 2:
        folds = []
        for rec in ckpt["folds"]:
            model = AttentionFusionNet(eeg_dim=len(ckpt["eeg_cols"]),
                                       eye_dim=len(ckpt["eye_cols"]),
                                       embed_dim=ckpt.get("embed_dim", 24),
                                       n_classes=len(ckpt["label_classes"]))
            model.load_state_dict(rec["model_state"])
            model.eval()
            folds.append({"model": model, "scaler_eeg": rec["scaler_eeg"],
                          "scaler_eye": rec["scaler_eye"]})
        return {"folds": folds, "eeg_cols": ckpt["eeg_cols"],
                "eye_cols": ckpt["eye_cols"], "label_classes": ckpt["label_classes"],
                "legacy": False}

    # Fallback for v1 models - try to load anyway with default dimensions
    try:
        print(f"Loading legacy model from {model_path}")
        model = AttentionFusionNet(eeg_dim=78, eye_dim=18, embed_dim=24, n_classes=3)
        model.load_state_dict(ckpt)
        model.eval()
        
        # Create dummy scalers
        from sklearn.preprocessing import StandardScaler
        scaler_eeg = StandardScaler()
        scaler_eye = StandardScaler()
        
        return {
            "folds": [{"model": model, "scaler_eeg": scaler_eeg, "scaler_eye": scaler_eye}],
            "eeg_cols": [f"eeg__{k}" for k in range(78)],
            "eye_cols": [f"eye__{k}" for k in range(18)],
            "label_classes": ["low", "medium", "high"],
            "legacy": True
        }
    except Exception as e:
        raise ValueError(
            f"Failed to load model from {model_path}: {e}. "
            f"Please ensure the model file exists and is compatible.")


def history_record(record):
    """Keep historical state small; raw traces belong only to the latest epoch."""
    fields = ("timestamp", "cycle", "prediction", "probs", "eeg_trust_weight",
              "confidence", "calibrated")
    return {field: record.get(field) for field in fields}


def load_existing_history():
    """Preserve recent records when the live pipeline is restarted."""
    try:
        state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        history = state.get("history", [])
        return [history_record(record) for record in history if isinstance(record, dict)]
    except (json.JSONDecodeError, OSError):
        return []


def write_state_atomic(payload: dict, attempts: int = 4):
    """Write dashboard_state.json atomically, tolerating the Windows quirk where
    os.replace fails with PermissionError while another process (e.g. the
    Streamlit dashboard) holds a concurrent read handle on the destination."""
    tmp_state_path = STATE_PATH.with_suffix(".tmp")
    tmp_state_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for attempt in range(attempts):
        try:
            tmp_state_path.replace(STATE_PATH)
            return
        except PermissionError:
            if attempt == attempts - 1:
                # Last resort: write in place. Rare, but better than crashing
                # a long-running live pipeline over a transient file lock.
                STATE_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                tmp_state_path.unlink(missing_ok=True)
            else:
                time.sleep(0.05 * (attempt + 1))


def synthetic_eye_epoch(load_hint=0.5, epoch_sec=4, fs_eye=60):
    """Fallback eye-epoch generator when no webcam is available -- mirrors
    data/simulate_data.py's structure so the feature schema matches exactly."""
    rng = np.random.default_rng()
    n = epoch_sec * fs_eye
    pupil = 3.0 + 1.0 * load_hint + rng.normal(0, 0.15, n)
    fixation_flags = rng.random(n) < (0.55 + 0.25 * load_hint)
    blink_flags = rng.random(n) < max(0.01, 0.05 - 0.03 * load_hint)
    gaze_x = np.cumsum(rng.normal(0, 0.5 + 1.5 * (1 - load_hint), n))
    gaze_y = np.cumsum(rng.normal(0, 0.5 + 1.5 * (1 - load_hint), n))
    return pd.DataFrame({"t": np.arange(n) / fs_eye, "pupil_mm": pupil,
                         "is_fixation": fixation_flags, "is_blink": blink_flags,
                         "gaze_x": gaze_x, "gaze_y": gaze_y})


def synthetic_eeg_epoch(load_hint=0.5, epoch_sec=4, fs=FS_EEG):
    """Load-modulated synthetic EEG, mirroring data/simulate_data.py's
    generator (alpha drops, theta+beta rise with load) plus noise, so live
    predictions actually track the drifting load_hint instead of sitting flat."""
    t = np.arange(0, epoch_sec, 1 / fs)
    rng = np.random.default_rng()
    sig = np.zeros((len(CHANNEL_NAMES), len(t)))
    for ch in range(len(CHANNEL_NAMES)):
        alpha = (1.2 - 0.7 * load_hint) * np.sin(2 * np.pi * 10 * t)
        theta = (0.4 + 0.9 * load_hint) * np.sin(2 * np.pi * 6 * t)
        beta = (0.3 + 0.6 * load_hint) * np.sin(2 * np.pi * 22 * t)
        noise = rng.normal(0, 0.6, len(t))
        sig[ch] = alpha + theta + beta + noise
    return sig


def session_calibrate(raw_window: pd.DataFrame, current: pd.Series):
    """Session-adaptive calibration from the rolling raw-feature window.

    Recentered/robustly scaled version of `current` using the window's own
    median/MAD — identical math to training's adaptive_subject_norm, applied
    to the live stream. Returns None while the window is too small.
    """
    if len(raw_window) < CALIB_MIN_EPOCHS:
        return None
    med = raw_window.median()
    mad = (raw_window - med).abs().median()
    scale = 1.4826 * mad
    scale = scale.where(scale > 1e-8, raw_window.std()).where(scale > 1e-8, 1.0)
    med = med.fillna(0.0)
    scale = scale.fillna(1.0)
    return ((current - med) / scale).fillna(0.0)


def run(eeg_source="synthetic", eye_source="synthetic", n_cycles=None,
        epoch_sec=4, model_path="models/fusion_model.pt"):
    bundle = load_model(model_path)
    eeg_cols, eye_cols = bundle["eeg_cols"], bundle["eye_cols"]
    label_classes = bundle["label_classes"]
    folds = bundle["folds"]

    # Initialize advanced features
    analytics = AdvancedAnalytics()
    realtime_features = RealTimeFeaturesManager()
    data_logger = DataLogger()
    session_manager = SessionManager()
    alert_system = AlertSystem()
    
    # Start session
    session_id = session_manager.create_session('default_user', {'mode': f'{eeg_source}_{eye_source}'})
    print(f"Started session: {session_id}")

    use_real_eeg_board = eeg_source != "synthetic"
    eeg_streamer = None
    if use_real_eeg_board:
        from brainflow.board_shim import BoardIds
        eeg_streamer = EEGStreamer(board_id=BoardIds.MUSE_S_BOARD.value)
        eeg_streamer.start()

    webcam_tracker = None
    if eye_source == "webcam":
        from webcam_eyetracker import WebcamEyeTracker
        webcam_tracker = WebcamEyeTracker()

    history = load_existing_history()
    raw_eeg_window = deque(maxlen=CALIB_WINDOW)   # raw (unscaled) EEG feature rows
    raw_eye_window = deque(maxlen=CALIB_WINDOW)
    cycle = 0
    sustained_high_count = 0
    smoothed_probs = None
    ema_alpha = 0.65  # Temporal smoothing weight across consecutive epochs
    print("Real-time pipeline started. Ctrl+C to stop.")
    print(f"Model: {model_path} | ensemble folds: {len(folds)} | "
          f"calibration window: {CALIB_WINDOW} epochs")
    print("Advanced features enabled: analytics, real-time monitoring, data logging, alerts")
    try:
        while n_cycles is None or cycle < n_cycles:
            cycle += 1
            epoch_start_time = time.time()
            
            # --- EEG epoch ---
            if eeg_streamer is not None:
                eeg_raw = eeg_streamer.get_epoch(epoch_sec=epoch_sec)
                n_ch = min(len(CHANNEL_NAMES), eeg_raw.shape[0])
                eeg_epoch = eeg_raw[:n_ch]
            else:
                # drift the load hint on a slow sine so the demo visibly moves
                # between low/medium/high over time instead of sitting flat
                load_hint = max(0.0, min(1.0, 0.5 + 0.5 * np.sin(cycle / 5)))
                eeg_epoch = synthetic_eeg_epoch(load_hint=load_hint, epoch_sec=epoch_sec)
            eeg_feat_dict = {f"eeg__{k}": float(v) for k, v in
                             extract_eeg_features(eeg_epoch,
                                                  channel_names=CHANNEL_NAMES[:eeg_epoch.shape[0]]).items()}

            # --- Eye epoch ---
            if webcam_tracker is not None:
                eye_df = webcam_tracker.stream_epoch(epoch_sec=epoch_sec)
                if eye_df is None or eye_df.empty:
                    eye_df = synthetic_eye_epoch(epoch_sec=epoch_sec)
            else:
                eye_df = synthetic_eye_epoch(load_hint=load_hint, epoch_sec=epoch_sec)
            eye_feat_dict = {f"eye__{k}": float(v) for k, v in
                             extract_eye_features(eye_df).items()}

            # --- align to training column order ---
            eeg_vec = pd.Series({c: eeg_feat_dict.get(c, 0.0) for c in eeg_cols}, dtype=float)
            eye_vec = pd.Series({c: eye_feat_dict.get(c, 0.0) for c in eye_cols}, dtype=float)

            # --- session-adaptive calibration (after warmup) ---
            raw_eeg_window.append(eeg_vec)
            raw_eye_window.append(eye_vec)
            calibrated = len(raw_eeg_window) >= CALIB_MIN_EPOCHS
            if calibrated:
                eeg_cal = session_calibrate(pd.DataFrame(list(raw_eeg_window)), eeg_vec)
                eye_cal = session_calibrate(pd.DataFrame(list(raw_eye_window)), eye_vec)
            else:
                eeg_cal, eye_cal = eeg_vec, eye_vec

            # --- ensemble inference: average probabilities across folds ---
            prob_sum = np.zeros(len(label_classes))
            w_sum = 0.0
            eeg_in = eeg_cal.reindex(eeg_cols).fillna(0.0).to_numpy()[None, :]
            eye_in = eye_cal.reindex(eye_cols).fillna(0.0).to_numpy()[None, :]
            with torch.no_grad():
                for fold in folds:
                    xe = fold["scaler_eeg"].transform(pd.DataFrame(eeg_in, columns=eeg_cols))
                    ye = fold["scaler_eye"].transform(pd.DataFrame(eye_in, columns=eye_cols))
                    logits, w = fold["model"](torch.tensor(xe, dtype=torch.float32),
                                              torch.tensor(ye, dtype=torch.float32),
                                              return_weight=True)
                    prob_sum += torch.softmax(logits, dim=1).numpy()[0]
                    w_val = w.mean().item() if hasattr(w, "mean") else w.item()
                    w_sum += float(w_val)
            probs = prob_sum / len(folds)
            # EMA temporal smoothing over consecutive epochs to filter transient noise
            if smoothed_probs is None:
                smoothed_probs = probs
            else:
                smoothed_probs = ema_alpha * probs + (1.0 - ema_alpha) * smoothed_probs

            trust = w_sum / len(folds)
            pred_idx = int(smoothed_probs.argmax())
            
            # Calculate processing time
            processing_time = time.time() - epoch_start_time
            
            # Track sustained high load
            if label_classes[pred_idx] == 'high':
                sustained_high_count += 1
            else:
                sustained_high_count = 0
            
            # Process through real-time features
            realtime_summary = realtime_features.process_epoch(
                prediction=label_classes[pred_idx],
                confidence=float(probs[pred_idx]),
                eeg_trust=float(trust),
                eye_trust=1.0 - float(trust),
                processing_time=processing_time
            )
            
            # Check for alerts
            alerts = alert_system.check_alerts(
                prediction=label_classes[pred_idx],
                confidence=float(probs[pred_idx]),
                epoch_count=cycle,
                sustained_high_count=sustained_high_count
            )
            
            # Log data
            epoch_data = {
                'prediction': label_classes[pred_idx],
                'confidence': float(probs[pred_idx]),
                'eeg_trust': float(trust),
                'eye_trust': 1.0 - float(trust),
                'processing_time': processing_time,
                'eeg_features': eeg_feat_dict,
                'eye_features': eye_feat_dict,
                'realtime_features': realtime_summary,
                'alerts': alerts
            }
            data_logger.log_epoch(epoch_data, session_id)
            session_manager.add_epoch_to_session(session_id, epoch_data)

            record = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "cycle": cycle,
                "prediction": label_classes[pred_idx],
                "probs": {label_classes[i]: round(float(probs[i]), 5)
                          for i in range(len(label_classes))},
                "eeg_trust_weight": round(float(trust), 5),
                "confidence": round(float(probs[pred_idx]), 5),
                "calibrated": bool(calibrated),
                "eeg_trace": [round(float(v), 4) for v in eeg_epoch[0][:FS_EEG * 2]],
                "gaze": {
                    "x": [round(float(v), 3) for v in eye_df["gaze_x"].tolist()],
                    "y": [round(float(v), 3) for v in eye_df["gaze_y"].tolist()],
                },
                "features": {
                    "eeg": {k: round(float(v), 5) for k, v in eeg_feat_dict.items()},
                    "eye": {k: round(float(v), 5) for k, v in eye_feat_dict.items()},
                },
                "advanced_features": {
                    "break_recommendation": realtime_summary.get('break_recommendation', {}),
                    "stress_assessment": realtime_summary.get('stress_assessment', {}),
                    "performance_metrics": realtime_summary.get('performance_metrics', {}),
                    "active_alerts": alerts
                }
            }
            
            # Convert numpy types to Python types for JSON serialization
            record = convert_numpy_types(record)
            history.append(history_record(record))
            history = history[-120:]  # keep last 120 epochs (~8 min at 4s/epoch)

            write_state_atomic({"latest": record, "history": history})
            calib_tag = "" if calibrated else "  [calibrating]"
            alert_tag = f" [ALERTS: {len(alerts)}]" if alerts else ""
            print(f"[{cycle:04d}] {record['prediction']:>6s} | "
                  f"probs={ {k: round(v, 2) for k, v in record['probs'].items()} } | "
                  f"eeg_trust={record['eeg_trust_weight']:.2f}{calib_tag}{alert_tag}")

    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        # End session
        session_summary = session_manager.close_session(session_id)
        print(f"Session ended: {session_summary.get('epoch_count', 0)} epochs processed")
        
        # Print comprehensive summary
        comprehensive_summary = realtime_features.get_comprehensive_summary()
        print("Real-time features summary:")
        print(f"  Break statistics: {comprehensive_summary.get('break_statistics', {})}")
        print(f"  Performance: {comprehensive_summary.get('performance_summary', {})}")
        print(f"  Stress trend: {comprehensive_summary.get('stress_trend', {})}")
        
        # Print database statistics
        db_stats = data_logger.get_statistics()
        print(f"Database statistics: {db_stats}")
        
        if eeg_streamer is not None:
            eeg_streamer.stop()
        if webcam_tracker is not None:
            webcam_tracker.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eeg", choices=["synthetic", "muse"], default="synthetic")
    parser.add_argument("--eye", choices=["synthetic", "webcam"], default="synthetic")
    parser.add_argument("--cycles", type=int, default=None, help="Number of 4s cycles to run (default: infinite)")
    parser.add_argument("--model", default="models/fusion_model.pt",
                        help="Ensemble checkpoint (default: models/fusion_model.pt; "
                             "use models/fusion_model_real.pt for the real-data model)")
    args = parser.parse_args()
    run(eeg_source=args.eeg, eye_source=args.eye, n_cycles=args.cycles,
        model_path=args.model)
