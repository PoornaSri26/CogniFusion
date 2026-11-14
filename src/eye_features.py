"""
Eye-tracking feature extraction (v3).

Base features (any source with gaze_x/gaze_y):
  - gaze dispersion (x/y std), gaze speed mean/std/percentiles (p50/p90/p95)
  - acceleration (mean |a|, std), peak speed
  - saccade rate (thresholded jumps)
  - BCEA — bivariate contour ellipse area, a standard fixation-dispersion metric
  - path efficiency — net displacement / total path length
  - directional-change rate — velocity reversals per second (searching behavior)

Fixation/blink features (flag/pupil columns present — synthetic & webcam sources):
  - fixation ratio, blink rate, pupil mean/std
  - NEW: Microsaccade detection and analysis
  - NEW: Pupillometry trend analysis (dilation/constriction patterns)
  - NEW: Pupil variability metrics

Target-tracking features (real dataset with stimulus_x/stimulus_y):
  - fixation error to target mean/std/median/p90
  - target velocity + gaze-target speed correlation (tracking effort)
  - lag-0 correlation between gaze and target displacement (tracking quality)

One extractor serves every data source; downstream column alignment handles
missing keys via fill values.
"""
import numpy as np
import pandas as pd
from scipy.signal import welch

# percentile levels used for speed/error summaries
_PCTLS = (50, 90, 95)


def _speeds(gx, gy, t):
    """Per-sample speed (units/s) with dt sanitization."""
    dt = np.diff(t)
    good = np.isfinite(dt) & (dt > 1e-4)
    med = np.median(dt[good]) if good.any() else 1.0
    dt = np.where(good, dt, med)
    return np.sqrt(np.diff(gx) ** 2 + np.diff(gy) ** 2) / dt, dt


def _bcea(gx, gy):
    """Bivariate contour ellipse area (95%) in units^2."""
    s_x, s_y = np.std(gx), np.std(gy)
    s_xy = np.cov(gx, gy, bias=True)[0, 1] if len(gx) > 1 else 0.0
    det = max(s_x ** 2 * s_y ** 2 - s_xy ** 2, 1e-12)
    return float(2 * np.pi * np.sqrt(det) * 1.96 ** 2)


def _base_gaze_feats(gx, gy, t):
    feats = {
        "gaze_x_std": float(np.std(gx)),
        "gaze_y_std": float(np.std(gy)),
    }
    speed, dt = _speeds(gx, gy, t)
    feats["gaze_speed_mean"] = float(np.mean(speed))
    feats["gaze_speed_std"] = float(np.std(speed))
    for p in _PCTLS:
        feats[f"gaze_speed_p{p}"] = float(np.percentile(speed, p))
    accel = np.abs(np.diff(speed)) / dt[1:]
    feats["accel_mean"] = float(np.mean(accel))
    feats["accel_std"] = float(np.std(accel))
    feats["gaze_speed_peak"] = float(np.max(speed))

    # saccade rate: jumps above the epoch's own mean+std threshold
    jump = np.sqrt(np.diff(gx) ** 2 + np.diff(gy) ** 2)
    thresh = np.mean(jump) + np.std(jump)
    duration = max(t[-1] - t[0], 1e-6)
    feats["saccade_rate_hz"] = float((jump > thresh).sum() / duration)

    # BCEA + path efficiency
    feats["bcea"] = _bcea(gx, gy)
    path_len = float(np.sum(jump))
    net = float(np.hypot(gx[-1] - gx[0], gy[-1] - gy[0]))
    feats["path_efficiency"] = net / (path_len + 1e-8)

    # directional-change rate: sign flips of velocity x-component per second
    vx = np.diff(gx)
    flips = np.sum(np.sign(vx[1:]) != np.sign(vx[:-1]))
    feats["dir_change_rate"] = float(flips / duration)
    return feats


def _target_tracking_feats(gx, gy, t, sx, sy):
    feats = {}
    err = np.hypot(gx - sx, gy - sy)
    feats["fixation_error_mean"] = float(np.mean(err))
    feats["fixation_error_std"] = float(np.std(err))
    feats["fixation_error_median"] = float(np.median(err))
    feats["fixation_error_p90"] = float(np.percentile(err, 90))

    t_vel = np.sqrt(np.diff(sx) ** 2 + np.diff(sy) ** 2) / _speeds(sx, sy, t)[1]
    feats["target_speed_mean"] = float(np.mean(t_vel))
    if len(t_vel) > 2 and np.std(t_vel) > 1e-9 and np.std(np.sqrt(np.diff(gx) ** 2 + np.diff(gy) ** 2)) > 1e-9:
        feats["gaze_target_speed_corr"] = float(np.corrcoef(
            np.sqrt(np.diff(gx) ** 2 + np.diff(gy) ** 2), t_vel)[0, 1])
    else:
        feats["gaze_target_speed_corr"] = 0.0

    # lag-0 displacement correlation (tracking quality on moving targets)
    tgx, tgy = np.diff(sx), np.diff(sy)
    ggx, ggy = np.diff(gx), np.diff(gy)
    if np.std(tgx) > 1e-9 and np.std(ggx) > 1e-9:
        feats["gaze_target_x_corr"] = float(np.corrcoef(ggx, tgx)[0, 1])
    else:
        feats["gaze_target_x_corr"] = 0.0
    if np.std(tgy) > 1e-9 and np.std(ggy) > 1e-9:
        feats["gaze_target_y_corr"] = float(np.corrcoef(ggy, tgy)[0, 1])
    else:
        feats["gaze_target_y_corr"] = 0.0
    return feats


def _detect_microsaccades(gx, gy, t, min_amplitude=0.1, max_amplitude=1.0, min_duration=0.01):
    """Detect microsaccades using velocity-based algorithm."""
    speed, dt = _speeds(gx, gy, t)
    
    # Velocity threshold (2× median speed)
    vel_threshold = 2.0 * np.median(speed)
    
    # Find segments above threshold
    above_threshold = speed > vel_threshold
    segments = []
    start = None
    
    for i, val in enumerate(above_threshold):
        if val and start is None:
            start = i
        elif not val and start is not None:
            duration = (t[i] - t[start]) if i < len(t) and start < len(t) else 0
            if duration >= min_duration:
                amplitude = np.hypot(gx[i] - gx[start], gy[i] - gy[start])
                if min_amplitude <= amplitude <= max_amplitude:
                    segments.append((start, i, amplitude, duration))
            start = None
    
    return segments


def _microsaccade_features(gx, gy, t):
    """Extract microsaccade features."""
    microsaccades = _detect_microsaccades(gx, gy, t)
    
    if not microsaccades:
        return {
            "microsaccade_rate_hz": 0.0,
            "microsaccade_amplitude_mean": 0.0,
            "microsaccade_amplitude_std": 0.0,
            "microsaccade_duration_mean": 0.0,
            "microsaccade_count": 0
        }
    
    amplitudes = [m[2] for m in microsaccades]
    durations = [m[3] for m in microsaccades]
    total_duration = max(t[-1] - t[0], 1e-6)
    
    return {
        "microsaccade_rate_hz": float(len(microsaccades) / total_duration),
        "microsaccade_amplitude_mean": float(np.mean(amplitudes)),
        "microsaccade_amplitude_std": float(np.std(amplitudes)),
        "microsaccade_duration_mean": float(np.mean(durations)),
        "microsaccade_count": len(microsaccades)
    }


def _pupillometry_features(pupil_data, t):
    """Extract pupillometry trend and variability features."""
    if len(pupil_data) < 5:
        return {
            "pupil_trend_slope": 0.0,
            "pupil_variability_cv": 0.0,
            "pupil_dilation_episodes": 0,
            "pupil_constriction_episodes": 0
        }
    
    # Linear trend (slope)
    valid = np.isfinite(pupil_data) & np.isfinite(t)
    if valid.sum() < 3:
        return {
            "pupil_trend_slope": 0.0,
            "pupil_variability_cv": 0.0,
            "pupil_dilation_episodes": 0,
            "pupil_constriction_episodes": 0
        }
    
    pupil_clean = pupil_data[valid]
    t_clean = t[valid]
    slope, _ = np.polyfit(t_clean, pupil_clean, 1)
    
    # Coefficient of variation
    cv = np.std(pupil_clean) / (np.mean(pupil_clean) + 1e-8)
    
    # Dilation/constriction episodes
    diff = np.diff(pupil_clean)
    dt_clean = np.diff(t_clean)
    rate = diff / (dt_clean + 1e-8)
    
    dilation_episodes = np.sum(rate > 0.01)  # dilation threshold
    constriction_episodes = np.sum(rate < -0.01)  # constriction threshold

    # Low frequency pupillometry oscillation power (0.1 - 0.5 Hz - cognitive hippus)
    if len(pupil_clean) >= 30:
        fs_pupil = 1.0 / (np.median(dt_clean) + 1e-8)
        freqs_p, psd_p = welch(pupil_clean - np.mean(pupil_clean), fs=fs_pupil, nperseg=min(64, len(pupil_clean)))
        lfo_idx = np.where((freqs_p >= 0.1) & (freqs_p <= 0.5))[0]
        lfo_power = float(np.sum(psd_p[lfo_idx])) if len(lfo_idx) > 0 else 0.0
    else:
        lfo_power = 0.0

    return {
        "pupil_trend_slope": float(slope),
        "pupil_variability_cv": float(cv),
        "pupil_dilation_episodes": int(dilation_episodes),
        "pupil_constriction_episodes": int(constriction_episodes),
        "pupil_lfo_power": float(lfo_power)
    }


def extract_eye_features(eye_df: pd.DataFrame):
    feats = {}

    # --- pupil features (synthetic + webcam sources only) ---
    if "pupil_mm" in eye_df.columns and eye_df["pupil_mm"].notna().sum() > 1:
        pupil_data = eye_df["pupil_mm"].to_numpy(float)
        t_pupil = eye_df["t"].to_numpy(float) if "t" in eye_df.columns else np.arange(len(pupil_data)) / 60.0
        
        feats["pupil_mean"] = float(eye_df["pupil_mm"].mean())
        feats["pupil_std"] = float(eye_df["pupil_mm"].std())
        
        # Advanced pupillometry features
        pupillometry_feats = _pupillometry_features(pupil_data, t_pupil)
        feats.update(pupillometry_feats)

    # --- blink features (sources with a blink flag only) ---
    if "is_blink" in eye_df.columns:
        t_arr = eye_df["t"].to_numpy(float) if "t" in eye_df.columns else np.arange(len(eye_df)) / 60.0
        t_end = float(t_arr[-1]) if len(t_arr) > 0 else 4.0
        blink_arr = eye_df["is_blink"].astype(int).to_numpy()
        feats["blink_rate_hz"] = float(blink_arr.sum() / (t_end + 1e-8))

        # Kinematic duration metrics
        diff = np.diff(np.pad(blink_arr, (1, 1), 'constant'))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        feats["blink_event_count"] = int(len(starts))
        if len(starts) > 0 and len(ends) >= len(starts):
            durations_s = t_arr[np.minimum(ends[:len(starts)], len(t_arr) - 1)] - t_arr[np.minimum(starts, len(t_arr) - 1)]
            feats["blink_duration_mean_ms"] = float(np.mean(durations_s) * 1000.0)
            feats["blink_duration_std_ms"] = float(np.std(durations_s) * 1000.0)
        else:
            feats["blink_duration_mean_ms"] = 0.0
            feats["blink_duration_std_ms"] = 0.0

    # --- fixation ratio from the flag column if present ---
    if "is_fixation" in eye_df.columns:
        feats["fixation_ratio"] = float(eye_df["is_fixation"].mean())

    gx, gy = eye_df["gaze_x"].to_numpy(float), eye_df["gaze_y"].to_numpy(float)
    valid = np.isfinite(gx) & np.isfinite(gy)
    if valid.sum() < 10:  # not enough usable gaze in this epoch
        return feats

    gx, gy = gx[valid], gy[valid]
    t = eye_df["t"].to_numpy(float)[valid] if "t" in eye_df.columns else np.arange(len(gx)) / 60.0

    feats.update(_base_gaze_feats(gx, gy, t))
    
    # Microsaccade features
    microsaccade_feats = _microsaccade_features(gx, gy, t)
    feats.update(microsaccade_feats)

    # --- target-tracking features (real dataset with Stimulus_x/y) ---
    if "stimulus_x" in eye_df.columns and "stimulus_y" in eye_df.columns:
        sx = eye_df["stimulus_x"].to_numpy(float)[valid]
        sy = eye_df["stimulus_y"].to_numpy(float)[valid]
        ok = np.isfinite(sx) & np.isfinite(sy)
        if ok.sum() >= 10:
            feats.update(_target_tracking_feats(gx[ok], gy[ok], t[ok], sx[ok], sy[ok]))

    # --- derived fixation ratio if no flag column existed ---
    if "fixation_ratio" not in feats:
        w = 5
        if len(gx) > 2 * w:
            disp = np.array([
                np.std(gx[i - w:i + w + 1]) + np.std(gy[i - w:i + w + 1])
                for i in range(w, len(gx) - w)
            ])
            feats["fixation_ratio"] = float(np.mean(disp < (np.median(disp) + np.std(disp))))
        else:
            feats["fixation_ratio"] = 0.5

    return feats


def extract_batch(eye_paths):
    rows = []
    for p in eye_paths:
        df = pd.read_csv(p)
        rows.append(extract_eye_features(df))
    return pd.DataFrame(rows)
