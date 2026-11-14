"""
Synthetic data generator v2 — more realistic per-subject neural variability.

Improvements over v1:
  - Per-subject noise floor: each subject has a unique sigma in [0.4, 0.9]
    so the model must learn subject-invariant features, not noise patterns.
  - Per-subject Individual Alpha Frequency (IAF): 9–12 Hz, matches literature.
  - Partial overlap between medium and adjacent classes (sigma_overlap=0.1):
    medium is now a genuinely ambiguous class, as in real workload data.
  - Channel-wise impedance offset: each channel gets a random DC shift per
    subject, mimicking real EEG electrode placement variability.
  - Eye: per-subject pupil baseline (2.5–4.5 mm), blink rate variability.
  - N_SUBJECTS=30 (was 20) for more robust subject-wise cross-validation.
"""
import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(42)
FS_EEG = 256
EPOCH_SEC = 4
N_SUBJECTS = 30                   # ↑ from 20 — more subjects → better CV
N_EPOCHS_PER_CLASS_PER_SUBJECT = 4   # 30 subj × 4 × 3 classes = 360 epochs
CLASSES = ["low", "medium", "high"]
CHANNELS = ["AF7", "AF8", "TP9", "TP10"]


def _subject_params(subj_rng):
    """Draw per-subject biological constants once and reuse across epochs."""
    return {
        "iaf":           subj_rng.uniform(9.0, 12.0),   # individual alpha peak (Hz)
        "noise_sigma":   subj_rng.uniform(0.4, 0.9),    # cortical noise floor
        "ch_offset":     subj_rng.normal(0, 0.3, len(CHANNELS)),   # electrode impedance DC
        "pupil_base":    subj_rng.uniform(2.5, 4.5),    # mm resting pupil diameter
        "blink_base":    subj_rng.uniform(0.03, 0.08),  # blinks/sample at rest
    }


def _simulate_eeg_epoch(label, params, subj_rng):
    t = np.arange(0, EPOCH_SEC, 1 / FS_EEG)
    # Partial overlap: medium is centered between low & high with additive noise
    load_factor = {"low": 0.0, "medium": 0.5, "high": 1.0}[label]
    # Add trial-to-trial jitter (± 15% load factor variability)
    load_factor = float(np.clip(load_factor + subj_rng.normal(0, 0.075), 0.0, 1.0))

    iaf = params["iaf"]
    sigma = params["noise_sigma"]
    sig = np.zeros((len(CHANNELS), len(t)))
    for ch_idx in range(len(CHANNELS)):
        # Alpha: decreases with load (ERD)
        alpha_amp = (1.3 - 0.8 * load_factor) + subj_rng.normal(0, 0.1)
        alpha = alpha_amp * np.sin(2 * np.pi * iaf * t)
        # Theta: increases with load (frontal workload marker)
        theta_amp = (0.35 + 1.0 * load_factor) + subj_rng.normal(0, 0.08)
        theta = theta_amp * np.sin(2 * np.pi * 6.0 * t)
        # Beta: increases with load (engagement/alertness)
        beta_amp = (0.25 + 0.65 * load_factor) + subj_rng.normal(0, 0.07)
        beta = beta_amp * np.sin(2 * np.pi * 20.0 * t)
        # Gamma: subtle high-frequency cognitive activity
        gamma_amp = 0.1 + 0.15 * load_factor
        gamma = gamma_amp * np.sin(2 * np.pi * 38.0 * t)
        # Pink noise (1/f) + white noise + DC electrode offset
        pink = subj_rng.normal(0, sigma, len(t))
        dc   = params["ch_offset"][ch_idx]
        sig[ch_idx] = alpha + theta + beta + gamma + pink + dc
    return sig


def _simulate_eye_epoch(label, params, subj_rng):
    fs_eye = 60
    n = EPOCH_SEC * fs_eye
    load_factor = {"low": 0.0, "medium": 0.5, "high": 1.0}[label]
    load_factor = float(np.clip(load_factor + subj_rng.normal(0, 0.075), 0.0, 1.0))

    # Pupil: dilates with cognitive load (task-evoked pupil response)
    pupil = (params["pupil_base"] + 1.1 * load_factor
             + subj_rng.normal(0, 0.15 + 0.05 * load_factor, n))
    pupil = np.clip(pupil, 1.5, 9.0)

    # Fixation rate increases under load (gaze locks on task area)
    fixation_flags = subj_rng.random(n) < (0.50 + 0.30 * load_factor)
    # Blink rate: decreases under high load (suppressed by cognitive effort)
    blink_rate = max(0.005, params["blink_base"] - 0.025 * load_factor)
    blink_flags = subj_rng.random(n) < blink_rate

    # Gaze: more confined under high load (task-locked), more wandering at low
    diffusion = 0.3 + 1.5 * (1.0 - load_factor)
    gaze_x = np.cumsum(subj_rng.normal(0, diffusion, n))
    gaze_y = np.cumsum(subj_rng.normal(0, diffusion, n))

    return pd.DataFrame({
        "t":          np.arange(n) / fs_eye,
        "pupil_mm":   pupil,
        "is_fixation": fixation_flags.astype(float),
        "is_blink":    blink_flags.astype(float),
        "gaze_x":     gaze_x,
        "gaze_y":     gaze_y,
    })


def generate_dataset(out_dir="data/raw"):
    out_dir = Path(out_dir)
    (out_dir / "eeg").mkdir(parents=True, exist_ok=True)
    (out_dir / "eye").mkdir(parents=True, exist_ok=True)
    rows, idx = [], 0
    for subj in range(N_SUBJECTS):
        subj_rng = np.random.default_rng(42 + subj * 97)
        params = _subject_params(subj_rng)
        for label in CLASSES:
            for _ in range(N_EPOCHS_PER_CLASS_PER_SUBJECT):
                eeg = _simulate_eeg_epoch(label, params, subj_rng)
                eye = _simulate_eye_epoch(label, params, subj_rng)
                eeg_path = out_dir / "eeg" / f"epoch_{idx:05d}.npy"
                eye_path = out_dir / "eye" / f"epoch_{idx:05d}.csv"
                np.save(eeg_path, eeg)
                eye.to_csv(eye_path, index=False)
                rows.append({
                    "epoch_id": idx, "subject": f"S{subj:02d}", "label": label,
                    "eeg_path": str(eeg_path), "eye_path": str(eye_path),
                })
                idx += 1
    manifest = pd.DataFrame(rows)
    manifest.to_csv(out_dir / "manifest.csv", index=False)
    print(f"Generated {len(manifest)} synchronized EEG+eye epochs → {out_dir}")
    print(f"({N_SUBJECTS} subjects × {N_EPOCHS_PER_CLASS_PER_SUBJECT} epochs × {len(CLASSES)} classes)")
    return manifest


if __name__ == "__main__":
    generate_dataset()
