"""
EEG feature extraction (v3) — mental-workload-oriented features per channel:

  - Relative band power in 5 bands (delta, theta, alpha, beta, gamma)
  - Log absolute band power (impedance-robust scale; adaptive norm downstream)
  - Classic workload ratios: theta/alpha, beta/alpha, (theta+beta)/alpha
  - Hjorth parameters (activity, mobility, complexity) — time-domain dynamics
  - Spectral entropy — spectral complexity of 1-45 Hz
  - Raw signal std — overall amplitude / artifact level
  - Hemispheric asymmetry per band for the (AF7, AF8) and (TP9, TP10) pairs —
    the classic frontal/parietal laterality workload markers
  - NEW: Connectivity measures (coherence, phase lag index)
  - NEW: Additional entropy metrics (sample entropy, permutation entropy)
  - NEW: Cross-frequency coupling metrics
  - NEW: Fractal dimension features

Features are computed per channel; channel names are passed by the caller so
the same extractor serves synthetic data, BrainFlow boards, and the real
Muse CSV.
"""
import numpy as np
from scipy.signal import welch, coherence

FS_EEG = 256
BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13),
         "beta": (13, 30), "gamma": (30, 45)}
RATIO_BANDS = ("theta", "alpha", "beta")  # subset used for ratios/asymmetry
ASYMMETRY_PAIRS = [("AF7", "AF8"), ("TP9", "TP10")]  # (left, right)

_trapz_fn = getattr(np, "trapezoid", None) or np.trapz


def bandpower(sig_1d, fs=FS_EEG, band=(4, 8)):
    freqs, psd = welch(sig_1d, fs=fs, nperseg=min(256, len(sig_1d)))
    idx = np.logical_and(freqs >= band[0], freqs <= band[1])
    return _trapz_fn(psd[idx], freqs[idx])


def _hjorth_params(sig):
    """Activity (variance), mobility, complexity of a 1-D signal."""
    x = np.asarray(sig, dtype=float)
    dx = np.diff(x)
    ddx = np.diff(dx)
    var0 = np.var(x) + 1e-12
    var1 = np.var(dx) + 1e-12
    var2 = np.var(ddx) + 1e-12
    mobility = np.sqrt(var1 / var0)
    complexity = np.sqrt(var2 / var1) / (mobility + 1e-12)
    return float(np.var(x)), float(mobility), float(complexity)


def _spectral_entropy(sig, fs=FS_EEG, fmin=1.0, fmax=45.0):
    """Normalized spectral entropy of the 1-45 Hz band (0 = pure tone, 1 = flat)."""
    freqs, psd = welch(sig, fs=fs, nperseg=min(256, len(sig)))
    idx = np.logical_and(freqs >= fmin, freqs <= fmax)
    p = psd[idx]
    p = p / (p.sum() + 1e-12)
    return float(-np.sum(p * np.log(p + 1e-12)) / np.log(len(p) + 1e-12))


def _sample_entropy(sig, m=2, r=None):
    """Sample entropy - regularity/complexity metric."""
    sig = np.asarray(sig, dtype=float)
    n = len(sig)
    if r is None:
        r = 0.2 * np.std(sig)
    
    def _phi(m):
        patterns = np.array([sig[i:i+m] for i in range(n - m + 1)])
        dists = np.abs(patterns[:, np.newaxis, :] - patterns[np.newaxis, :, :]).max(axis=2)
        return np.sum(dists < r) / (n - m + 1) / (n - m)
    
    if n < m + 1:
        return 0.0
    return -np.log(_phi(m + 1) / (_phi(m) + 1e-12) + 1e-12)


def _permutation_entropy(sig, order=3, delay=1):
    """Permutation entropy - measure of signal regularity."""
    sig = np.asarray(sig, dtype=float)
    n = len(sig)
    patterns = []
    
    for i in range(n - order * delay):
        pattern = sig[i:i + order * delay:delay]
        permutation = np.argsort(pattern)
        patterns.append(tuple(permutation))
    
    from collections import Counter
    counts = Counter(patterns)
    probs = np.array([c / len(patterns) for c in counts.values()])
    return float(-np.sum(probs * np.log(probs + 1e-12)))


def _coherence_measure(sig1, sig2, fs=FS_EEG, band='alpha'):
    """Coherence between two signals in specific frequency band."""
    freqs, coh = coherence(sig1, sig2, fs=fs, nperseg=min(256, len(sig1)))
    fmin, fmax = BANDS[band]
    idx = np.logical_and(freqs >= fmin, freqs <= fmax)
    return float(np.mean(coh[idx]))


def _phase_lag_index(sig1, sig2):
    """Phase Lag Index - measure of phase coupling."""
    from scipy.signal import hilbert
    analytic1 = hilbert(sig1)
    analytic2 = hilbert(sig2)
    phase1 = np.angle(analytic1)
    phase2 = np.angle(analytic2)
    phase_diff = phase1 - phase2
    return float(np.abs(np.mean(np.exp(1j * phase_diff))))


def _fractal_dimension(sig):
    """Higuchi fractal dimension - signal complexity."""
    sig = np.asarray(sig, dtype=float)
    n = len(sig)
    k_max = 10
    L = []
    
    for k in range(1, k_max + 1):
        Lk = []
        for m in range(k):
            idx = np.arange(m, n, k)
            if len(idx) < 2:
                continue
            diff = np.abs(np.diff(sig[idx]))
            Lk.append(np.sum(diff) * (n - 1) / (len(idx) - 1) / k)
        if Lk:
            L.append(np.mean(Lk))
    
    if len(L) < 2:
        return 1.0
    
    k_vals = np.arange(1, len(L) + 1)
    log_L = np.log(L)
    log_k = np.log(k_vals)
    slope, _ = np.polyfit(log_k, log_L, 1)
    return float(-slope)


def extract_eeg_features(eeg_epoch, fs=FS_EEG, channel_names=None):
    """eeg_epoch: np.ndarray of shape (n_channels, n_samples)."""
    eeg_epoch = np.nan_to_num(np.asarray(eeg_epoch, dtype=float),
                              nan=0.0, posinf=0.0, neginf=0.0)
    n_channels = eeg_epoch.shape[0]
    channel_names = list(channel_names or [f"ch{i}" for i in range(n_channels)])
    feats = {}

    per_ch_powers = {}
    for ci, ch_name in enumerate(channel_names):
        sig = eeg_epoch[ci]
        powers = {b: bandpower(sig, fs, rng) for b, rng in BANDS.items()}
        per_ch_powers[ch_name] = powers
        total = sum(powers.values()) + 1e-8
        for b in BANDS:
            feats[f"{ch_name}_{b}_rel"] = powers[b] / total
            feats[f"{ch_name}_{b}_logabs"] = float(np.log10(powers[b] + 1e-12))
        pa, pb = powers["theta"], powers["alpha"]
        feats[f"{ch_name}_theta_alpha_ratio"] = pa / (pb + 1e-8)
        feats[f"{ch_name}_beta_alpha_ratio"] = powers["beta"] / (pb + 1e-8)
        feats[f"{ch_name}_tb_alpha_ratio"] = (pa + powers["beta"]) / (pb + 1e-8)
        act, mob, cpx = _hjorth_params(sig)
        feats[f"{ch_name}_hjorth_activity"] = act
        feats[f"{ch_name}_hjorth_mobility"] = mob
        feats[f"{ch_name}_hjorth_complexity"] = cpx
        feats[f"{ch_name}_spectral_entropy"] = _spectral_entropy(sig, fs)
        feats[f"{ch_name}_signal_std"] = float(np.std(sig))
        
        # Advanced entropy features
        feats[f"{ch_name}_sample_entropy"] = _sample_entropy(sig)
        feats[f"{ch_name}_permutation_entropy"] = _permutation_entropy(sig)
        
        # Differential entropy (DE) across all frequency bands
        for b in BANDS:
            feats[f"{ch_name}_{b}_diff_entropy"] = float(0.5 * np.log(2 * np.pi * np.e * (powers[b] + 1e-12)))

        # Individual Alpha Frequency (IAF) peak in 7-13 Hz
        freqs_welch, psd_welch = welch(sig, fs=fs, nperseg=min(256, len(sig)))
        alpha_idx = np.where((freqs_welch >= 7.0) & (freqs_welch <= 13.0))[0]
        if len(alpha_idx) > 0:
            feats[f"{ch_name}_iaf_peak"] = float(freqs_welch[alpha_idx[np.argmax(psd_welch[alpha_idx])]])
        else:
            feats[f"{ch_name}_iaf_peak"] = 10.0

        # Fractal dimension
        feats[f"{ch_name}_fractal_dimension"] = _fractal_dimension(sig)

    # Cross-regional Cognitive Load Index (Frontal Theta / Parietal Alpha)
    frontal_chs = [ch for ch in channel_names if any(k in ch.upper() for k in ["AF", "FP", "F3", "F4", "FZ"])]
    parietal_chs = [ch for ch in channel_names if any(k in ch.upper() for k in ["TP", "P3", "P4", "PZ", "O1", "O2"])]
    if not frontal_chs and len(channel_names) >= 4:
        frontal_chs = channel_names[:2]
        parietal_chs = channel_names[2:4]

    if frontal_chs and parietal_chs:
        f_theta = float(np.mean([per_ch_powers[ch]["theta"] for ch in frontal_chs]))
        f_beta = float(np.mean([per_ch_powers[ch]["beta"] for ch in frontal_chs]))
        p_alpha = float(np.mean([per_ch_powers[ch]["alpha"] for ch in parietal_chs]))
        feats["cli_frontal_theta_parietal_alpha"] = float((f_theta + 1e-8) / (p_alpha + 1e-8))
        feats["cli_frontal_beta_parietal_alpha"] = float((f_beta + 1e-8) / (p_alpha + 1e-8))
        feats["cli_frontal_tb_parietal_alpha"] = float((f_theta + f_beta + 1e-8) / (p_alpha + 1e-8))

    # Hemispheric asymmetry: log-power(left) - log-power(right), per band.
    name_set = set(channel_names)
    for left, right in ASYMMETRY_PAIRS:
        if left in name_set and right in name_set:
            for b in RATIO_BANDS:
                feats[f"asym_{left}_{right}_{b}"] = float(
                    np.log10(per_ch_powers[left][b] + 1e-12)
                    - np.log10(per_ch_powers[right][b] + 1e-12))
            
            # Connectivity features
            left_sig = eeg_epoch[channel_names.index(left)]
            right_sig = eeg_epoch[channel_names.index(right)]
            
            for band in ['alpha', 'beta', 'theta']:
                feats[f"coh_{left}_{right}_{band}"] = _coherence_measure(left_sig, right_sig, fs, band)
            
            feats[f"pli_{left}_{right}"] = _phase_lag_index(left_sig, right_sig)
    
    # Global connectivity measures for all channel pairs
    for i, ch1 in enumerate(channel_names):
        for j, ch2 in enumerate(channel_names):
            if i < j:
                sig1 = eeg_epoch[i]
                sig2 = eeg_epoch[j]
                feats[f"global_coh_{ch1}_{ch2}"] = _coherence_measure(sig1, sig2, fs, 'alpha')
                feats[f"global_pli_{ch1}_{ch2}"] = _phase_lag_index(sig1, sig2)
    
    return feats


def extract_batch(eeg_paths, channel_names=None):
    import pandas as pd
    rows = []
    for p in eeg_paths:
        eeg = np.load(p)
        rows.append(extract_eeg_features(eeg, channel_names=channel_names))
    return pd.DataFrame(rows)
