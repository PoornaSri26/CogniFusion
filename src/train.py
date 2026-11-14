"""
Trains AttentionFusionNet v2 on a fused feature table.

Approach (v2):
  - SUBJECT-WISE splits: every held-out person is fully unseen (no random-row
    split, which leaks subject identity and overstates accuracy).
  - SESSION-ADAPTIVE NORMALIZATION: features are z-scored per (subject,
    session) using that session's own median/MAD before the population
    scaler. This removes per-session impedance/skin/gaze-calibration offsets —
    the dominant nuisance in EEG — and is exactly what the live pipeline can
    reproduce from its own rolling window (no training data needed).
  - CLASS-WEIGHTED cross-entropy + auxiliary single-modality losses (each
    encoder must be predictive on its own) + modality dropout (the fused
    model must survive a dead webcam or poor EEG contact).
  - Subject-wise K-fold CV: the saved checkpoint contains ALL fold models +
    their scalers; live inference averages the folds (an ensemble).
  - The checkpoint is fully self-contained (columns, label order, per-fold
    scalers) so the real-time pipeline only needs the one .pt file.

Examples:
  python src/train.py                                  # synthetic, holdout
  python src/train.py --features data/processed/features_real.parquet \
      --out models/fusion_model_real --cv 4            # real data, 4-fold CV
"""
import sys
import argparse
import pickle
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from model import AttentionFusionNet

LABELS = ["low", "medium", "high"]


# ----------------------------------------------------------------- splitting
def subject_groups(df):
    """Unique subject IDs in a stable, shuffled-able order."""
    return np.array(sorted(df["subject"].unique()))


def subject_wise_split(df, test_frac=0.25, seed=42):
    rng = np.random.default_rng(seed)
    subjects = subject_groups(df)
    rng.shuffle(subjects)
    n_test = max(1, int(len(subjects) * test_frac))
    test_subj = set(subjects[:n_test])
    return (df[~df["subject"].isin(test_subj)].reset_index(drop=True),
            df[df["subject"].isin(test_subj)].reset_index(drop=True))


def subject_kfold(df, k=4, seed=42):
    """Yield (train_df, test_df) per fold, splitting at the SUBJECT level."""
    rng = np.random.default_rng(seed)
    subjects = subject_groups(df)
    rng.shuffle(subjects)
    folds = np.array_split(subjects, k)
    for i in range(k):
        test_subj = set(folds[i])
        train_subj = set(subjects) - test_subj
        yield (df[df["subject"].isin(train_subj)].reset_index(drop=True),
               df[df["subject"].isin(test_subj)].reset_index(drop=True))


# ----------------------------------------------- session-adaptive normalization
def make_norm_key(df):
    """Adaptive-normalization key: the SUBJECT.

    Deliberately NOT subject+session: in the real dataset every session is a
    single task at a single difficulty level, so the label is constant within
    a session — per-session centering would subtract exactly the class signal
    we want to learn. Per-subject centering instead removes the person's
    overall impedance/gaze-calibration offset while keeping between-session
    (between-class) deviations intact. The live pipeline reproduces this with
    a rolling window that acts as the 'current subject' baseline."""
    return df["subject"].astype(str)


def coerce_numeric(df, columns):
    return df[columns].apply(pd.to_numeric, errors="coerce").replace(
        [np.inf, -np.inf], np.nan)


def adaptive_subject_norm(values: pd.DataFrame, keys: pd.Series):
    """Per-key (subject|session) robust z-score using that group's own
    median and MAD (MAD -> std via the 1.4826 consistency constant; falls
    back to the group std, then to 1.0, for degenerate features).

    Returns (normalized copy, {key: (center ndarray, scale ndarray)}).
    The stats are exactly what the live pipeline recomputes from a rolling
    calibration window, so train and inference see comparable inputs.
    """
    values = values.copy()
    stats = {}
    key_arr = keys.to_numpy()
    for key in pd.unique(key_arr):
        mask = key_arr == key
        sub = values.loc[mask]
        med = sub.median()
        mad = (sub - med).abs().median()
        scale = 1.4826 * mad
        scale = scale.where(scale > 1e-8, sub.std())
        scale = scale.where(scale > 1e-8, 1.0)
        med = med.fillna(0.0)
        scale = scale.fillna(1.0)
        values.loc[mask] = ((sub - med) / scale).fillna(0.0).to_numpy()
        stats[key] = (med.to_numpy(float), scale.to_numpy(float))
    return values, stats


# ------------------------------------------------------------------- training
class FocalLoss(nn.Module):
    """Focal Loss with class-weighting to combat hard-to-distinguish class boundaries."""
    def __init__(self, weight=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        return focal_loss.sum()


def train_fold(train_df, test_df, eeg_cols, eye_cols, le, *,
               epochs=200, lr=1e-3, weight_decay=5e-4, aux_weight=0.3,
               modality_dropout=0.2, embed_dim=24, use_focal=True, focal_gamma=2.0,
               ema_eval=True, seed=42, verbose=True):
    torch.manual_seed(seed)
    np.random.seed(seed)

    key_tr, key_te = train_df["norm_key"], test_df["norm_key"]

    # session-adaptive normalization within each split (test uses ONLY its
    # own session stats — no train information crosses the split)
    eeg_tr_norm, _ = adaptive_subject_norm(coerce_numeric(train_df, eeg_cols), key_tr)
    eye_tr_norm, _ = adaptive_subject_norm(coerce_numeric(train_df, eye_cols), key_tr)
    eeg_te_norm, _ = adaptive_subject_norm(coerce_numeric(test_df, eeg_cols), key_te)
    eye_te_norm, _ = adaptive_subject_norm(coerce_numeric(test_df, eye_cols), key_te)

    # population scaler on top (fit on TRAIN only; named frames avoid sklearn warnings)
    scaler_eeg, scaler_eye = StandardScaler(), StandardScaler()
    eeg_tr = scaler_eeg.fit_transform(eeg_tr_norm)
    eye_tr = scaler_eye.fit_transform(eye_tr_norm)
    eeg_te = scaler_eeg.transform(eeg_te_norm)
    eye_te = scaler_eye.transform(eye_te_norm)

    y_tr = torch.tensor(le.transform(train_df["label"]), dtype=torch.long)
    y_te = torch.tensor(le.transform(test_df["label"]), dtype=torch.long)
    eeg_tr = torch.tensor(np.nan_to_num(eeg_tr), dtype=torch.float32)
    eye_tr = torch.tensor(np.nan_to_num(eye_tr), dtype=torch.float32)
    eeg_te = torch.tensor(np.nan_to_num(eeg_te), dtype=torch.float32)
    eye_te = torch.tensor(np.nan_to_num(eye_te), dtype=torch.float32)

    counts = torch.bincount(y_tr, minlength=len(LABELS)).float()
    class_weights = counts.sum() / (len(counts) * counts.clamp(min=1))
    if use_focal:
        loss_main = FocalLoss(weight=class_weights, gamma=focal_gamma)
        loss_aux = FocalLoss(weight=class_weights, gamma=focal_gamma)
    else:
        loss_main = nn.CrossEntropyLoss(weight=class_weights)
        loss_aux = nn.CrossEntropyLoss(weight=class_weights)

    model = AttentionFusionNet(eeg_dim=len(eeg_cols), eye_dim=len(eye_cols),
                               embed_dim=embed_dim, n_classes=len(LABELS), gate_type="vector")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        logits, aux = model(eeg_tr, eye_tr, modality_dropout=modality_dropout)
        loss = loss_main(logits, y_tr) + aux_weight * (loss_aux(aux[0], y_tr)
                                                       + loss_aux(aux[1], y_tr))
        loss.backward()
        opt.step()
        sched.step()
        if verbose and (epoch + 1) % 50 == 0:
            acc = (logits.argmax(1) == y_tr).float().mean().item()
            print(f"    epoch {epoch + 1:3d} | loss {loss.item():.4f} | train_acc {acc:.3f}")

    model.eval()
    with torch.no_grad():
        logits_te, w_te = model(eeg_te, eye_te, return_weight=True)
        raw_probs = torch.softmax(logits_te, dim=1).numpy()
        if ema_eval and len(raw_probs) > 1:
            smoothed = np.zeros_like(raw_probs)
            smoothed[0] = raw_probs[0]
            alpha_ema = 0.65
            for t in range(1, len(raw_probs)):
                smoothed[t] = alpha_ema * raw_probs[t] + (1.0 - alpha_ema) * smoothed[t - 1]
            probs = smoothed
        else:
            probs = raw_probs
        preds = probs.argmax(1)
    y_true = y_te.numpy()

    macro_f1 = f1_score(y_true, preds, average="macro")
    acc = np.mean(preds == y_true)
    if verbose:
        print(classification_report(y_true, preds, target_names=le.classes_, digits=3))
        print("Confusion matrix:\n", confusion_matrix(y_true, preds))
        print(f"    accuracy {acc:.3f} | macro-F1 {macro_f1:.3f} | mean EEG-trust {w_te.mean().item():.3f}")

    fold_record = {
        "model_state": {k: v.cpu() for k, v in model.state_dict().items()},
        "scaler_eeg": scaler_eeg,
        "scaler_eye": scaler_eye,
        "val_macro_f1": float(macro_f1),
        "val_accuracy": float(acc),
    }
    return fold_record, macro_f1


# ----------------------------------------------------------------------- main
def population_robust_stats(values: pd.DataFrame):
    """Population-level median/MAD stats — the live pipeline's warmup fallback
    until its own rolling calibration window has enough epochs."""
    med = values.median().fillna(0.0)
    mad = (values - med).abs().median()
    scale = 1.4826 * mad
    scale = scale.where(scale > 1e-8, values.std()).where(scale > 1e-8, 1.0).fillna(1.0)
    return med.to_dict(), scale.to_dict()


def main(features_path="data/processed/features.parquet", epochs=200, lr=1e-3,
         weight_decay=5e-4, aux_weight=0.3, modality_dropout=0.2, cv=0,
         test_frac=0.25, use_focal=True, focal_gamma=2.0, ema_eval=True,
         out_prefix="models/fusion_model", seed=42):
    df = pd.read_parquet(features_path)
    eeg_cols = [c for c in df.columns if c.startswith("eeg__")]
    eye_cols = [c for c in df.columns if c.startswith("eye__")]
    df["norm_key"] = make_norm_key(df)
    eeg_pop_center, eeg_pop_scale = population_robust_stats(coerce_numeric(df, eeg_cols))
    eye_pop_center, eye_pop_scale = population_robust_stats(coerce_numeric(df, eye_cols))

    le = LabelEncoder().fit(df["label"])
    print(f"Features: {len(eeg_cols)} EEG + {len(eye_cols)} eye | epochs: {len(df)} | "
          f"subjects: {df['subject'].nunique()}")
    print(f"Enhancements active: Vector-Gated Fusion, Focal Loss (gamma={focal_gamma}), Temporal EMA Smoothing={ema_eval}")
    print(df["label"].value_counts().to_string())

    if cv and cv > 1:
        splits = list(subject_kfold(df, k=cv, seed=seed))
        print(f"\n=== {cv}-fold subject-wise cross-validation ===")
    else:
        splits = [subject_wise_split(df, test_frac=test_frac, seed=seed)]
        print("\n=== Single subject-wise holdout split ===")

    fold_records, f1s, fold_reports = [], [], []
    for i, (train_df, test_df) in enumerate(splits):
        print(f"\n--- Fold {i + 1}/{len(splits)} | train subjects: "
              f"{train_df['subject'].nunique()} | test subjects: "
              f"{test_df['subject'].nunique()} ---")
        record, macro_f1 = train_fold(
            train_df, test_df, eeg_cols, eye_cols, le,
            epochs=epochs, lr=lr, weight_decay=weight_decay,
            aux_weight=aux_weight, modality_dropout=modality_dropout,
            use_focal=use_focal, focal_gamma=focal_gamma, ema_eval=ema_eval,
            seed=seed + i)
        fold_records.append(record)
        f1s.append(macro_f1)

    print(f"\n=== CV summary ===")
    print(f"macro-F1 per fold: {[round(f, 3) for f in f1s]}")
    print(f"mean macro-F1: {np.mean(f1s):.3f} ± {np.std(f1s):.3f}")

    Path("models").mkdir(exist_ok=True)
    model_path = f"{out_prefix}.pt"
    torch.save({
        "version": 2,
        "model_class": "AttentionFusionNet",
        "gate_type": "vector",
        "eeg_cols": eeg_cols,
        "eye_cols": eye_cols,
        "label_classes": list(le.classes_),
        "embed_dim": 24,
        "pop_center": {"eeg": eeg_pop_center, "eye": eye_pop_center},
        "pop_scale": {"eeg": eeg_pop_scale, "eye": eye_pop_scale},
        "folds": fold_records,
        "cv_summary": {"macro_f1_per_fold": [float(f) for f in f1s],
                        "mean_macro_f1": float(np.mean(f1s)),
                        "std_macro_f1": float(np.std(f1s)),
                        "n_subjects": int(df["subject"].nunique()),
                        "n_epochs": int(len(df))},
    }, model_path)
    print(f"\nSaved self-contained ensemble checkpoint -> {model_path}")
    print("(scalers are embedded per fold; the live pipeline needs only this file)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train AttentionFusionNet v2 with High Accuracy Enhancements")
    parser.add_argument("--features", default="data/processed/features.parquet")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--aux-weight", type=float, default=0.3)
    parser.add_argument("--modality-dropout", type=float, default=0.2)
    parser.add_argument("--focal-loss", dest="use_focal", action="store_true", default=True,
                        help="Use Focal Loss for hard boundary discrimination")
    parser.add_argument("--no-focal-loss", dest="use_focal", action="store_false")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--ema-eval", dest="ema_eval", action="store_true", default=True,
                        help="Evaluate with sequence EMA temporal smoothing")
    parser.add_argument("--no-ema-eval", dest="ema_eval", action="store_false")
    parser.add_argument("--cv", type=int, default=0,
                        help="K-fold subject-wise CV (0 = single holdout split)")
    parser.add_argument("--test-frac", type=float, default=0.25)
    parser.add_argument("--out", dest="out_prefix", default="models/fusion_model")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(features_path=args.features, epochs=args.epochs, lr=args.lr,
         weight_decay=args.weight_decay, aux_weight=args.aux_weight,
         modality_dropout=args.modality_dropout, use_focal=args.use_focal,
         focal_gamma=args.focal_gamma, ema_eval=args.ema_eval,
         cv=args.cv, test_frac=args.test_frac, out_prefix=args.out_prefix, seed=args.seed)
