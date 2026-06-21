#!/usr/bin/env python3

"""Score one submission (a parquet of subject_id -> feature vector) against a
frozen task, using the exact downstream protocol from the benchmark paper."""

import argparse
import sys
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.metrics import (balanced_accuracy_score, roc_auc_score,
                             mean_absolute_error, r2_score)
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier, DummyRegressor

SEEDS = [0, 1, 2, 3, 42]
alphas = [0.01, 0.05, 0.1, 0.15, 0.25, 0.4, 0.6, 0.8, 1.0]

# From the benchmark paper pipeline (do not edit: keeps scores identical)

def run_classification(X_dict, y, test_size=0.1, task_name="Classification"):
    print(f"\n{task_name}")
    results = {}

    for model, X in X_dict.items():
        all_val_auc = {a: [] for a in alphas}
        all_val_acc = {a: [] for a in alphas}
        all_test_acc, all_test_auc = [], []

        for seed in SEEDS:
            cv_idx, test_idx = train_test_split(np.arange(len(y)), test_size=int(test_size * len(y)),
                                                stratify=y, random_state=seed)
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
            for train_rel, val_rel in skf.split(cv_idx, y[cv_idx]):
                train_idx, val_idx = cv_idx[train_rel], cv_idx[val_rel]
                for alpha in alphas:
                    n = int(alpha * len(train_idx))
                    if n >= len(train_idx):
                        tr_idx = train_idx
                    else:
                        tr_idx, _ = train_test_split(train_idx, train_size=n, stratify=y[train_idx], random_state=seed)
                    if len(np.unique(y[tr_idx])) < 2:
                        continue
                    scaler = StandardScaler()
                    rf = RandomForestClassifier(n_estimators=200, max_depth=6, min_samples_split=5,
                                                random_state=seed, n_jobs=1, max_features='sqrt', class_weight='balanced')
                    rf.fit(scaler.fit_transform(X[tr_idx]), y[tr_idx])
                    val_pred_proba = rf.predict_proba(scaler.transform(X[val_idx]))[:, 1]
                    all_val_acc[alpha].append(balanced_accuracy_score(y[val_idx], rf.predict(scaler.transform(X[val_idx]))))
                    all_val_auc[alpha].append(roc_auc_score(y[val_idx], val_pred_proba))

            scaler = StandardScaler()
            rf = RandomForestClassifier(n_estimators=200, max_depth=6, min_samples_split=5,
                                        random_state=seed, n_jobs=1, max_features='sqrt', class_weight='balanced')
            rf.fit(scaler.fit_transform(X[cv_idx]), y[cv_idx])
            test_pred_proba = rf.predict_proba(scaler.transform(X[test_idx]))[:, 1]
            all_test_acc.append(balanced_accuracy_score(y[test_idx], rf.predict(scaler.transform(X[test_idx]))))
            all_test_auc.append(roc_auc_score(y[test_idx], test_pred_proba))

        avg_acc = {a: np.mean(all_val_acc[a]) for a in alphas if all_val_acc[a]}
        best_alpha = max(avg_acc, key=avg_acc.get)
        results[model] = {
            'best_alpha': best_alpha,
            'test_acc_mean': np.mean(all_test_acc), 'test_acc_std': np.std(all_test_acc),
            'test_auc_mean': np.mean(all_test_auc), 'test_auc_std': np.std(all_test_auc),
            'cv_acc_mean': np.mean(all_val_acc[best_alpha]), 'cv_acc_std': np.std(all_val_acc[best_alpha]),
        }
        print(f"{model}: Acc={np.mean(all_test_acc):.3f}+/-{np.std(all_test_acc):.3f}, "
              f"AUC={np.mean(all_test_auc):.3f}+/-{np.std(all_test_auc):.3f}")
    return results


def run_regression(X_dict, y, test_size=0.1, task_name="Regression"):
    print(f"\n{task_name}")
    results = {}

    for model, X in X_dict.items():
        all_val_mae = {a: [] for a in alphas}
        all_test_r2, all_test_mae = [], []

        for seed in SEEDS:
            cv_idx, test_idx = train_test_split(np.arange(len(y)), test_size=int(test_size * len(y)), random_state=seed)
            kf = KFold(n_splits=5, shuffle=True, random_state=seed)
            for train_rel, val_rel in kf.split(cv_idx):
                train_idx, val_idx = cv_idx[train_rel], cv_idx[val_rel]
                for alpha in alphas:
                    n = int(alpha * len(train_idx))
                    tr_idx, _ = train_test_split(train_idx, train_size=n, random_state=seed) if n < len(train_idx) else (train_idx, None)
                    scaler = StandardScaler()
                    rf = RandomForestRegressor(n_estimators=200, max_depth=6, min_samples_split=5,
                                               random_state=seed, n_jobs=1)
                    rf.fit(scaler.fit_transform(X[tr_idx]), y[tr_idx])
                    pred = rf.predict(scaler.transform(X[val_idx]))
                    all_val_mae[alpha].append(mean_absolute_error(y[val_idx], pred))

            scaler = StandardScaler()
            rf = RandomForestRegressor(n_estimators=200, max_depth=6, min_samples_split=5,
                                       random_state=seed, n_jobs=1)
            rf.fit(scaler.fit_transform(X[cv_idx]), y[cv_idx])
            test_pred = rf.predict(scaler.transform(X[test_idx]))
            all_test_r2.append(r2_score(y[test_idx], test_pred))
            all_test_mae.append(mean_absolute_error(y[test_idx], test_pred))

        avg_mae = {a: np.mean(all_val_mae[a]) for a in alphas if all_val_mae[a]}
        best_alpha = min(avg_mae, key=avg_mae.get)
        results[model] = {
            'best_alpha': best_alpha,
            'test_mae_mean': np.mean(all_test_mae), 'test_mae_std': np.std(all_test_mae),
            'test_r2_mean': np.mean(all_test_r2), 'test_r2_std': np.std(all_test_r2),
            'cv_mae_mean': np.mean(all_val_mae[best_alpha]), 'cv_mae_std': np.std(all_val_mae[best_alpha]),
        }
        print(f"{model}: MAE={np.mean(all_test_mae):.2f}+/-{np.std(all_test_mae):.2f}, "
              f"R2={np.mean(all_test_r2):.3f}+/-{np.std(all_test_r2):.3f}")
    return results

# Thin wrapper: align one submission to the canonical manifest, then score

TASK_TYPE = {"sex": "classification", "age": "regression"}


def load_aligned(features_path, labels_path, task):
    """Load a submitted feature parquet and the label table, aligned by
    subject_id to a single canonical order. Returns (model_name, X, y)."""
    feats = pd.read_parquet(features_path)
    labels = pd.read_csv(labels_path)

    if "subject_id" not in feats.columns:
        raise ValueError("features parquet must have a 'subject_id' column")
    if "subject_id" not in labels.columns or task not in labels.columns:
        raise ValueError(f"labels csv must have 'subject_id' and '{task}' columns")

    # Canonical order = sorted subject_ids present in BOTH files
    common = sorted(set(feats["subject_id"]) & set(labels["subject_id"]))
    if len(common) == 0:
        raise ValueError("no overlapping subject_id between features and labels")

    feats = feats.set_index("subject_id").loc[common]
    labels = labels.set_index("subject_id").loc[common]

    feat_cols = [c for c in feats.columns]
    X = feats[feat_cols].to_numpy(dtype=float)
    y = labels[task].to_numpy()

    model_name = feats.attrs.get("model_name", "submission")
    return model_name, X, y, len(common)


def score(features_path, labels_path, dataset, task):
    model_name, X, y, n = load_aligned(features_path, labels_path, task)
    print(f"Scoring '{model_name}' on {dataset}/{task}: "
          f"{n} subjects, {X.shape[1]}-d features")

    ttype = TASK_TYPE[task]
    if ttype == "classification":
        y = y.astype(int)
        res = run_classification({model_name: X}, y, task_name=f"{dataset} {task}")
        r = res[model_name]
        row = {"dataset": dataset, "task": task, "model": model_name,
               "test_acc_mean": r["test_acc_mean"], "test_acc_std": r["test_acc_std"],
               "test_auc_mean": r["test_auc_mean"], "test_auc_std": r["test_auc_std"]}
    else:
        y = y.astype(float)
        res = run_regression({model_name: X}, y, task_name=f"{dataset} {task}")
        r = res[model_name]
        row = {"dataset": dataset, "task": task, "model": model_name,
               "test_mae_mean": r["test_mae_mean"], "test_mae_std": r["test_mae_std"],
               "test_r2_mean": r["test_r2_mean"], "test_r2_std": r["test_r2_std"]}
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features", required=True, help="submission parquet (subject_id + feature cols)")
    p.add_argument("--labels", required=True, help="labels csv (subject_id + task column)")
    p.add_argument("--dataset", required=True, choices=["NKI", "HBN"])
    p.add_argument("--task", required=True, choices=list(TASK_TYPE.keys()))
    args = p.parse_args()

    try:
        row = score(args.features, args.labels, args.dataset, args.task)
    except Exception as e:
        print(f"SCORING FAILED: {e}")
        return 1

    print("\nRESULT:")
    for k, v in row.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
