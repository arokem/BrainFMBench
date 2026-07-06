#!/usr/bin/env python3
"""
Score one submission (subject_id -> feature vector) against a frozen task,
using the benchmark paper's protocol at the LARGEST training sample only
(alpha = 1.0).

For each model/dataset/task this produces exactly what the paper figure shows:
  * box   = 5-fold CV performance across 5 seeds  -> 25 values  (cv_*_values)
  * dots  = held-out test result for the same 5 seeds -> 5 values (test_*_seeds)
"""

import argparse
import sys
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.metrics import (balanced_accuracy_score, roc_auc_score,
                             mean_absolute_error, r2_score)
from sklearn.preprocessing import StandardScaler

SEEDS = [0, 1, 2, 3, 42]


def run_classification(X_dict, y, test_size=0.1, task_name="Classification"):
    print(f"\n{task_name}")
    results = {}
    for model, X in X_dict.items():
        cv_acc, cv_auc = [], []          # 25 values (5 seeds x 5 folds)
        test_acc, test_auc = [], []      # 5 values (per seed)

        for seed in SEEDS:
            cv_idx, test_idx = train_test_split(
                np.arange(len(y)), test_size=int(test_size * len(y)),
                stratify=y, random_state=seed)

            # ---- box: 5-fold CV at full training (alpha = 1.0) ----
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
            for train_rel, val_rel in skf.split(cv_idx, y[cv_idx]):
                train_idx, val_idx = cv_idx[train_rel], cv_idx[val_rel]
                if len(np.unique(y[train_idx])) < 2:
                    continue
                scaler = StandardScaler()
                rf = RandomForestClassifier(n_estimators=200, max_depth=6, min_samples_split=5,
                                            random_state=seed, n_jobs=1, max_features='sqrt',
                                            class_weight='balanced')
                rf.fit(scaler.fit_transform(X[train_idx]), y[train_idx])
                proba = rf.predict_proba(scaler.transform(X[val_idx]))[:, 1]
                cv_acc.append(balanced_accuracy_score(y[val_idx], rf.predict(scaler.transform(X[val_idx]))))
                cv_auc.append(roc_auc_score(y[val_idx], proba))

            # ---- dots: held-out test for this seed ----
            scaler = StandardScaler()
            rf = RandomForestClassifier(n_estimators=200, max_depth=6, min_samples_split=5,
                                        random_state=seed, n_jobs=1, max_features='sqrt',
                                        class_weight='balanced')
            rf.fit(scaler.fit_transform(X[cv_idx]), y[cv_idx])
            proba = rf.predict_proba(scaler.transform(X[test_idx]))[:, 1]
            test_acc.append(balanced_accuracy_score(y[test_idx], rf.predict(scaler.transform(X[test_idx]))))
            test_auc.append(roc_auc_score(y[test_idx], proba))

        results[model] = {
            'test_acc_mean': float(np.mean(test_acc)), 'test_acc_std': float(np.std(test_acc)),
            'test_auc_mean': float(np.mean(test_auc)), 'test_auc_std': float(np.std(test_auc)),
            'test_acc_seeds': [float(v) for v in test_acc],
            'test_auc_seeds': [float(v) for v in test_auc],
            'cv_acc_values': [float(v) for v in cv_acc],
            'cv_auc_values': [float(v) for v in cv_auc],
            'cv_acc_mean': float(np.mean(cv_acc)), 'cv_acc_std': float(np.std(cv_acc)),
        }
        print(f"{model}: test Acc={np.mean(test_acc):.3f}+/-{np.std(test_acc):.3f}, "
              f"AUC={np.mean(test_auc):.3f}+/-{np.std(test_auc):.3f}  "
              f"(cv Acc={np.mean(cv_acc):.3f}, n_cv={len(cv_acc)})")
    return results


def run_regression(X_dict, y, test_size=0.1, task_name="Regression"):
    print(f"\n{task_name}")
    results = {}
    for model, X in X_dict.items():
        cv_mae, cv_r2 = [], []           # 25 values
        test_mae, test_r2 = [], []       # 5 values

        for seed in SEEDS:
            cv_idx, test_idx = train_test_split(
                np.arange(len(y)), test_size=int(test_size * len(y)), random_state=seed)

            # ---- box: 5-fold CV at full training (alpha = 1.0) ----
            kf = KFold(n_splits=5, shuffle=True, random_state=seed)
            for train_rel, val_rel in kf.split(cv_idx):
                train_idx, val_idx = cv_idx[train_rel], cv_idx[val_rel]
                scaler = StandardScaler()
                rf = RandomForestRegressor(n_estimators=200, max_depth=6, min_samples_split=5,
                                           random_state=seed, n_jobs=1)
                rf.fit(scaler.fit_transform(X[train_idx]), y[train_idx])
                pred = rf.predict(scaler.transform(X[val_idx]))
                cv_mae.append(mean_absolute_error(y[val_idx], pred))
                cv_r2.append(r2_score(y[val_idx], pred))

            # ---- dots: held-out test for this seed ----
            scaler = StandardScaler()
            rf = RandomForestRegressor(n_estimators=200, max_depth=6, min_samples_split=5,
                                       random_state=seed, n_jobs=1)
            rf.fit(scaler.fit_transform(X[cv_idx]), y[cv_idx])
            pred = rf.predict(scaler.transform(X[test_idx]))
            test_mae.append(mean_absolute_error(y[test_idx], pred))
            test_r2.append(r2_score(y[test_idx], pred))

        results[model] = {
            'test_mae_mean': float(np.mean(test_mae)), 'test_mae_std': float(np.std(test_mae)),
            'test_r2_mean': float(np.mean(test_r2)), 'test_r2_std': float(np.std(test_r2)),
            'test_mae_seeds': [float(v) for v in test_mae],
            'test_r2_seeds': [float(v) for v in test_r2],
            'cv_mae_values': [float(v) for v in cv_mae],
            'cv_r2_values': [float(v) for v in cv_r2],
            'cv_mae_mean': float(np.mean(cv_mae)), 'cv_mae_std': float(np.std(cv_mae)),
        }
        print(f"{model}: test MAE={np.mean(test_mae):.2f}+/-{np.std(test_mae):.2f}, "
              f"R2={np.mean(test_r2):.3f}+/-{np.std(test_r2):.3f}  "
              f"(cv MAE={np.mean(cv_mae):.2f}, n_cv={len(cv_mae)})")
    return results


TASK_TYPE = {"sex": "classification", "age": "regression", "bmi": "regression"}


def _read_table(path):
    if str(path).endswith(".csv"):
        return pd.read_csv(path)
    return pd.read_parquet(path)


def load_aligned(features_path, labels_path, task, model_name=None):
    feats = _read_table(features_path)
    labels = pd.read_csv(labels_path)

    if "subject_id" not in feats.columns:
        raise ValueError("features table must have a 'subject_id' column")
    if "subject_id" not in labels.columns or task not in labels.columns:
        raise ValueError(f"labels csv must have 'subject_id' and '{task}' columns")

    common = sorted(set(feats["subject_id"]) & set(labels["subject_id"]))
    if len(common) == 0:
        raise ValueError("no overlapping subject_id between features and labels")

    feats = feats.set_index("subject_id").loc[common]
    labels = labels.set_index("subject_id").loc[common]

    feat_cols = [c for c in feats.columns]
    X = feats[feat_cols].to_numpy(dtype=float)
    y = labels[task].to_numpy()

    keep = ~pd.isna(y)
    n_dropped = int((~keep).sum())
    if n_dropped:
        print(f"  dropped {n_dropped} subjects with missing '{task}' label")
    X, y = X[keep], y[keep]
    n_kept = int(keep.sum())
    if n_kept == 0:
        raise ValueError(f"no subjects have a '{task}' label")

    if model_name is None:
        model_name = feats.attrs.get("model_name", "submission")
    return model_name, X, y, n_kept


def score(features_path, labels_path, dataset, task, model_name=None):
    model_name, X, y, n = load_aligned(features_path, labels_path, task, model_name)
    print(f"Scoring '{model_name}' on {dataset}/{task}: "
          f"{n} subjects, {X.shape[1]}-d features")

    ttype = TASK_TYPE[task]
    if ttype == "classification":
        y = y.astype(int)
        r = run_classification({model_name: X}, y, task_name=f"{dataset} {task}")[model_name]
        row = {"dataset": dataset, "task": task, "model": model_name,
               "test_acc_mean": r["test_acc_mean"], "test_acc_std": r["test_acc_std"],
               "test_auc_mean": r["test_auc_mean"], "test_auc_std": r["test_auc_std"],
               "test_acc_seeds": r["test_acc_seeds"], "test_auc_seeds": r["test_auc_seeds"],
               "cv_acc_values": r["cv_acc_values"], "cv_auc_values": r["cv_auc_values"]}
    else:
        y = y.astype(float)
        r = run_regression({model_name: X}, y, task_name=f"{dataset} {task}")[model_name]
        row = {"dataset": dataset, "task": task, "model": model_name,
               "test_mae_mean": r["test_mae_mean"], "test_mae_std": r["test_mae_std"],
               "test_r2_mean": r["test_r2_mean"], "test_r2_std": r["test_r2_std"],
               "test_mae_seeds": r["test_mae_seeds"], "test_r2_seeds": r["test_r2_seeds"],
               "cv_mae_values": r["cv_mae_values"], "cv_r2_values": r["cv_r2_values"]}
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--dataset", required=True, choices=["NKI", "HBN"])
    p.add_argument("--task", required=True, choices=list(TASK_TYPE.keys()))
    p.add_argument("--model-name", default=None)
    args = p.parse_args()
    try:
        row = score(args.features, args.labels, args.dataset, args.task, args.model_name)
    except Exception as e:
        print(f"SCORING FAILED: {e}")
        return 1
    print("\nRESULT keys:", list(row.keys()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
