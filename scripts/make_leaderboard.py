#!/usr/bin/env python3

"""python scripts/make_leaderboard.py            # writes LEADERBOARD.md"""

import glob
import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def fmt(mean, std, places=3):
    if mean is None:
        return "-"
    return f"{mean:.{places}f} ± {std:.{places}f}"


def load_rows():
    rows = []
    for path in sorted(glob.glob(os.path.join(REPO, "models", "*", "results.json"))):
        try:
            rows.extend(json.load(open(path)))
        except Exception as e:
            print(f"skip {path}: {e}")
    return rows


def main():
    rows = load_rows()
    # index: (model, dataset) -> {task: row}
    table = {}
    for r in rows:
        key = (r["model"], r["dataset"])
        table.setdefault(key, {})[r["task"]] = r

    # build one display row per (model, dataset)
    display = []
    for (model, dataset), tasks in table.items():
        sex = tasks.get("sex", {})
        age = tasks.get("age", {})
        bmi = tasks.get("bmi", {})
        display.append({
            "model": model,
            "dataset": dataset,
            "sex_acc": sex.get("test_acc_mean"),
            "sex_cell": fmt(sex.get("test_acc_mean"), sex.get("test_acc_std")),
            "age_cell": fmt(age.get("test_mae_mean"), age.get("test_mae_std"), 2),
            "bmi_cell": fmt(bmi.get("test_mae_mean"), bmi.get("test_mae_std"), 2),
        })

    display.sort(key=lambda d: (d["sex_acc"] is not None, d["sex_acc"] or 0), reverse=True)

    lines = [
        "# OpenMRIBench Leaderboard",
        "",
        "Downstream probing of frozen features (Random Forest, 5-seed held-out test).",
        "Sex = balanced accuracy (higher is better); Age / BMI = MAE in years (lower is better).",
        "",
        "| Rank | Model | Dataset | Sex (acc) | Age (MAE) | BMI (MAE) |",
        "|-----:|-------|---------|-----------|-----------|-----------|",
    ]
    for i, d in enumerate(display, 1):
        lines.append(f"| {i} | {d['model']} | {d['dataset']} | "
                     f"{d['sex_cell']} | {d['age_cell']} | {d['bmi_cell']} |")
    lines.append("")
    lines.append("_Auto-generated from `models/*/results.json`. Do not edit by hand._")

    out = os.path.join(REPO, "LEADERBOARD.md")
    with open(out, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote {out}  ({len(display)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
