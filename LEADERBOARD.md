# OpenMRIBench Leaderboard

Downstream probing of frozen features (Random Forest, 5-seed held-out test).
Sex = balanced accuracy (higher is better); Age / BMI = MAE in years (lower is better).

| Rank | Model | Dataset | Sex (acc) | Age (MAE) | BMI (MAE) |
|-----:|-------|---------|-----------|-----------|-----------|
| 1 | Default Untrained 3D CNN | NKI | 0.847 ± 0.028 | 10.00 ± 0.47 | 4.00 ± 0.27 |

_Auto-generated from `models/*/results.json`. Do not edit by hand._