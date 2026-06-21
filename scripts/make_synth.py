import numpy as np, pandas as pd

rng = np.random.default_rng(0)
n, dim = 300, 64
subj = [f"sub-{i:04d}" for i in range(n)]

# labels
sex = rng.integers(0, 2, size=n)
age = rng.uniform(8, 80, size=n)

# features: planted signal so the probe scores above chance
X = rng.standard_normal((n, dim))
X[:, 0] += 2.0 * sex            # one dim separates sex
X[:, 1] += 0.06 * age           # one dim tracks age

feats = pd.DataFrame(X, columns=[f"f{i}" for i in range(dim)])
feats.insert(0, "subject_id", subj)
feats.attrs["model_name"] = "synthetic-demo"
feats.to_parquet("synth_features.parquet")

pd.DataFrame({"subject_id": subj, "sex": sex, "age": age}).to_csv("synth_labels.csv", index=False)
print("wrote synth_features.parquet (", feats.shape, ") and synth_labels.csv")
