"""Dinov3D-neuro for BrainFMBench.

Model code comes from asagilmore/dinov3d (private), cloned on Rorqual at
DINOV3D_SRC and pinned to commit e5ef877. Weights and config come from
huggingbrain/Dinov3d-Neuro via weights.txt.
"""
import os
import sys
import glob

import numpy as np
import nibabel as nib
import torch

DINOV3D_SRC = "/home/arelbaha/links/scratch/dinov3d_git"
PATCH = 16


def load_volume(path):
    v = nib.load(path).get_fdata().astype(np.float32)
    brain = v > v.min()
    v[brain] = (v[brain] - v[brain].mean()) / (v[brain].std() + 1e-8)
    v[~brain] = v[brain].min()
    v = np.pad(v, [(0, (-s) % PATCH) for s in v.shape], constant_values=v.min())
    return torch.from_numpy(v)[None, None]


def extract(input_dir, output_csv, weights_dir):
    sys.path.insert(0, DINOV3D_SRC)
    from dinov3.inference.load import load_teacher_model_for_inference

    if os.environ.get("SLURM_CPUS_PER_TASK"):
        torch.set_num_threads(int(os.environ["SLURM_CPUS_PER_TASK"]))
    torch.manual_seed(0)

    model, dim = load_teacher_model_for_inference(
        os.path.join(weights_dir, "config.yaml"),
        os.path.join(weights_dir, "teacher_checkpoint.pth"),
        device="cpu",
    )
    model.eval()

    paths = sorted(glob.glob(os.path.join(input_dir, "*", "normalized.nii.gz")))
    print(f"[dinov3d] {len(paths)} subjects in {input_dir}, embed_dim={dim}", flush=True)

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    with open(output_csv, "w") as f, torch.no_grad():
        f.write("subject_id," + ",".join(f"f{i}" for i in range(dim)) + "\n")
        for i, p in enumerate(paths):
            cls = model.forward_features(load_volume(p))["x_norm_clstoken"][0].float().numpy()
            f.write(os.path.basename(os.path.dirname(p)) + "," + ",".join(f"{v:.7g}" for v in cls) + "\n")
            if (i + 1) % 50 == 0:
                print(f"[dinov3d] {i + 1}/{len(paths)}", flush=True)
    print(f"[dinov3d] wrote {len(paths)} x {dim} -> {output_csv}", flush=True)
