import os
ROOT_DIR = os.path.join(os.path.dirname(__file__), "..")

import torch
import yaml

DATA_PATH = os.path.join(ROOT_DIR, "dataloader/cmg_training_data.pt")
OUTPUT_PATH = os.path.join(ROOT_DIR, "evaluations/cmg_stats.yaml")

data = torch.load(DATA_PATH, weights_only=False)
stats = data["stats"]

out = {
    "motion_dim": int(stats["motion_dim"]),
    "command_dim": int(stats["command_dim"]),
    "num_joints": int(stats["num_joints"]),
    "motion_mean": stats["motion_mean"].tolist(),
    "motion_std": stats["motion_std"].tolist(),
    "command_min": stats["command_min"].tolist(),
    "command_max": stats["command_max"].tolist(),
}

with open(OUTPUT_PATH, "w") as f:
    yaml.dump(out, f, default_flow_style=False, sort_keys=False)

print(f"Saved stats in cmg_stats.yaml")
