import sys
import os
ROOT_DIR = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT_DIR)

import torch
import torch.onnx
from module.cmg import CMG
checkpoint = torch.load(os.path.join(ROOT_DIR, "runs/cmg_20260211_040530/cmg_ckpt_700.pt"),
                        map_location="cpu", weights_only=False)

model = CMG( # the same as train.py
        motion_dim=58,
        command_dim=3,
        hidden_dim=512,
        num_experts=4,
        num_layers=3,
    )

# Load weights
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Dummy inputs for both arguments
dummy_obs = torch.zeros(1, 58)
dummy_command = torch.zeros(1, 3)  # e.g., 3 for velocity commands

torch.onnx.export(
    model,
    (dummy_obs, dummy_command),  # tuple of inputs
    os.path.join(ROOT_DIR, "eval/cmg_exported.onnx"),
    input_names=["prev_motion", "command"],
    output_names=["motion"],
    opset_version=11,
)

print(f"cmg_exported.onnx done exporting.")