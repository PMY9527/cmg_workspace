import sys
import os
ROOT_DIR = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT_DIR)

import torch
import numpy as np
from module.cmg import CMG

MODEL_PATH = os.path.join(ROOT_DIR, 'runs/cmg_20260211_040530/cmg_ckpt_800.pt')
DATA_PATH = os.path.join(ROOT_DIR, 'dataloader/cmg_training_data.pt')
OUTPUT_PATH = os.path.join(ROOT_DIR, 'evaluations/autoregressive_motion.npz')

VX = 1.5 # fails for all vx < 1.2 m/s fuck this shit
VY = 0.0
YAW = 0.0
DURATION = 1200 # 60 fps, 20 sec.

def load_model_and_stats(model_path, data_path, device='cuda'):
    
    data = torch.load(data_path, weights_only=False)
    stats = data["stats"]
    
    model = CMG( # the same as train.py
        motion_dim=stats["motion_dim"],
        command_dim=stats["command_dim"],
        hidden_dim=512,
        num_experts=4,
        num_layers=3,
    )

    # model.load_state_dict(torch.load(model_path, weights_only=True)) # cmg_best

    checkpoint = torch.load(model_path, weights_only=False) # checkpoints
    model.load_state_dict(checkpoint["model_state_dict"])

    model = model.to(device)
    model.eval()
    
    return model, stats, data["samples"]


def generate_motion(model, init_motion, commands, stats, device='cuda'):
    """
    
    Args:
        model: CMG
        init_motion: 初始帧 [58] (未归一化)
        commands: 命令序列 [T, 3] (未归一化)
        stats: stats
    }
    
    Returns:
        generated: [T+1, 58] 生成的动作 (未归一化)
    """
    motion_mean = torch.from_numpy(stats["motion_mean"]).to(device)
    motion_std = torch.from_numpy(stats["motion_std"]).to(device)
    cmd_min = torch.from_numpy(stats["command_min"]).to(device)
    cmd_max = torch.from_numpy(stats["command_max"]).to(device)
    
    current = (torch.from_numpy(init_motion).to(device) - motion_mean) / motion_std # 归一化初始帧
    
    commands = torch.from_numpy(commands).to(device)
    commands_norm = (commands - cmd_min) / (cmd_max - cmd_min) * 2 - 1 #  归一化cmd [min, max] → [-1, 1]
    
    generated = [current.clone()]
    
    with torch.no_grad():
        for t in range(len(commands_norm)):
            cmd = commands_norm[t:t+1]
            curr = current.unsqueeze(0)

            pred = model(curr, cmd)
            current = pred.squeeze(0)  # Autoregressive
            generated.append(current.clone())
    
    generated = torch.stack(generated)
    generated = generated * motion_std + motion_mean
    
    return generated.cpu().numpy()


def motion_to_npz(motion, output_path, fps):

    T = motion.shape[0]
    
    dof_positions = motion[:, :29].astype(np.float32)
    dof_velocities = motion[:, 29:].astype(np.float32)
    
    body_positions = np.zeros((T, 30, 3), dtype=np.float32)
    body_positions[:, 0, 2] = 0.75
    
    body_rotations = np.zeros((T, 30, 4), dtype=np.float32)
    body_rotations[:, :, 0] = 1.0
    
    np.savez(
        output_path,
        fps=np.array([fps], dtype=np.float32),
        dof_positions=dof_positions,
        dof_velocities=dof_velocities,
        body_positions=body_positions,
        body_rotations=body_rotations,
        dof_names=np.array([f"joint_{i}" for i in range(29)]),
        body_names=np.array([f"body_{i}" for i in range(30)]),
    )
    print(f"Saved to {output_path}")

if __name__ == "__main__":

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    model, stats, samples = load_model_and_stats(MODEL_PATH, DATA_PATH, device)
    
    # First frame
    init_motion = np.zeros(58, dtype=np.float32)
    commands = np.tile([VX, VY, YAW], (DURATION, 1)).astype(np.float32)
    
    print(" ==== Custom Cmd: ==== ")
    print(f"Init with zeros")
    print(f"vx: {VX} m/s")
    print(f"vy: {VY} m/s")
    print(f"yaw: {YAW} rad/s")
    print(f"# Frames: {DURATION}, ({DURATION/50:.2f}s)")

    generated = generate_motion(model, init_motion, commands, stats, device)
    
    motion_to_npz(generated, OUTPUT_PATH, fps=60)
    
    print(f"\n=== To Play: ===")
    print(f"python mujoco_player.py autoregressive_motion.npz")