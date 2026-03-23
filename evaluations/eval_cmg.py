import sys
import os
ROOT_DIR = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT_DIR)

import torch
import numpy as np
from module.cmg import CMG

MODEL_PATH = os.path.join(ROOT_DIR, 'runs/cmg_20260316_162827/cmg_ckpt_350.pt')
DATA_PATH = os.path.join(ROOT_DIR, 'dataloader/cmg_training_data_new.pt')
OUTPUT_PATH = os.path.join(ROOT_DIR, 'evaluations/autoregressive_motion.npz')

VX = 2.5
VY = 0.0
YAW = 0.0
DURATION = 1200 # 60 fps, 20 sec.

# Must match train.py
NUM_EXPERTS = 8

def load_model_and_stats(model_path, data_path, device='cuda'):

    data = torch.load(data_path, weights_only=False)
    print(f"model:{MODEL_PATH}")
    stats = data["stats"]

    model = CMG(
        motion_dim=stats["motion_dim"],
        command_dim=stats["command_dim"],
        hidden_dim=512,
        num_experts=NUM_EXPERTS,
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
    cmd_mean = torch.from_numpy(stats["command_mean"]).to(device)
    cmd_std = torch.from_numpy(stats["command_std"]).to(device)

    current = (torch.from_numpy(init_motion).to(device) - motion_mean) / motion_std

    commands = torch.from_numpy(commands).to(device)
    commands_norm = (commands - cmd_mean) / cmd_std
    
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


def find_init_by_speed(samples, target_vx, target_vy=0.0):
    """Find a training sample whose average command matches the target speed.
    Prefers steady, straight-line clips (low vx variance, low yaw).
    Returns the first frame of the best-matching sample (unnormalized)."""
    best_idx = 0
    best_score = float('inf')
    for i, s in enumerate(samples):
        cmd = s["command"]  # [seq_len, 3]
        avg_vx = cmd[:, 0].mean()
        avg_vy = cmd[:, 1].mean()
        vx_std = cmd[:, 0].std()
        avg_yaw = abs(cmd[:, 2].mean())
        # Speed match + penalize unsteady/turning clips
        score = (avg_vx - target_vx)**2 + (avg_vy - target_vy)**2 + vx_std**2 + avg_yaw**2
        if score < best_score:
            best_score = score
            best_idx = i
    cmd = samples[best_idx]["command"]
    print(f"Init from sample {best_idx} (avg_vx={cmd[:,0].mean():.3f}, vx_std={cmd[:,0].std():.3f}, avg_yaw={cmd[:,2].mean():.3f})")
    return samples[best_idx]["motion"][0].astype(np.float32)


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
    
    # Init from speed-matched sample instead of standing pose
    init_motion = find_init_by_speed(samples, target_vx=VX, target_vy=VY)
    commands = np.tile([VX, VY, YAW], (DURATION, 1)).astype(np.float32)

    print(" ==== Custom Cmd: ==== ")
    print(f"vx: {VX} m/s")
    print(f"vy: {VY} m/s")
    print(f"yaw: {YAW} rad/s")
    print(f"# Frames: {DURATION}, ({DURATION/50:.2f}s)")
    print(f"Model: experts={NUM_EXPERTS}")

    generated = generate_motion(model, init_motion, commands, stats, device)
    
    motion_to_npz(generated, OUTPUT_PATH, fps=60)
    
    print(f"\n=== To Play: ===")
    print(f"python mujoco_player.py autoregressive_motion.npz")