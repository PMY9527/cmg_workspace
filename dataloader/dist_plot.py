"""
绘制数据集速度分布和Motion数据分布
包含归一化前后的对比
"""
import torch
import numpy as np
import matplotlib.pyplot as plt

# 加载数据
data_path = "cmg_training_data.pt"
data = torch.load(data_path, weights_only=False)
samples = data["samples"]
stats = data["stats"]

print(f"(Preprocessed) Loaded {len(samples)} samples")
print(f"Stats keys: {stats.keys()}")

# 提取所有原始数据
all_vx = []
all_motions_raw = []

for s in samples:
    motion = s["motion"]  # [seq_len+1, 58] - 原始数据
    cmd = s["command"]    # [seq_len, 3] - 原始数据

    all_vx.append(cmd[:, 0])  # vx
    all_motions_raw.append(motion)

all_vx = np.concatenate(all_vx)
all_motions_raw = np.concatenate(all_motions_raw, axis=0)  # [total_frames, 58]

print(f"Total frames: {len(all_vx)}")
print(f"vx range: [{all_vx.min():.2f}, {all_vx.max():.2f}] m/s")
print(f"Motion shape: {all_motions_raw.shape}")

# 归一化 motion
motion_mean = stats["motion_mean"]
motion_std = stats["motion_std"]
motion_std_clipped = np.maximum(motion_std, 0.1)  # 和你的 dataloader 一致

all_motions_norm = (all_motions_raw - motion_mean) / motion_std_clipped

# ========== 图1: 速度分布 (保持原样) ==========
fig, ax = plt.subplots(figsize=(10, 6))

counts, bins, patches = ax.hist(all_vx, bins=50, edgecolor='black', alpha=0.7)
percentages = counts / len(all_vx) * 100
ax.clear()
ax.bar(bins[:-1], percentages, width=np.diff(bins), edgecolor='black', alpha=0.7, align='edge')

ax.set_xlabel('Body Frame Longitudinal Speed vx (m/s)', fontsize=12)
ax.set_ylabel('Percentage (%)', fontsize=12)
ax.set_title('Speed Distribution in Training Data', fontsize=14)
ax.grid(True, alpha=0.3)

mean_vx = all_vx.mean()
std_vx = all_vx.std()
ax.axvline(mean_vx, color='r', linestyle='--', label=f'Mean: {mean_vx:.2f} m/s')
ax.axvline(mean_vx + std_vx, color='orange', linestyle=':', label=f'Std: {std_vx:.2f} m/s')
ax.axvline(mean_vx - std_vx, color='orange', linestyle=':')
ax.legend()

plt.tight_layout()
plt.savefig('speed_distribution.png', dpi=150)
print("Saved to speed_distribution.png")

# 打印分段统计
print("\n=== Speed Distribution ===")
bins_custom = [-4, -2, -1, 0, 1, 2, 4]
for i in range(len(bins_custom)-1):
    mask = (all_vx >= bins_custom[i]) & (all_vx < bins_custom[i+1])
    pct = mask.sum() / len(all_vx) * 100
    print(f"  [{bins_custom[i]:+.0f}, {bins_custom[i+1]:+.0f}) m/s: {pct:.1f}%")


print("\n=== Motion Statistics (Normalized) ===")
print(f"Overall - Mean: {all_motions_norm.mean():.4f}, Std: {all_motions_norm.std():.4f}")
print(f"        - Range: [{all_motions_norm.min():.2f}, {all_motions_norm.max():.2f}]")
print(f"Values > 5:  {(np.abs(all_motions_norm) > 5).sum() / all_motions_norm.size * 100:.2f}%")
print(f"Values > 10: {(np.abs(all_motions_norm) > 10).sum() / all_motions_norm.size * 100:.2f}%")

# 找出 std 太小的维度
print("\n=== Dimensions with Small Std (< 0.1) ===")
small_std_dims = np.where(motion_std < 0.1)[0]
print(f"Found {len(small_std_dims)} dimensions:")
for dim in small_std_dims[:10]:  # 只显示前10个
    print(f"  Dim {dim}: std = {motion_std[dim]:.6f}")

plt.show()