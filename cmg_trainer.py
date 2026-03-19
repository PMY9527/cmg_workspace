import torch
import torch.nn as nn
from tqdm import tqdm
from module.cmg import CMG


# Configs
START_TEACHER_PROB = 1.0
FINAL_TEACHER_PROB = 0.1
TEACHER_DECAY_RATE = 0.99

INPUT_NOISE_STD = 0.05
AR_ROLLOUT_LEN = 3

class CMGTrainer:
    def __init__(
        self,
        model: CMG,
        lr: float,
        device: str = "cuda",
        standing_pose: torch.Tensor = None,
        standing_swap_prob: float = 0.1,
    ):
        self.model = model.to(device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        self.device = device

        # Standing pose swap: teach standing→walking transitions
        self.standing_pose = standing_pose  # [motion_dim] normalized
        self.standing_swap_prob = standing_swap_prob

        # Scheduled sampling
        self.teacher_prob = START_TEACHER_PROB
        self.teacher_prob_decay = TEACHER_DECAY_RATE
        self.teacher_prob_min = FINAL_TEACHER_PROB

        # AR rollout: backprop through consecutive model predictions
        self.ar_rollout_len = AR_ROLLOUT_LEN  # backprop through 3 AR steps

        # Input noise: teach model to handle imperfect inputs (like AR mode)
        self.input_noise_std = INPUT_NOISE_STD
    
    def train_epoch(self, dataloader, use_scheduled_sampling: bool = True):
        self.model.train()
        total_loss = 0
        
        for batch in tqdm(dataloader, desc="Training"):
            motion_seq = batch["motion"].to(self.device)    # [batch, seq_len, motion_dim]
            command_seq = batch["command"].to(self.device)  # [batch, seq_len, command_dim]
            
            batch_size, seq_len, _ = motion_seq.shape
            loss = 0
            
            # First frame, with optional standing pose swap
            current_motion = motion_seq[:, 0]
            if self.standing_pose is not None and torch.rand(1).item() < self.standing_swap_prob:
                current_motion = self.standing_pose.unsqueeze(0).expand(batch_size, -1)

            ar_steps = 0  # count consecutive AR steps (for gradient truncation)

            for t in range(seq_len - 1):
                command = command_seq[:, t]
                target = motion_seq[:, t + 1]

                # Absolute prediction: model directly outputs next frame
                pred = self.model(current_motion, command)

                loss = loss + nn.functional.mse_loss(pred, target)

                # Scheduled sampling: decide next frame source
                if use_scheduled_sampling and torch.rand(1).item() >= self.teacher_prob:
                    ar_steps += 1
                    if ar_steps >= self.ar_rollout_len:
                        current_motion = pred.detach()
                        ar_steps = 0
                    else:
                        current_motion = pred
                else:
                    current_motion = motion_seq[:, t + 1] + self.input_noise_std * torch.randn_like(motion_seq[:, t + 1])
                    ar_steps = 0
            
            loss = loss / (seq_len - 1)

            self.optimizer.zero_grad()
            loss.backward()

            # Clip gradients to prevent explosion
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()
            
            total_loss += loss.item()
        
        # 衰减teacher probability
        self.teacher_prob = max(self.teacher_prob_min, self.teacher_prob * self.teacher_prob_decay)
        
        return total_loss / len(dataloader)