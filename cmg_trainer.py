import torch
import torch.nn as nn
from tqdm import tqdm
from module.cmg import CMG

class CMGTrainer:
    def __init__(
        self,
        model: CMG,
        lr: float,
        device: str = "cuda",
        standing_pose: torch.Tensor = None,
        standing_swap_prob: float = 0.2,
    ):
        self.model = model.to(device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        self.device = device

        # Standing pose swap: teach standing→walking transitions
        self.standing_pose = standing_pose  # [motion_dim] normalized
        self.standing_swap_prob = standing_swap_prob

        # Scheduled sampling参数
        self.teacher_prob = 1.0
        self.teacher_prob_decay = 0.995  # 每个epoch衰减
        self.teacher_prob_min = 0.05  # Low to force autoregressive learning
    
    def train_epoch(self, dataloader, use_scheduled_sampling: bool = True):
        self.model.train()
        total_loss = 0
        
        for batch in tqdm(dataloader, desc="Training"):
            motion_seq = batch["motion"].to(self.device)    # [batch, seq_len, motion_dim]
            command_seq = batch["command"].to(self.device)  # [batch, seq_len, command_dim]
            
            batch_size, seq_len, _ = motion_seq.shape
            loss = 0
            
            # 第一帧用ground truth, 有概率替换为standing pose来学习transition
            current_motion = motion_seq[:, 0]
            if self.standing_pose is not None and torch.rand(1).item() < self.standing_swap_prob:
                current_motion = self.standing_pose.unsqueeze(0).expand(batch_size, -1)

            for t in range(seq_len - 1):
                command = command_seq[:, t]
                target = motion_seq[:, t + 1]

                # 预测下一帧
                pred = self.model(current_motion, command)

                # 计算损失
                loss = loss + nn.functional.mse_loss(pred, target)

                # Scheduled sampling: 决定下一步用预测还是ground truth
                if use_scheduled_sampling:
                    if torch.rand(1).item() < self.teacher_prob:
                        current_motion = motion_seq[:, t + 1]  # teacher: 用ground truth
                    else:
                        current_motion = pred.detach()  # student: 用自己的预测
                else:
                    current_motion = motion_seq[:, t + 1]
            
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