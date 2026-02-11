import torch
import time
import os
from module.cmg import CMG
from cmg_trainer import CMGTrainer
from dataloader.dataloader import get_dataloader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime, timedelta

# Config
DATA_PATH = "dataloader/cmg_training_data.pt"
BATCH_SIZE = 256
NUM_EPOCHS = 1000
SAVE_INTERVAL = 50
LR = 3e-4

# Resume (set to checkpoint path to resume, None to start fresh)
RESUME_PATH ="runs/cmg_20260211_040530/cmg_ckpt_800.pt" # =None to restart
RESUME_LR = None # None

# TensorBoard setup
log_dir = f"runs/cmg_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir)
print(f"TensorBoard logs: {log_dir}")

# DataLoader
dataloader, stats = get_dataloader(DATA_PATH, batch_size=BATCH_SIZE)
print(f"Batches per epoch: {len(dataloader)}")

# Model
motion_dim = stats["motion_dim"]
command_dim = stats["command_dim"]

model = CMG(
    motion_dim=motion_dim,
    command_dim=command_dim,
    hidden_dim=512,
    num_experts=4,
    num_layers=3,
)

# Compute normalized standing pose from sample 22287
raw_data = torch.load(DATA_PATH, weights_only=False)
standing_raw = torch.from_numpy(raw_data["samples"][22287]["motion"][0]).float()
standing_pose = (standing_raw - stats["motion_mean"]) / stats["motion_std"]
standing_pose = standing_pose.cuda()
print(f"Standing pose loaded (sample 22287)")

trainer = CMGTrainer(model, lr=LR, standing_pose=standing_pose)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    trainer.optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6
)

# Resume from checkpoint
start_epoch = 0
if RESUME_PATH:
    ckpt = torch.load(RESUME_PATH, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    trainer.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    start_epoch = ckpt["epoch"] + 1
    if RESUME_LR:
        for pg in trainer.optimizer.param_groups:
            pg['lr'] = RESUME_LR
    trainer.teacher_prob = max(trainer.teacher_prob_min,
                               trainer.teacher_prob_decay ** start_epoch)
    print(f"Resumed from epoch {start_epoch}, teacher_prob={trainer.teacher_prob:.3f}")

# Training
best_loss = float('inf')
start_time = time.time()
print("Starting training...")

for epoch in range(start_epoch, NUM_EPOCHS):
    epoch_start = time.time()
    loss = trainer.train_epoch(dataloader)
    scheduler.step(loss)  
    epoch_time = time.time() - epoch_start
    elapsed = time.time() - start_time
    remaining = (elapsed / (epoch + 1)) * (NUM_EPOCHS - epoch - 1)
    
    current_lr = trainer.optimizer.param_groups[0]['lr']
    # TensorBoard
    writer.add_scalar('metrics/Learning Rate', current_lr, epoch)
    writer.add_scalar('metrics/MSE Loss', loss, epoch)
    writer.add_scalar('metrics/Teacher Prob', trainer.teacher_prob, epoch)
    # writer.add_scalar('Time/epoch_seconds', epoch_time, epoch)
    
    remaining_h = int(remaining // 3600)
    remaining_m = int((remaining % 3600) // 60)
    remaining_s = int(remaining % 60)
    
    # Console
    print(f"[{epoch+1:4d}/{NUM_EPOCHS}] "
          f"Loss: {loss:.4f} | "
          f"Teacher: {trainer.teacher_prob:.3f} | "
          f"Time: {epoch_time:.1f}s | "
          f"ETA: {remaining_h:02d}:{remaining_m:02d}:{remaining_s:02d}")
    
    # Save best
    if loss < best_loss:
        best_loss = loss
        torch.save(model.state_dict(), f"{log_dir}/cmg_best.pt")
        print(f"  -> New best: {best_loss:.4f}")
    
    # Checkpoint
    if (epoch + 1) % SAVE_INTERVAL == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': trainer.optimizer.state_dict(),
            'loss': loss,
            'stats': stats,
        }, f"{log_dir}/cmg_ckpt_{epoch+1}.pt")

# Final
torch.save({
    'model_state_dict': model.state_dict(),
    'stats': stats,
}, f"{log_dir}/cmg_final.pt")

writer.close()
print(f"\nDone! Total: {(time.time()-start_time)/60:.1f}min, Best: {best_loss:.4f}")
print(f"Models saved to: {log_dir}/")