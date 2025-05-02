# --- evaluation_model class (no changes needed from previous version) ---
import torch
import torch.nn as nn
import torchvision.models as models
from pathlib import Path

class evaluation_model():
    def __init__(self, device='cuda'):
        ckpt_path = Path('file/checkpoint.pth')
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Evaluation model checkpoint not found at {ckpt_path}")
        self.device = torch.device(device)
        print(f"Initializing evaluation model on device: {self.device}")
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        self.resnet18 = models.resnet18(weights=None)
        self.resnet18.fc = nn.Sequential(nn.Linear(512,24), nn.Sigmoid())
        self.resnet18.load_state_dict(checkpoint['model'])
        self.resnet18 = self.resnet18.to(self.device)
        self.resnet18.eval()
        self.classnum = 24
    def compute_acc(self, out, onehot_labels):
        out = out.to(self.device); onehot_labels = onehot_labels.to(self.device)
        batch_size = out.size(0); acc = 0; total = 0
        for i in range(batch_size):
            k = int(onehot_labels[i].sum().item());
            if k == 0: continue
            total += k
            outv, outi = out[i].topk(k); lv, li = onehot_labels[i].topk(k)
            for j in outi:
                if j in li: acc += 1
        return acc / total if total > 0 else 0.0
    def eval(self, images, labels):
        images = images.to(self.device)
        with torch.no_grad():
            out = self.resnet18(images)
            acc = self.compute_acc(out, labels)
            return acc

# --- Main Script Imports ---
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from argparse import ArgumentParser
from diffusers import DDPMScheduler
from torch.utils.data import DataLoader, Dataset
# from file.evaluator import evaluation_model # Defined above now
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
from diffusers.optimization import get_scheduler
import torchvision.transforms as T
import json
from pathlib import Path
from PIL import Image
import time # For default wandb run name
import random # For set_seed

import wandb # <<< Import WANDB

# --- AttentionBlock, SinusoidalTimeEmbedding, ResidualBlock, iCLEVRDataset (no changes needed) ---
class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        num_groups = 8 if channels >= 32 else max(1, channels // 4)
        self.norm = nn.GroupNorm(num_groups, channels)
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.proj_out = nn.Sequential(nn.Conv2d(channels, channels, 1), nn.GroupNorm(num_groups, channels))
    def forward(self, x):
        B, C, H, W = x.shape; h = self.norm(x)
        q = self.q(h).reshape(B, C, H * W).permute(0, 2, 1)
        k = self.k(h).reshape(B, C, H * W)
        v = self.v(h).reshape(B, C, H * W).permute(0, 2, 1)
        attn = torch.softmax(torch.bmm(q, k) * (C ** -0.5), dim=-1)
        out = torch.bmm(attn, v).permute(0, 2, 1).reshape(B, C, H, W)
        out = self.proj_out(out); return x + out

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__(); self.dim = dim
    def forward(self, t):
        if self.dim % 2 != 0: raise ValueError(f"Embedding dimension must be even, got {self.dim}")
        half_dim = self.dim // 2
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -(torch.log(torch.tensor(10000.0, device=t.device)) / (half_dim - 1)))
        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=1); return emb

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, cond_emb_dim):
        super().__init__(); self.same_channels = (in_channels == out_channels)
        def get_norm(c):
             if c <= 0: return nn.Identity()
             num_groups = 8
             while c % num_groups != 0 and num_groups > 1: num_groups //= 2
             if num_groups == 1 and c > 1:
                 num_groups = 4
                 while c % num_groups != 0 and num_groups > 1: num_groups //= 2
             if c % num_groups != 0: num_groups = 1
             return nn.GroupNorm(num_groups, c)
        self.conv1 = nn.Sequential(get_norm(in_channels), nn.SiLU(), nn.Conv2d(in_channels, out_channels, 3, padding=1))
        self.conv2 = nn.Sequential(get_norm(out_channels), nn.SiLU(), nn.Conv2d(out_channels, out_channels, 3, padding=1))
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, out_channels))
        self.cond_mlp = nn.Sequential(nn.SiLU(), nn.Linear(cond_emb_dim, out_channels))
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1) if not self.same_channels else nn.Identity()
    def forward(self, x, t_emb, c_emb):
        h = self.conv1(x)
        time_out = self.time_mlp(t_emb); cond_out = self.cond_mlp(c_emb)
        h = h + time_out[:, :, None, None] + cond_out[:, :, None, None]
        h = self.conv2(h); out = (h + self.shortcut(x)) * (1 / 1.414); return out

class iCLEVRDataset(Dataset):
    def __init__(self, image_dir, json_path, object_map, transform=None):
        self.image_dir = Path(image_dir); self.transform = transform
        if not isinstance(object_map, dict): raise TypeError(f"object_map must be a dictionary, got {type(object_map)}")
        self.object_map = object_map; self.num_classes = len(object_map); self.json_path = Path(json_path)
        print(f"Loading data from: {self.json_path}")
        try:
            with open(self.json_path, 'r') as f: self.data = json.load(f)
        except FileNotFoundError: print(f"Error: JSON file not found at {self.json_path}"); raise
        except json.JSONDecodeError: print(f"Error: Could not decode JSON from {self.json_path}"); raise
        if isinstance(self.data, dict):
            self.image_files = list(self.data.keys()); self.labels_list = [self.data[fname] for fname in self.image_files]
            print(f"Loaded {len(self.image_files)} samples from dictionary format.")
        elif isinstance(self.data, list):
            num_samples = len(self.data); self.image_files = [f"{i}.png" for i in range(num_samples)]; self.labels_list = self.data
            print(f"Loaded {num_samples} samples from list format (assuming names 0.png, 1.png...).")
        else: raise ValueError(f"Unsupported JSON format in {self.json_path}. Expected dict or list.")
        valid_indices = [i for i, fname in enumerate(self.image_files) if (self.image_dir / fname).exists()]
        if len(valid_indices) < len(self.image_files):
             original_count = len(self.image_files)
             self.image_files = [self.image_files[i] for i in valid_indices]; self.labels_list = [self.labels_list[i] for i in valid_indices]
             print(f"Warning: Filtered out {original_count - len(self.image_files)} missing image files. {len(self.image_files)} remain.")
        print(f"Dataset initialized with {len(self.image_files)} valid samples.")
    def __len__(self): return len(self.image_files)
    def __getitem__(self, idx):
        if idx >= len(self.image_files): raise IndexError(f"Index {idx} out of bounds for dataset length {len(self.image_files)}")
        img_name = self.image_dir / self.image_files[idx]
        try: image = Image.open(img_name).convert('RGB')
        except FileNotFoundError:
             print(f"Error: Image file not found {img_name} during __getitem__.")
             dummy_image = torch.zeros((3, 64, 64)); dummy_label = torch.zeros(self.num_classes, dtype=torch.float32); return dummy_image, dummy_label
        labels = self.labels_list[idx]; label_vector = torch.zeros(self.num_classes, dtype=torch.float32)
        if not isinstance(labels, list): print(f"Warning: Unexpected label format for image {self.image_files[idx]}. Expected list, got {type(labels)}. Labels: {labels}")
        else:
            for obj_name in labels:
                if obj_name in self.object_map: label_vector[self.object_map[obj_name]] = 1.0
                else: print(f"Warning: Unknown object '{obj_name}' in labels for image {self.image_files[idx]}. Not found in object_map.")
        if self.transform: image = self.transform(image)
        return image, label_vector

# --- SimpleConditionalUNet (no changes needed) ---
class SimpleConditionalUNet(nn.Module):
    def __init__(self, num_classes, base_channels=64, time_emb_dim=256, cond_emb_dim=256):
        super().__init__(); self.num_classes = num_classes
        self.time_embedding = SinusoidalTimeEmbedding(time_emb_dim)
        self.time_mlp = nn.Sequential(nn.Linear(time_emb_dim, time_emb_dim), nn.SiLU(), nn.Linear(time_emb_dim, time_emb_dim))
        self.class_embedding = nn.Sequential(nn.Linear(self.num_classes, cond_emb_dim), nn.SiLU(), nn.Linear(cond_emb_dim, cond_emb_dim))
        self.enc1 = ResidualBlock(3, base_channels, time_emb_dim, cond_emb_dim)
        self.enc2 = ResidualBlock(base_channels, base_channels * 2, time_emb_dim, cond_emb_dim)
        self.enc3 = ResidualBlock(base_channels * 2, base_channels * 4, time_emb_dim, cond_emb_dim)
        self.attn1 = AttentionBlock(base_channels * 4)
        self.bottleneck = ResidualBlock(base_channels * 4, base_channels * 4, time_emb_dim, cond_emb_dim)
        self.dec3 = ResidualBlock(base_channels * 8, base_channels * 2, time_emb_dim, cond_emb_dim)
        self.attn2 = AttentionBlock(base_channels * 2)
        self.dec2 = ResidualBlock(base_channels * 4, base_channels, time_emb_dim, cond_emb_dim)
        self.dec1 = ResidualBlock(base_channels * 2, base_channels, time_emb_dim, cond_emb_dim)
        self.downsample = nn.AvgPool2d(2); self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.final_norm = nn.GroupNorm(max(1, base_channels // 8), base_channels); self.final_act = nn.SiLU(); self.final_conv = nn.Conv2d(base_channels, 3, kernel_size=1)
    def forward(self, x, t, label_onehot):
        t_emb = self.time_embedding(t); t_emb = self.time_mlp(t_emb); c_emb = self.class_embedding(label_onehot)
        x1 = self.enc1(x, t_emb, c_emb); x2 = self.enc2(self.downsample(x1), t_emb, c_emb); x3 = self.enc3(self.downsample(x2), t_emb, c_emb)
        x3_attn = self.attn1(x3); b = self.bottleneck(self.downsample(x3_attn), t_emb, c_emb)
        u3 = self.upsample(b); cat3 = torch.cat([u3, x3_attn], dim=1); d3 = self.dec3(cat3, t_emb, c_emb); d3_attn = self.attn2(d3)
        u2 = self.upsample(d3_attn); cat2 = torch.cat([u2, x2], dim=1); d2 = self.dec2(cat2, t_emb, c_emb)
        u1 = self.upsample(d2); cat1 = torch.cat([u1, x1], dim=1); d1 = self.dec1(cat1, t_emb, c_emb)
        out = self.final_norm(d1); out = self.final_act(out); out = self.final_conv(out); return out

# --- get_random_timesteps, save_checkpoint (no changes needed) ---
def get_random_timesteps(batch_size, total_timesteps, device):
    return torch.randint(0, total_timesteps, (batch_size,), device=device).long()
def save_checkpoint(model, optimizer, epoch, save_dir, is_best=False):
    os.makedirs(save_dir, exist_ok=True)
    filename = Path(save_dir) / (f"best_checkpoint.pth" if is_best else f"checkpoint_epoch_{epoch}.pth")
    print(f"Saving {'best' if is_best else ''} checkpoint to {filename} (epoch {epoch})")
    torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'epoch': epoch}, filename)

# --- train_ddpm (MODIFIED FOR WANDB) ---
def train_ddpm(model, dataloader, optimizer, lr_scheduler, device, epochs, # Added lr_scheduler
               noise_scheduler_train, # Renamed scheduler
               eval_freq, eval_dataloader, eval_model_instance,
               noise_scheduler_eval, num_eval_timesteps,
               save_every, save_dir, output_dir_eval,
               gradient_accumulation_steps): # Added grad accum steps

    best_loss = float('inf')
    best_eval_acc = -1.0
    acc_threshold = 0.8
    loss_list = []
    eval_acc_list = []
    eval_epochs = []
    start_epoch = 0
    global_step = 0 # <<< Initialize Global Step

    # --- Checkpoint Resuming Logic ---
    latest_checkpoint_path = None
    save_dir_path = Path(save_dir)
    if save_dir_path.exists():
        checkpoints = sorted(
            [f for f in os.listdir(save_dir_path) if f.startswith("checkpoint_epoch_") and f.endswith(".pth")],
            key=lambda x: int(x.split('_')[-1].split('.')[0])
        )
        if checkpoints: latest_checkpoint_path = save_dir_path / checkpoints[-1]

    if latest_checkpoint_path:
        print(f"Resuming training from checkpoint: {latest_checkpoint_path}")
        checkpoint = torch.load(latest_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        # TODO: Add logic to resume global_step and lr_scheduler state if needed
        # global_step = checkpoint.get('global_step', 0)
        # lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        print(f"Resumed from epoch {start_epoch - 1}")
    # --- End Resuming Logic ---

    model.train()
    num_training_steps = (len(dataloader) // gradient_accumulation_steps) * epochs # Recalculate for progress bar

    # Outer progress bar for epochs
    epoch_pbar = tqdm(range(start_epoch, epochs), desc="Epochs")

    for epoch in epoch_pbar:
        saved_best_this_epoch = False
        running_loss = 0.0
        # Inner progress bar for steps within an epoch
        step_pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        optimizer.zero_grad() # Zero gradients at the start of epoch

        for step, (images, labels) in enumerate(step_pbar):
            images = images.to(device)
            labels = labels.to(device)
            batch_size = images.size(0)
            timesteps = get_random_timesteps(batch_size, noise_scheduler_train.config.num_train_timesteps, device)
            noise = torch.randn_like(images)
            noisy_images = noise_scheduler_train.add_noise(images, noise, timesteps)

            # Predict noise
            outputs = model(noisy_images, timesteps, labels)
            loss = F.mse_loss(outputs, noise)

            # Scale loss for accumulation
            loss = loss / gradient_accumulation_steps
            loss.backward()
            running_loss += loss.item() * gradient_accumulation_steps # Accumulate unscaled loss for logging

            # Optimizer step check
            if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(dataloader):
                # Clip gradients (optional but recommended)
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm) # Need max_grad_norm arg if used

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad() # Zero gradients after step

                # <<< Log to WandB >>>
                current_lr = lr_scheduler.get_last_lr()[0]
                wandb.log({"train_loss": loss.item() * gradient_accumulation_steps, # Log unscaled loss
                           "lr": current_lr},
                          step=global_step)
                global_step += 1
                # Update inner progress bar postfix less frequently
                step_pbar.set_postfix(Loss=f"{loss.item() * gradient_accumulation_steps:.4f}", LR=f"{current_lr:.1e}")


        # --- End Epoch ---
        avg_epoch_loss = running_loss / len(dataloader) # Calculate average loss for the epoch
        loss_list.append(avg_epoch_loss)
        epoch_pbar.set_postfix(AvgLoss=f"{avg_epoch_loss:.4f}") # Update epoch progress bar

        # --- Periodic Evaluation ---
        if (epoch + 1) % eval_freq == 0 or epoch == epochs - 1:
            print(f"\n--- Running evaluation for epoch {epoch} ---")
            model.eval()

            current_eval_acc_list = inference(
                dataloader=eval_dataloader, noise_scheduler=noise_scheduler_eval,
                timesteps_to_run=num_eval_timesteps, model=model,
                eval_model=eval_model_instance, save_prefix=f'eval_epoch_{epoch}',
                output_dir=output_dir_eval, device=device
            )
            avg_eval_acc = np.mean(current_eval_acc_list) if current_eval_acc_list else 0.0

            if current_eval_acc_list:
                eval_acc_list.append(avg_eval_acc)
                eval_epochs.append(epoch)
                print(f"--- Evaluation Epoch {epoch} Avg Accuracy: {avg_eval_acc:.4f} ---")

                # <<< Log Evaluation Accuracy to WandB >>>
                wandb.log({"mean_accuracy": avg_eval_acc, "epoch": epoch}, step=global_step)

                # --- Accuracy-based Saving Logic ---
                if avg_eval_acc > acc_threshold and avg_eval_acc > best_eval_acc:
                    print(f"--- !!! New best evaluation accuracy (>{acc_threshold:.2f}): {avg_eval_acc:.4f} (previous best: {best_eval_acc:.4f}) !!! ---")
                    best_eval_acc = avg_eval_acc
                    save_checkpoint(model, optimizer, epoch, save_dir, is_best=True)
                    saved_best_this_epoch = True
                elif avg_eval_acc > best_eval_acc:
                    print(f"--- Evaluation accuracy improved but below threshold {acc_threshold:.2f}: {avg_eval_acc:.4f} (previous best: {best_eval_acc:.4f}) ---")
                    best_eval_acc = avg_eval_acc
                else:
                    print(f"--- Evaluation accuracy {avg_eval_acc:.4f} did not improve or meet threshold (best >{acc_threshold:.2f}: {best_eval_acc:.4f}) ---")
            else:
                print("--- Evaluation failed or produced no results ---")
                # <<< Log placeholder if eval fails >>>
                wandb.log({"eval_accuracy": 0.0, "epoch": epoch}, step=global_step)

            model.train() # Set back to train mode
            print("--------------------------------------------\n")
        # --- End Periodic Evaluation ---

        # --- Regular Checkpoint Saving ---
        if (epoch + 1) % save_every == 0 or epoch == epochs - 1:
            if not saved_best_this_epoch:
                save_checkpoint(model, optimizer, epoch, save_dir, is_best=False)
        # --- End Regular Saving ---

    # --- End Training Loop ---

    # --- Plotting ---
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1); plt.plot(range(start_epoch, epochs), loss_list)
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Training Loss Curve'); plt.grid(True)
    if eval_epochs:
        plt.subplot(1, 2, 2); plt.plot(eval_epochs, eval_acc_list, marker='o')
        plt.xlabel('Epoch'); plt.ylabel('Evaluation Accuracy'); plt.title(f'Evaluation Accuracy Curve (Threshold: {acc_threshold:.2f})')
        plt.axhline(y=acc_threshold, color='r', linestyle='--', label=f'Threshold ({acc_threshold:.2f})'); plt.legend(); plt.grid(True)
    plt.tight_layout(); plot_curve_path = Path(save_dir) / "training_curves.png"
    plt.savefig(plot_curve_path); print(f"Training curves saved to {plot_curve_path}"); plt.close()

# --- inference function (MODIFIED to not save images during eval call from train) ---
def inference(dataloader, noise_scheduler, timesteps_to_run, model, eval_model,
              save_prefix='test', output_dir='res', device='cuda',
              save_images=True): # <<< Add flag to control image saving
    output_dir = Path(output_dir)
    if save_images: # Only create dir if saving images
        output_dir.mkdir(exist_ok=True)

    all_results_acc = []
    model.eval()
    noise_scheduler.set_timesteps(timesteps_to_run)

    progress_bar = tqdm(dataloader, desc=f"Inference {save_prefix}", leave=False) # Set leave=False for nested loops
    batch_idx_processed = 0

    for idx, conditions_batch in enumerate(progress_bar):
        y_batch = conditions_batch.to(device)
        current_batch_size = y_batch.shape[0]
        image_size = 64
        x_batch = torch.randn(current_batch_size, 3, image_size, image_size).to(device)

        denoising_result_grid_img0 = [] # Only needed if save_images is True
        denoising_steps_to_save = 10
        save_interval = max(1, timesteps_to_run // denoising_steps_to_save)

        # Denoising loop
        for i, t in enumerate(noise_scheduler.timesteps):
            with torch.no_grad():
                t_batch_val = t.expand(current_batch_size).to(device)
                noise_pred_batch = model(x_batch, t_batch_val, y_batch)
            scheduler_output = noise_scheduler.step(noise_pred_batch, t, x_batch)
            x_batch = scheduler_output.prev_sample

            # --- Save intermediate step (only if saving images) ---
            if save_images and (i % save_interval == 0 or i == timesteps_to_run - 1):
                img0_to_save = (x_batch[0].clamp(-1, 1) + 1) / 2
                denoising_result_grid_img0.append(img0_to_save.cpu())

        # Final generated batch
        final_batch_norm = x_batch
        final_batch_save = (x_batch.clamp(-1, 1) + 1) / 2 # Denormalize needed only if saving

        # Evaluate the whole batch
        current_acc_batch = eval_model.eval(final_batch_norm, y_batch)
        all_results_acc.append(current_acc_batch)
        progress_bar.set_postfix_str(f'Batch: {idx}, Avg Batch Acc: {current_acc_batch:.4f}')

        # --- Save visualization grid (only if saving images and it's the first batch) ---
        if save_images and batch_idx_processed == 0 and denoising_result_grid_img0:
            grid_image = make_grid(denoising_result_grid_img0, nrow=len(denoising_result_grid_img0), pad_value=0.5)
            save_image(grid_image, output_dir / f'{save_prefix}_steps_batch{idx}_img0.png')

        # --- Save final results grid (only if saving images and it's the first batch) ---
        if save_images and batch_idx_processed == 0:
            final_grid_first_batch = make_grid(final_batch_save, nrow=current_batch_size, pad_value=0.5)
            save_image(final_grid_first_batch, output_dir / f'{save_prefix}_final_batch{idx}.png')

        batch_idx_processed += 1

    return all_results_acc

# --- LabelDataset (no changes needed) ---
class LabelDataset(Dataset):
     def __init__(self, json_path, object_map):
          self.json_path = Path(json_path); self.object_map = object_map; self.num_classes = len(object_map)
          if not isinstance(object_map, dict): raise TypeError("object_map must be a dictionary")
          print(f"Loading labels from: {self.json_path}")
          try:
               with open(self.json_path, 'r') as f: self.labels_list = json.load(f)
          except FileNotFoundError: print(f"Error: JSON file not found at {self.json_path}"); raise
          except json.JSONDecodeError: print(f"Error: Could not decode JSON from {self.json_path}"); raise
          if not isinstance(self.labels_list, list): raise ValueError(f"Expected {self.json_path} to contain a list of labels.")
          print(f"Loaded {len(self.labels_list)} labels.")
     def __len__(self): return len(self.labels_list)
     def __getitem__(self, idx):
          labels = self.labels_list[idx]; label_vector = torch.zeros(self.num_classes, dtype=torch.float32)
          if not isinstance(labels, list): print(f"Warning: Unexpected label format index {idx}")
          else:
               for obj_name in labels:
                    if obj_name in self.object_map: label_vector[self.object_map[obj_name]] = 1.0
                    else: print(f"Warning: Unknown object '{obj_name}' index {idx}")
          return label_vector

# --- Argument Parser (MODIFIED FOR WANDB) ---
def arg_parser():
    parser = ArgumentParser()
    # Add Training Args from previous version...
    parser.add_argument('--lr', type=float, default=5e-3, help="Learning rate")
    parser.add_argument('--num_epochs', type=int, default=550, help="Number of training epochs")
    parser.add_argument('--total_timesteps', type=int, default=1000, help="Number of diffusion timesteps for training")
    parser.add_argument('--beta_schedule', type=str, default='linear', choices=['linear', 'cosine', 'squaredcos_cap_v2'], help="Noise schedule")
    parser.add_argument('--batch_size', type=int, default=256, help="Batch size for training")
    parser.add_argument('--num_workers', type=int, default=4, help="Number of dataloader workers")
    parser.add_argument('--save_dir', type=str, default='checkpoints_unet', help="Directory to save checkpoints")
    parser.add_argument('--save_freq', type=int, default=10, help="Frequency (epochs) to save checkpoints")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help="Gradient accumulation steps")

    # Add Evaluation Args from previous version...
    parser.add_argument('--eval_freq', type=int, default=3, help="Frequency (epochs) to run evaluation during training")
    parser.add_argument('--eval_json', type=str, default='file/test.json', help="Path to JSON file for evaluation during training")
    parser.add_argument('--eval_batch_size', type=int, default=4, help="Batch size for generating images during evaluation")

    # Add Inference Args from previous version...
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'], help="Run mode: train or test")
    parser.add_argument('--ckpt', type=str, default='checkpoints_unet/best_checkpoint.pth', help="Path to checkpoint for testing")
    parser.add_argument('--test_timesteps', type=int, default=200, help="Number of timesteps for inference/evaluation generation")
    parser.add_argument('--output_dir', type=str, default='res', help="Directory to save inference/evaluation results")

    # Add Common Args from previous version...
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help="Device to use (cuda/cpu)")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility")

    # Add Data Args from previous version...
    parser.add_argument('--data_dir', type=str, default='data/iclevr', help="Path to image directory")
    parser.add_argument('--object_dict_path', type=str, default='file/objects.json', help="Path to object mapping JSON")
    parser.add_argument('--train_json', type=str, default='file/train.json', help="Path to training labels JSON")
    parser.add_argument('--test_json', type=str, default='file/test.json', help="Path to test labels JSON")
    parser.add_argument('--new_test_json', type=str, default='file/new_test.json', help="Path to new test labels JSON")
    parser.add_argument('--image_size', type=int, default=64, help="Image size (width and height)")

    # --- Add WandB Args ---
    parser.add_argument("--wandb_project", type=str, default="DLP-Lab6-DDPM", help="WandB project name")
    parser.add_argument("--wandb_run_name", type=str, default=f"ddpm_simpleU_{time.strftime('%Y%m%d_%H%M%S')}", help="WandB run name")

    # Add AdamW args from example (if needed, otherwise defaults are fine)
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="AdamW beta1")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="AdamW beta2")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="AdamW weight decay")
    parser.add_argument("--adam_epsilon", type=float, default=1e-8, help="AdamW epsilon")

    # Add LR Scheduler args from example
    parser.add_argument("--lr_scheduler_type", type=str, default="linear", help="Learning rate scheduler type")
    parser.add_argument("--lr_warmup_steps", type=int, default=0, help="Number of warmup steps for LR scheduler") # Default 0 if no warmup needed

    args = parser.parse_args()
    return args

# --- set_seed function ---
def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

# --- Main Function (MODIFIED FOR WANDB) ---
def main():
    args = arg_parser()
    set_seed(args.seed) # <<< Set seed
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # <<< Initialize WandB >>>
    try:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args), # Log all args
            save_code=True, # Save main script to wandb
        )
        print("WandB initialized successfully.")
    except Exception as e:
        print(f"Error initializing WandB: {e}. Proceeding without WandB.")
        # Optionally disable wandb logging if init fails by setting wandb.log to a dummy function
        # wandb.log = lambda *args, **kwargs: None

    # --- Load Object Map ---
    object_dict_path = Path(args.object_dict_path)
    try:
        with open(object_dict_path, 'r') as f: object_map = json.load(f)
        num_classes = len(object_map)
        print(f"Loaded object map with {num_classes} classes.")
    except Exception as e: print(f"Error loading object map: {e}"); sys.exit(1)

    # --- Initialize Model and Schedulers ---
    model = SimpleConditionalUNet(num_classes=num_classes).to(device)
    noise_scheduler_train = DDPMScheduler(num_train_timesteps=args.total_timesteps, beta_schedule=args.beta_schedule)
    noise_scheduler_eval = DDPMScheduler(num_train_timesteps=args.total_timesteps, beta_schedule=args.beta_schedule)
    print(f"Initialized model and schedulers.")

    # --- Training Mode ---
    if args.mode == "train":
        print("--- Starting Training ---")
        data_dir = Path(args.data_dir); train_json_path = Path(args.train_json)
        train_transform = T.Compose([ T.Resize((args.image_size, args.image_size)), T.ToTensor(), T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) ])

        # Training Dataset and Dataloader
        try:
            train_dataset = iCLEVRDataset(image_dir=data_dir, json_path=train_json_path, object_map=object_map, transform=train_transform)
        except Exception as e: print(f"Error creating training dataset: {e}"); sys.exit(1)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=device.type == "cuda")
        print(f"Training Dataloader created.")

        # --- Setup for Evaluation During Training ---
        print("--- Setting up evaluation components ---")
        eval_json_path = Path(args.eval_json)
        output_dir_eval = Path(args.output_dir) / "eval_during_train"
        output_dir_eval.mkdir(parents=True, exist_ok=True)
        eval_model_instance = None # Initialize
        eval_dataloader = None
        try:
            eval_model_instance = evaluation_model(device=args.device)
            eval_label_dataset = LabelDataset(json_path=eval_json_path, object_map=object_map)
            eval_dataloader = DataLoader(eval_label_dataset, batch_size=args.eval_batch_size, shuffle=False)
            print(f"Evaluation components created.")
        except Exception as e: print(f"Warning: Error setting up evaluation: {e}. Evaluation will be skipped.")
        # --- End Evaluation Setup ---

        # Optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

        # LR Scheduler
        num_update_steps_per_epoch = len(train_dataloader) // args.gradient_accumulation_steps
        num_training_steps = num_update_steps_per_epoch * args.num_epochs
        lr_scheduler = get_scheduler(
            args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps, # Adjust warmup steps if needed
            num_training_steps=num_training_steps * args.gradient_accumulation_steps, # Adjust total steps
        )
        print(f"Optimizer and LR Scheduler created.")

        # Start Training
        # Only run eval if components were set up correctly
        eval_freq_actual = args.eval_freq if eval_model_instance and eval_dataloader else args.num_epochs + 10 # Effectively disable eval if setup failed

        train_ddpm(
            model=model, dataloader=train_dataloader, optimizer=optimizer,
            lr_scheduler=lr_scheduler, # Pass scheduler
            device=device, epochs=args.num_epochs,
            noise_scheduler_train=noise_scheduler_train, # Pass train scheduler
            eval_freq=eval_freq_actual, # Use conditional freq
            eval_dataloader=eval_dataloader,
            eval_model_instance=eval_model_instance,
            noise_scheduler_eval=noise_scheduler_eval,
            num_eval_timesteps=args.test_timesteps,
            save_every=args.save_freq, save_dir=args.save_dir,
            output_dir_eval=output_dir_eval,
            gradient_accumulation_steps=args.gradient_accumulation_steps # Pass grad accum steps
         )
        print("--- Training Finished ---")

    # --- Testing Mode ---
    elif args.mode == "test":
        print("--- Starting Testing ---")
        output_dir_test = Path(args.output_dir) / "final_test"
        output_dir_test.mkdir(parents=True, exist_ok=True)
        ckpt_path = Path(args.ckpt)

        if not ckpt_path.exists(): print(f"Error: Checkpoint not found at {ckpt_path}"); sys.exit(1)
        print(f"Loading checkpoint from: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device)
        if 'model_state_dict' in checkpoint: model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint: model.load_state_dict(checkpoint['state_dict'])
        else: print("Error: Could not find model state dictionary in checkpoint."); sys.exit(1)
        model.eval(); print("Model loaded successfully.")

        try:
            eval_model = evaluation_model(device=args.device)
        except Exception as e: print(f"Error initializing evaluation model: {e}"); sys.exit(1)

        test_json_path = Path(args.test_json); new_test_json_path = Path(args.new_test_json)
        try:
            test_label_dataset = LabelDataset(json_path=test_json_path, object_map=object_map)
            new_test_label_dataset = LabelDataset(json_path=new_test_json_path, object_map=object_map)
        except Exception as e: print(f"Error creating test label dataset: {e}"); sys.exit(1)
        test_dataloader = DataLoader(test_label_dataset, batch_size=1, shuffle=False)
        new_test_dataloader = DataLoader(new_test_label_dataset, batch_size=1, shuffle=False)

        print(f"Running inference with {args.test_timesteps} steps...")
        # For final testing, enable image saving in inference
        test_acc = inference(test_dataloader, noise_scheduler_eval, args.test_timesteps,
                             model, eval_model, 'test', output_dir_test, device, save_images=True)
        print(f"Running inference on new test set...")
        new_test_acc = inference(new_test_dataloader, noise_scheduler_eval, args.test_timesteps,
                                 model, eval_model, 'new_test', output_dir_test, device, save_images=True)

        test_mean_acc = np.mean(test_acc) if test_acc else 0.0
        new_test_mean_acc = np.mean(new_test_acc) if new_test_acc else 0.0
        print(f'Test set average accuracy: {test_mean_acc:.4f}')
        print(f'New test set average accuracy: {new_test_mean_acc:.4f}')

        # <<< Log final test accuracies to WandB >>>
        wandb.log({
            "final_test_accuracy": test_mean_acc,
            "final_new_test_accuracy": new_test_mean_acc
        })
        print("--- Testing Finished ---")
    else:
         print(f"Error: Invalid mode '{args.mode}'. Choose 'train' or 'test'.")

    # <<< Finish WandB Run >>>
    if wandb.run is not None:
        wandb.finish()
        print("WandB run finished.")

if __name__ == "__main__":
    main()