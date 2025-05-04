# Spring 2025, 535507 Deep Learning
# Lab6: Lab6 - Generative Models / DDPM
# Instructor: Ping-Chun Hsieh、Wei Hung and Alison Wen

# Contributors: Huang Tzu Cheng

import gc
import torch
from torch.amp import autocast, GradScaler
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import numpy as np
import random
import os
import json
from PIL import Image
from tqdm.auto import tqdm 
import argparse
import time

# --- Custom Modules ---
import wandb
from diffusers import UNet2DConditionModel, DDPMScheduler, DDPMPipeline
from diffusers.optimization import get_scheduler
import torchvision.transforms as T
from torchvision.utils import make_grid, save_image
from pathlib import Path
from file.evaluator import evaluation_model
torch.cuda.ipc_collect()

def parse_args():
    parser = argparse.ArgumentParser(description="DDPM Training for iCLEVR Image Generation",
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # --- Paths ---
    parser.add_argument("--data_dir", type=str, default="file", help="Directory containing dataset files (iclevr.zip, *.json)")
    parser.add_argument("--image_dir", type=str, default="data/iclevr", help="Directory containing extracted images")
    parser.add_argument("--output_dir", type=str, default="res", help="Directory to save generated images and models")
    
    # --- Dataset & Loader ---
    parser.add_argument("--image_size", type=int, default=64, help="Image resolution")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers for DataLoader")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")

    # --- Model(UNet) ---
    parser.add_argument("--unet_channels", type=int, default=3, help="Number of input image channels")
    parser.add_argument("--unet_out_channels", type=int, default=3, help="Number of output image channels")
    parser.add_argument("--unet_block_out_channels", type=tuple, default=(128, 256, 512), help="UNet block output channels")
    parser.add_argument("--unet_cross_attn_dim", type=int, default=128, help="Cross attention dimension (should match condition embedding dim)")

    # --- Diffusion  ---
    parser.add_argument("--num_train_timesteps", type=int, default=1500, help="Number of diffusion timesteps")
    parser.add_argument("--beta_schedule", type=str, default="linear", choices=["linear", "cosine"], help="Beta schedule type")

    # --- Conditioning ---
    parser.add_argument("--num_classes", type=int, default=24, help="Number of object classes (from object.json)")
    parser.add_argument("--cond_embed_dim", type=int, default=128, help="Dimension of the condition embedding")

    # --- Training ---
    parser.add_argument("--num_epochs", type=int, default=550, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--lr_scheduler_type", type=str, default="linear", help="Learning rate scheduler type")
    parser.add_argument("--lr_warmup_steps", type=int, default=500, help="Number of warmup steps for LR scheduler")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="AdamW beta1")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="AdamW beta2")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="AdamW weight decay")
    parser.add_argument("--adam_epsilon", type=float, default=1e-8, help="AdamW epsilon")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm for clipping")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    # --- Evaluation & Sampling ---
    parser.add_argument("--eval_batch_size", type=int, default=64, help="Batch size for evaluation/sampling")
    parser.add_argument("--eval_epochs", type=int, default=1, help="Frequency (in epochs) to run evaluation") 
    parser.add_argument("--save_model_threshold", type=float, default=0, help="Mean accuracy threshold to save model")
    parser.add_argument("--save_image_threshold", type=float, default=0, help="Individual accuracy threshold to save images per test set")

    # --- WandB ---
    parser.add_argument("--wandb_project", type=str, default="DLP-Lab6-DDPM", help="WandB project name")
    parser.add_argument("--wandb_run_name", type=str, default=f"ddpm_light_{time.strftime('%Y%m%d_%H%M%S')}", help="WandB run name")
    parser.add_argument("--log_images_steps", type=int, default=500, help="Frequency (in steps) to log sample images to WandB during training")
    args = parser.parse_args()
    
    return args

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class iCLEVRDataset(Dataset):
    def __init__(self, image_dir, json_path, object_map, transform=None):
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.object_map = object_map
        self.num_classes = len(object_map)

        print(f"Loading data from: {json_path}")
        with open(json_path, 'r') as f:
            self.data = json.load(f)

        if isinstance(self.data, dict):
            self.image_files = list(self.data.keys())
            self.labels = [self.data[fname] for fname in self.image_files]
        elif isinstance(self.data, list):
            print("Assuming training images are named 0.png, 1.png, ... according to train.json order.")
            self.image_files = [f"{i}.png" for i in range(len(self.data))]
            self.labels = self.data
        else:
            raise ValueError(f"Unsupported JSON format in {json_path}")

        # Optional: Filter out missing files during init
        valid_indices = [i for i, fname in enumerate(self.image_files) if (self.image_dir / fname).exists()]
        if len(valid_indices) < len(self.image_files):
             print(f"Warning: Filtered out {len(self.image_files) - len(valid_indices)} missing image files.")
             self.image_files = [self.image_files[i] for i in valid_indices]
             self.labels = [self.labels[i] for i in valid_indices]

        print(f"Found {len(self.image_files)} valid samples.")


    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_dir / self.image_files[idx]
        image = Image.open(img_name).convert('RGB')
        labels = self.labels[idx]
        label_vector = torch.zeros(self.num_classes, dtype=torch.float32)
        for obj_name in labels:
            if obj_name in self.object_map:
                label_vector[self.object_map[obj_name]] = 1.0
            else:
                 print(f"Warning: Unknown object '{obj_name}' in labels for image {self.image_files[idx]}.")

        if self.transform:
            image = self.transform(image)

        return image, label_vector

class ConditionEmbedder(nn.Module):
    def __init__(self, num_classes, embed_dim, hidden_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(num_classes, hidden_dim),
            nn.SiLU(), # Or ReLU, GELU
            nn.Linear(hidden_dim, embed_dim)
        )
        
    def forward(self, class_labels):
        embeddings = self.mlp(class_labels)
        return embeddings


def evaluate_model(evaluator, pipeline, scheduler, condition_embedder, test_labels_dict, device, args, epoch, global_step):
    if evaluator is None:
        print("Evaluator not loaded, skipping evaluation.")
        return {'mean_accuracy': 0.0}

    print(f"\n--- Starting Evaluation Epoch {epoch} ---")
    pipeline.to(device)
    pipeline.unet.eval()
    condition_embedder.to(device)
    condition_embedder.eval()

    results = {}
    accuracies = []
    wandb_logs = {}
    img_denorm = T.Normalize([-1.0, -1.0, -1.0], [2.0, 2.0, 2.0])

    for test_set_name, labels_list in test_labels_dict.items():
        print(f"Evaluating on {test_set_name}...")
        num_samples = len(labels_list)
        batch_accuracies = []
        generated_images_for_grid = []
        table_data = []

        original_label_strings_list = list(labels_list.values()) if isinstance(labels_list, dict) else labels_list

        for i in tqdm(range(0, num_samples, args.eval_batch_size), desc=f"Sampling {test_set_name}"):
            batch_labels_str_original_order = original_label_strings_list[i : i + args.eval_batch_size]
            current_batch_size = len(batch_labels_str_original_order)

            batch_label_vectors = torch.zeros(current_batch_size, args.num_classes, device=device, dtype=torch.float32)
            batch_gt_one_hot = torch.zeros(current_batch_size, args.num_classes, device=device, dtype=torch.float32)

            for j, labels_str in enumerate(batch_labels_str_original_order):
                for obj_name in labels_str:
                    if obj_name in args.object_map:
                        batch_label_vectors[j, args.object_map[obj_name]] = 1.0
                        batch_gt_one_hot[j, args.object_map[obj_name]] = 1.0

            with torch.no_grad():
                with autocast(device_type='cuda'): 
                    condition_embeddings = condition_embedder(batch_label_vectors).unsqueeze(1)

                    gen = torch.Generator(device=device).manual_seed(args.seed + i + epoch)
                    images = torch.randn(
                        (current_batch_size, args.unet_channels, args.image_size, args.image_size),
                        generator=gen, device=device,
                    )
                    scheduler.set_timesteps(args.num_train_timesteps)
                    for t in scheduler.timesteps:
                        model_input = scheduler.scale_model_input(images, t)
                        noise_pred = pipeline.unet(
                            sample=model_input,
                            timestep=t,
                            encoder_hidden_states=condition_embeddings
                        ).sample
                        images = scheduler.step(noise_pred, t, images).prev_sample

                    images_cpu = images.detach().cpu()
                    gt_cpu = batch_gt_one_hot.detach().cpu()

                    batch_acc = evaluator.eval(
                        images_cpu.to(evaluator.device),
                        gt_cpu.to(evaluator.device)
                    )
                    batch_accuracies.append(batch_acc)



                    images_denorm = [img_denorm(img) for img in images_cpu]
                    generated_images_for_grid.extend(images_denorm)
                    for k in range(current_batch_size):
                        table_data.append([
                            i + k,
                            ", ".join(batch_labels_str_original_order[k]),
                            wandb.Image(images_denorm[k])
                        ])


            del images, images_cpu, gt_cpu
            if 'images_denorm' in locals():
                del images_denorm
            torch.cuda.empty_cache()

        accuracy = np.mean(batch_accuracies) if batch_accuracies else 0.0
        results[f"{test_set_name}_accuracy"] = accuracy
        accuracies.append(accuracy)
        print(f"{test_set_name} Accuracy: {accuracy:.4f}")


        if accuracy > args.save_image_threshold:
            if generated_images_for_grid:
                grid_tensor = make_grid(generated_images_for_grid, nrow=8, padding=2, normalize=False)
                wandb_logs[f"Generated Images/{test_set_name}_Grid_Epoch{epoch}"] = wandb.Image(grid_tensor)
                grid_save_path = Path(args.output_dir) / f"{test_set_name}_grid_epoch{epoch}.png"
                save_image(grid_tensor, grid_save_path)

            print(f"Accuracy ({accuracy:.4f}) > Threshold ({args.save_image_threshold}). Saving images...")
            save_dir = Path(args.output_dir) / "images" / f"epoch_{epoch}" / test_set_name
            save_dir.mkdir(parents=True, exist_ok=True)
            for idx, img_tensor in enumerate(generated_images_for_grid):
                save_image(img_tensor, save_dir / f"{idx}.png")

        if table_data:
            comparison_table = wandb.Table(columns=["Index", "Target Labels", "Generated Image"], data=table_data)
            wandb_logs[f"Comparison Table/{test_set_name}_Epoch{epoch}"] = comparison_table

    mean_accuracy = np.mean(accuracies) if accuracies else 0.0
    results["mean_accuracy"] = mean_accuracy
    print(f"Epoch {epoch} Mean Accuracy: {mean_accuracy:.4f}")

    # Denoising visualization 
    if mean_accuracy > args.save_model_threshold:
        denoise_labels_str = ["red sphere", "cyan cylinder", "cyan cube"]
        denoise_label_vector = torch.zeros(1, args.num_classes, device=device, dtype=torch.float32)
        for obj_name in denoise_labels_str:
            if obj_name in args.object_map:
                denoise_label_vector[0, args.object_map[obj_name]] = 1.0

        denoise_imgs = []
        with torch.no_grad():
            cond_embed = condition_embedder(denoise_label_vector).unsqueeze(1)
            gen = torch.Generator(device=device).manual_seed(args.seed + epoch)
            image = torch.randn((1, args.unet_channels, args.image_size, args.image_size),
                                generator=gen, device=device)
            scheduler.set_timesteps(args.num_train_timesteps)
            denoise_imgs.append(img_denorm(image[0].cpu()))
            for i, t in enumerate(tqdm(scheduler.timesteps, desc="Denoising Viz", leave=False)):
                model_input = scheduler.scale_model_input(image, t)
                noise_pred = pipeline.unet(model_input, t, encoder_hidden_states=cond_embed).sample
                image = scheduler.step(noise_pred, t, image).prev_sample
                if i % (args.num_train_timesteps // 10) == 0 or i == args.num_train_timesteps - 1:
                    denoise_imgs.append(img_denorm(image[0].cpu()))

        if denoise_imgs:
            denoise_grid = make_grid(denoise_imgs, nrow=len(denoise_imgs), padding=2, normalize=False)
            wandb_logs[f"Generated Images/Denoising_Process_Epoch{epoch}"] = wandb.Image(denoise_grid)
            denoise_save_path = Path(args.output_dir) / f"denoising_process_epoch{epoch}.png"
            save_image(denoise_grid, denoise_save_path)

    wandb.log({**results, **wandb_logs}, step=global_step)

    pipeline.unet.train()
    condition_embedder.train()
    print(f"--- Finished Evaluation Epoch {epoch} ---")
    return results

def main():
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True" 
    torch.set_num_threads(4)
    
    args = parse_args()
    set_seed(args.seed)
    eval_epochs =  args.eval_epochs 
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    #  Output Dirs 
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    #  Load Object Map 
    object_json_path = Path(args.data_dir) / "objects.json"
    with open(object_json_path, 'r') as f:
        args.object_map = json.load(f)
    args.object_map_inv = {v: k for k, v in args.object_map.items()}
    args.num_classes = len(args.object_map)
    print(f"Loaded {args.num_classes} classes from objects.json")

    # --- Datasets and DataLoaders ---
    train_transform = T.Compose([
        T.Resize((args.image_size, args.image_size)),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    # Ensure image directory exists (logic kept from previous version)
    image_dir_path = Path(args.image_dir)
    if not image_dir_path.exists() or not any(image_dir_path.iterdir()):
         zip_path = Path(args.data_dir) / "iclevr.zip"
         if zip_path.exists():
             print(f"Image directory {args.image_dir} seems empty or missing. Attempting to extract from {zip_path}...")
             import zipfile
             with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                 zip_ref.extractall(Path(args.data_dir))
             print(f"Extraction complete. Expecting images in {args.image_dir}.")
             if not image_dir_path.exists() or not any(image_dir_path.iterdir()):
                raise FileNotFoundError(f"Image directory {args.image_dir} still empty after extraction attempt.")
         else:
            raise FileNotFoundError(f"Image directory {args.image_dir} not found or empty, and iclevr.zip not found at {zip_path}.")


    train_json_path = Path(args.data_dir) / "train.json"
    if not train_json_path.exists():
        raise FileNotFoundError(f"train.json not found at {train_json_path}")

    train_dataset = iCLEVRDataset(image_dir=args.image_dir,
                                json_path=train_json_path,
                                object_map=args.object_map,
                                transform=train_transform)
    train_dataloader = DataLoader(train_dataset,
                                batch_size=args.batch_size,
                                shuffle=True,
                                num_workers=args.num_workers,
                                pin_memory=True if device == torch.device("cuda") else False) # Pin memory only for CUDA

    # --- Load Test Labels ---
    test_labels_dict = {}
    test_json_path = Path(args.data_dir) / "test.json"
    new_test_json_path = Path(args.data_dir) / "new_test.json"
    if test_json_path.exists():
        with open(test_json_path, 'r') as f:
            test_data_raw = json.load(f)
            test_labels_dict["test"] = test_data_raw
    else:
        print(f"Warning: test.json not found at {test_json_path}")

    if new_test_json_path.exists():
        with open(new_test_json_path, 'r') as f:
            new_test_data_raw = json.load(f)
            test_labels_dict["new_test"] = new_test_data_raw
    else:
        print(f"Warning: new_test.json not found at {new_test_json_path}")


    block_out_channels = (64, 128, 256)  # 或 (64, 64, 128) 測試看看效果 # 每層輸出的通道數 (層數與 block_types 對應)
    down_block_types = ("DownBlock2D","CrossAttnDownBlock2D","CrossAttnDownBlock2D")
    up_block_types = ("CrossAttnUpBlock2D","CrossAttnUpBlock2D","UpBlock2D")

    # 確保 cross_attention_dim 與 ConditionEmbedder 的輸出維度一致
    unet_cross_attn_dim = args.cond_embed_dim # 128

    unet = UNet2DConditionModel(
        sample_size=args.image_size,         # 64
        in_channels=args.unet_channels,      # in_channels 3 for RGB)
        out_channels=args.unet_out_channels, 
        block_out_channels=block_out_channels, # 指定每層通道數
        down_block_types=down_block_types,     
        up_block_types=up_block_types,         
        mid_block_type="UNetMidBlock2DCrossAttn", # 中間block type
        cross_attention_dim=unet_cross_attn_dim, 
        layers_per_block=1,                   # 每個 ResNet包含的層數(預設 2)
    ).to(device) 

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=args.num_train_timesteps,
        beta_schedule=args.beta_schedule,
    )

    condition_embedder = ConditionEmbedder(
        num_classes=args.num_classes,
        embed_dim=args.cond_embed_dim # 確保有把cond_embed_dim這個值 (128) 傳遞給 UNet 的 cross_attention_dim
    ).to(device) 

    params_to_optimize = list(unet.parameters()) + list(condition_embedder.parameters())
    optimizer = torch.optim.AdamW(params_to_optimize,lr=args.learning_rate,betas=(args.adam_beta1, args.adam_beta2),weight_decay=args.adam_weight_decay,eps=args.adam_epsilon,)

    num_update_steps_per_epoch = len(train_dataloader) // args.gradient_accumulation_steps
    num_training_steps = num_update_steps_per_epoch * args.num_epochs

    lr_scheduler = get_scheduler(args.lr_scheduler_type,optimizer=optimizer,num_warmup_steps=args.lr_warmup_steps, num_training_steps=num_training_steps,)

    #  Load Evaluator 
    evaluator = evaluation_model()

    wandb.init(project=args.wandb_project,name=args.wandb_run_name,config=vars(args),save_code=True,)

    #  Training Loop 
    global_step = 0
    print("***** Running training *****")
    print(f"  Num examples = {len(train_dataset)}")
    print(f"  Num Epochs = {args.num_epochs}")
    #  Create pipeline for sampling 
    pipeline = DDPMPipeline(unet=unet, scheduler=noise_scheduler)
    scaler = GradScaler() 
        
    for epoch in range(args.num_epochs):
        unet.train()
        condition_embedder.train()
        progress_bar = tqdm(total=len(train_dataloader), desc=f"Epoch {epoch+1}/{args.num_epochs}")
        optimizer.zero_grad() 

        for step, batch in enumerate(train_dataloader):
            images, label_vectors = batch
            images = images.to(device)
            label_vectors = label_vectors.to(device)

            # Sample noise & timesteps
            noise = torch.randn_like(images)
            bsz = images.shape[0]
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=images.device).long()

            # Add noise
            noisy_images = noise_scheduler.add_noise(images, noise, timesteps)

            # Get condition embeddings
            condition_embeddings = condition_embedder(label_vectors).unsqueeze(1)
            
            with autocast(device_type='cuda'):
                # Predict noise
                model_pred = unet(sample=noisy_images,timestep=timesteps,encoder_hidden_states=condition_embeddings).sample
                loss = F.mse_loss(model_pred.float(), noise.float())

            # --- Manual Gradient Accumulation ---
            loss = loss / args.gradient_accumulation_steps # Scale loss
            scaler.scale(loss).backward()

            if (step + 1) % args.gradient_accumulation_steps == 0 or (step + 1) == len(train_dataloader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                     list(unet.parameters()) + list(condition_embedder.parameters()),
                     args.max_grad_norm
                 )

                # Optimizer Step
                scaler.step(optimizer) 
                scaler.update()
                lr_scheduler.step()
                optimizer.zero_grad() # Zero gradients after optimization step

                global_step += 1
                current_lr = lr_scheduler.get_last_lr()[0]
                wandb.log({"train_loss": loss.item() * args.gradient_accumulation_steps, "lr": current_lr}, step=global_step)

            progress_bar.update(1)
            if (step + 1) % args.gradient_accumulation_steps == 0:
                progress_bar.set_postfix({"Loss": f"{loss.item() * args.gradient_accumulation_steps:.4f}", "LR": f"{current_lr:.1e}"})
            
        progress_bar.close()

        if (epoch + 1) % eval_epochs == 0 or epoch == args.num_epochs - 1:
            eval_results = evaluate_model(evaluator=evaluator,
                pipeline=pipeline, 
                scheduler=noise_scheduler,
                condition_embedder=condition_embedder, # Pass model directly
                test_labels_dict=test_labels_dict,
                device=device,
                args=args,
                epoch=epoch + 1,
                global_step=global_step
            )

            # Save Model 
            mean_accuracy = eval_results.get('mean_accuracy', 0.0)
            if mean_accuracy > args.save_model_threshold:
                eval_epochs = 2
                print(f"Mean Accuracy ({mean_accuracy:.4f}) > Threshold ({args.save_model_threshold}). Saving model for Epoch {epoch + 1}...")
                save_path = output_dir / f"model_epoch_{epoch+1}_acc_{mean_accuracy:.4f}"
                save_path.mkdir(exist_ok=True)
                torch.save(unet.state_dict(), save_path / "unet.pth")
                torch.save(condition_embedder.state_dict(), save_path / "condition_embedder.pth")
                print(f"Model saved to {save_path}")
            else:
                print(f"Mean Accuracy ({mean_accuracy:.4f}) <= Threshold ({args.save_model_threshold}). Skipping model saving for Epoch {epoch + 1}.")
            
            gc.collect()
            torch.cuda.empty_cache()

    print("Training Finished!")
    wandb.finish()

if __name__ == "__main__":
    main()