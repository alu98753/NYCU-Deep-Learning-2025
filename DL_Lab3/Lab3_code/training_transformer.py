import os
import numpy as np
from tqdm import tqdm
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import utils as vutils
from models import MaskGit as VQGANTransformer
from utils import LoadTrainData
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

#TODO2 step1-4: design the transformer training strategy
class TrainTransformer:
    def __init__(self, args, MaskGit_CONFIGS):
        self.model = VQGANTransformer(MaskGit_CONFIGS["model_param"]).to(device=args.device)
        self.optim,self.scheduler = self.configure_optimizers()
        self.prepare_training()
        self.writer = SummaryWriter("logs1e4_2/")

        # test:
        # img = torch.randn(1, 3, 256, 256).to('cuda')  # 假裝是輸入圖片
        # z_indices, z_q =self.model.encode_to_z(img)
        # print(f"aaaaa:  ",z_indices.shape)  # 應該會是 [1, num_tokens]
        # print(z_q.shape)
                
    @staticmethod
    def prepare_training():
        os.makedirs("transformer_checkpoints", exist_ok=True)

    def train_one_epoch(self,train_loader,epoch,args):
        self.model.train()
        self.optim.zero_grad()        
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}")
        
        total_loss = 0
        
        for step , batch in pbar:
            batch = batch.to(args.device)
            logits, z_indices = self.model(batch)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), z_indices.view(-1))
            loss.backward()
            
            if step% args.accum_grad == 0 or (step+1 == len(train_loader)):
                self.optim.step()
                self.optim.zero_grad()
            total_loss += loss.item()
            pbar.set_description_str(f"epoch: {epoch} / {args.epochs}, iter: {step} / {len(train_loader)}, loss: {total_loss/(step+1)}")
        
        mean_loss = total_loss/len(train_loader)
        self.writer.add_scalar("loss/train", mean_loss ,epoch)        
        return mean_loss

    def eval_one_epoch(self,val_loader,epoch,args):
        self.model.eval() 
        eval_loss = 0
        
        with torch.no_grad(),tqdm(enumerate(val_loader), total=len(val_loader), desc=f"Epoch {epoch}") as pbar:
            for step , batch in pbar:
                batch = batch.to(args.device)
                logits, z_indices = self.model(batch)
                
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), z_indices.reshape(-1))
                eval_loss += loss.item()
                
                pbar.set_description_str(f"epoch: {epoch} / {args.epochs}, iter: {step} / {len(val_loader)}, loss: {eval_loss/(step+1)}")
        
        mean_loss = eval_loss/len(val_loader)
        self.writer.add_scalar("loss/val", mean_loss ,epoch)        
        return mean_loss
    
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.learning_rate,weight_decay=0.01)
        scheduler = None
        return optimizer,scheduler


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MaskGIT")
    #TODO2:check your dataset path is correct 
    parser.add_argument('--train_d_path', type=str, default="./lab3_dataset/train", help='Training Dataset Path') # ./cat_face/train/
    parser.add_argument('--val_d_path', type=str, default="./lab3_dataset/val", help='Validation Dataset Path') # ./cat_face/val/
    parser.add_argument('--checkpoint-path', type=str, default='./checkpoints/last_ckpt.pt', help='Path to checkpoint.')
    parser.add_argument('--device', type=str, default="cuda:0", help='Which device the training is on.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker')
    parser.add_argument('--batch-size', type=int, default=10, help='Batch size for training.')
    parser.add_argument('--partial', type=float, default=1.0, help='Number of epochs to train (default: 50)')    
    parser.add_argument('--accum-grad', type=int, default=10, help='Number for gradient accumulation.')

    #you can modify the hyperparameters 
    parser.add_argument('--epochs', type=int, default=300, help='Number of epochs to train.')
    parser.add_argument('--save-per-epoch', type=int, default=100, help='Save CKPT per ** epochs(defcault: 1)')
    parser.add_argument('--start-from-epoch', type=int, default=0, help='Number of epochs to train.')
    parser.add_argument('--ckpt-interval', type=int, default=0, help='Number of epochs to train.')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate.')
                                                            # v1 : 1e-4  , v2: 2.25e-5
    parser.add_argument('--MaskGitConfig', type=str, default='config/MaskGit.yml', help='Configurations for TransformerVQGAN')

    args = parser.parse_args()

    MaskGit_CONFIGS = yaml.safe_load(open(args.MaskGitConfig, 'r'))
    train_transformer = TrainTransformer(args, MaskGit_CONFIGS)

    train_dataset = LoadTrainData(root= args.train_d_path, partial=args.partial)
    train_loader = DataLoader(train_dataset,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                drop_last=True,
                                pin_memory=True,
                                shuffle=True)
    
    val_dataset = LoadTrainData(root= args.val_d_path, partial=args.partial)
    val_loader =  DataLoader(val_dataset,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                drop_last=True,
                                pin_memory=True,
                                shuffle=False)
    
#TODO2 step1-5:    
    best_eval_loss = np.inf 
    # 載入模型 + optimizer + best_eval_loss
    # if args.start_from_epoch > 0 and os.path.exists(args.checkpoint_path):
    #     print(f"Loading checkpoint from: {args.checkpoint_path}")
    #     ckpt = torch.load(args.checkpoint_path, map_location=args.device)
    #     train_transformer.model.load_state_dict(ckpt['model_state_dict'])
    #     train_transformer.optim.load_state_dict(ckpt['optimizer_state_dict'])
    #     best_eval_loss = ckpt['best_eval_loss']


    for epoch in range(args.start_from_epoch, args.epochs+1):
        train_loss = train_transformer.train_one_epoch(train_loader,epoch,args)
        eval_loss = train_transformer.eval_one_epoch(val_loader,epoch,args)
        
        # 存模型
        if epoch % args.save_per_epoch == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': train_transformer.model.state_dict(),
                'optimizer_state_dict': train_transformer.optim.state_dict(),
                'best_eval_loss': best_eval_loss,
            }, f"transformer_checkpoints_1e4/full_ckpt_epoch_{epoch}.pt")
        
        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            torch.save(train_transformer.model.transformer.state_dict(), f"transformer_checkpoints_1e4/best_val.pth")