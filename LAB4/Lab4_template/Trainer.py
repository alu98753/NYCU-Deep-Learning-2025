import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader

from modules import Generator, Gaussian_Predictor, Decoder_Fusion, Label_Encoder, RGB_Encoder

from dataloader import Dataset_Dance
from torchvision.utils import save_image
import random
import torch.optim as optim
from torch import stack

from tqdm import tqdm
import imageio

import matplotlib.pyplot as plt
from math import log10
from torch.utils import tensorboard

def Generate_PSNR(imgs1, imgs2, data_range=1.):
    """PSNR for torch tensor"""
    mse = nn.functional.mse_loss(imgs1, imgs2) # wrong computation for batch size > 1
    psnr = 20 * log10(data_range) - 10 * torch.log10(mse)
    return psnr


def kl_criterion(mu, logvar, batch_size):
  KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
  KLD /= batch_size  
  return KLD


class kl_annealing():
    def __init__(self, args, current_epoch=0):
        # TODO
        self.kl_anneal_type = args.kl_anneal_type
        assert self.kl_anneal_type in ["Cyclical", "Monotonic", "Without"], f"Invalid kl_anneal_type: {self.kl_anneal_type}"
        self.current_epoch = current_epoch +1 
        self.total_epochs = args.num_epoch
        self.start = 0.0
        self.stop = 1.0
        
        if self.kl_anneal_type == "Cyclical":
            self.beta = self.frange_cycle_linear(self.total_epochs+2, self.start, self.stop, n_cycle=args.kl_anneal_cycle,  ratio=args.kl_anneal_ratio)
        elif self.kl_anneal_type == "Monotonic":
            self.beta = self.frange_cycle_linear(self.total_epochs+2, self.start, self.stop, n_cycle=1,  ratio=args.kl_anneal_ratio)
        elif self.kl_anneal_type == "Without":
            self.beta = np.ones(args.num_epoch + 1) 
                            
    def update(self):
        # TODO
        self.current_epoch += 1
                           
    def get_beta(self):
        # TODO
        return self.beta[self.current_epoch]

    # TODO
    # ref from: https://github.com/haofuml/cyclical_annealing
    def frange_cycle_linear(self, n_iter, start=0.0, stop=1.0,  n_cycle=4, ratio=0.5):
        L = np.ones(n_iter) * stop
        period = n_iter/n_cycle
        step = (stop-start)/(period*ratio) # linear schedule

        for c in range(n_cycle):
            v, i = start, 0
            while v <= stop and (int(i+c*period) < n_iter):
                L[int(i+c*period)] = v
                v += step
                i += 1
        return L      

class VAE_Model(nn.Module):
    def __init__(self, args):
        super(VAE_Model, self).__init__()
        self.args = args
        
        # Modules to transform image from RGB-domain to feature-domain
        self.frame_transformation = RGB_Encoder(3, args.F_dim)
        self.label_transformation = Label_Encoder(3, args.L_dim)
        
        # Conduct Posterior prediction in Encoder
        self.Gaussian_Predictor   = Gaussian_Predictor(args.F_dim + args.L_dim, args.N_dim)
        self.Decoder_Fusion       = Decoder_Fusion(args.F_dim + args.L_dim + args.N_dim, args.D_out_dim)
        
        # Generative model
        self.Generator            = Generator(input_nc=args.D_out_dim, output_nc=3)
        
        self.optim      = optim.Adam(self.parameters(), lr=self.args.lr)
        self.scheduler  = optim.lr_scheduler.MultiStepLR(self.optim, milestones=[2, 5], gamma=0.1)
        self.kl_annealing = kl_annealing(args, current_epoch=0)
        self.mse_criterion = nn.MSELoss()
        self.current_epoch = 0
        
        # Teacher forcing arguments
        self.tfr = args.tfr
        self.tfr_d_step = args.tfr_d_step
        self.tfr_sde = args.tfr_sde
        
        self.train_vi_len = args.train_vi_len
        self.val_vi_len   = args.val_vi_len
        self.batch_size = args.batch_size
        
        # writer 
        if not self.args.test:
            self.writer = tensorboard.SummaryWriter(f"logs/kl-type_{args.kl_anneal_type}_tfr_{args.tfr}_d_step_{args.tfr_d_step}")
   
    def forward(self, img, label):
        pass
    
    def training_stage(self):
        save_dir = os.path.join(self.args.save_root,f"kl-type_{self.args.kl_anneal_type}_tfr_{self.args.tfr}_d_step_{self.args.tfr_d_step}")
        os.makedirs(save_dir, exist_ok=True)

        for i in range(self.args.num_epoch):
            train_loader = self.train_dataloader()
            adapt_TeacherForcing = True if random.random() < self.tfr else False
            losses = []
            for (img, label) in (pbar := tqdm(train_loader, ncols=120)):
                img = img.to(self.args.device)
                label = label.to(self.args.device)
                loss = self.training_one_step(img, label, adapt_TeacherForcing)
                losses.append(loss.detach().cpu())
                beta = self.kl_annealing.get_beta()
                if adapt_TeacherForcing:
                    self.tqdm_bar('train [TeacherForcing: ON, {:.1f}], beta: {}'.format(self.tfr, beta), pbar, loss.detach().cpu(), lr=self.scheduler.get_last_lr()[0])
                else:
                    self.tqdm_bar('train [TeacherForcing: OFF, {:.1f}], beta: {}'.format(self.tfr, beta), pbar, loss.detach().cpu(), lr=self.scheduler.get_last_lr()[0])
            
            if self.current_epoch % self.args.per_save == 0 and self.current_epoch > 39:
                self.save(os.path.join(save_dir, f"epoch={self.current_epoch}.ckpt"))
            self.writer.add_scalar('Loss/train', np.mean(losses), self.current_epoch)
            self.writer.add_scalar('beta', beta, self.current_epoch)
            self.writer.add_scalar('tfr', self.tfr, self.current_epoch)    
            
            self.eval()
            self.current_epoch += 1
            self.scheduler.step()
            self.teacher_forcing_ratio_update()
            self.kl_annealing.update()
        self.save(os.path.join(save_dir, f"epoch={self.current_epoch}.ckpt"))

            
    @torch.no_grad()
    def eval(self):
        val_loader = self.val_dataloader()
        losses = []
        PSNRs = []
        for (img, label) in (pbar := tqdm(val_loader, ncols=120)):
            img = img.to(self.args.device)
            label = label.to(self.args.device)
            loss, PSNR ,psnr_list= self.val_one_step(img, label)
            self.tqdm_bar('val', pbar, loss.detach().cpu(), lr=self.scheduler.get_last_lr()[0])
            losses.append(loss.detach().cpu())
            PSNRs.append(PSNR)
            
        if not self.args.test:
            self.writer.add_scalar('Loss/val', np.mean(losses), self.current_epoch)
            self.writer.add_scalar('PSNR/val', np.mean(PSNRs), self.current_epoch)
        else:
            self.plot_psnr_line_chart(psnr_list)

    def plot_psnr_line_chart(self, psnr_values):
        plt.figure(figsize=(8, 6))
        plt.plot(range(len(psnr_values)), psnr_values, linewidth=1.5)
        plt.title(f'PSNR per Frame (Epoch {self.current_epoch})')
        plt.xlabel('Frame Index')
        plt.ylabel('PSNR')
        plt.grid(True)

        plt.legend([f"PSNR: {np.mean(psnr_values):.2f}"],loc='upper right')
        save_path = os.path.join(self.args.save_root, f"PSNR_{self.tfr}_{self.tfr_d_step}.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Saved PSNR plot at: {save_path}")
    
    def training_one_step(self, img_batch, label_batch, adapt_TeacherForcing):
        # img_batch: (B, T, C, H, W)
        B, T, C, H, W = img_batch.shape
        beta = self.kl_annealing.get_beta()

        ''' v1 : batah level '''
        # total_rec_loss = 0.0
        # total_kl_loss = 0.0

        # # 初始 frame：取每筆樣本的第 0 幀，形狀為 (B, C, H, W)
        # current_frame = img_batch[:, 0]  # B x C x H x W

        # for t in range(1, T):
        #     # 對齊當前time step的 label 與真實 frame
        #     label_t = label_batch[:, t]       # B x C x H x W
        #     true_frame_t = img_batch[:, t]    # B x C x H x W

        #     # 編碼
        #     frame_feature = self.frame_transformation(true_frame_t)   # B x F
        #     label_feature = self.label_transformation(label_t)        # B x F

        #     # 變分採樣
        #     z, mu, logvar = self.Gaussian_Predictor(frame_feature, label_feature)  # B x F

        #     # 依據 Teacher Forcing融合前一幀
        #     if adapt_TeacherForcing:
        #         prev_frame = img_batch[:, t - 1]
        #     else:
        #         prev_frame = current_frame

        #     prev_feature = self.frame_transformation(prev_frame).detach()  # B x F

        #     # 解碼下一幀
        #     fused_feature = self.Decoder_Fusion(prev_feature, label_feature, z)  # B x F
        #     pred_frame = self.Generator(fused_feature)  # B x C x H x W

        #     rec_loss = self.mse_criterion(pred_frame, true_frame_t)
        ##     kl_loss = kl_criterion(mu, logvar, B) # 感覺是/1 因為
        #     kl_loss = kl_criterion(mu, logvar, 1) 

        #     total_rec_loss += rec_loss
        #     total_kl_loss += kl_loss

        #     # 更新 current_frame
        #     current_frame = pred_frame

        # # 平均損失
        # total_loss = total_rec_loss / (T - 1) + beta * total_kl_loss / (T - 1) # 感覺多/ 了t-1
        # self.optim.zero_grad()
        # total_loss.backward()
        # self.optimizer_step()
        
        ''' v2 : frame level '''
        total_loss = 0.0
        
        for i in range(B):
            img = img_batch[i]  # (T, C, H, W)
            label = label_batch[i] # (T, C, H, W)
            
            rec_loss = 0
            kld = 0
            
            current_frame = img[0].unsqueeze(0)  # (1, C, H, W)
            # print(f" current_frame.shape: {current_frame.shape}")

    
            for j in range(1, T):
                frame_feature = self.frame_transformation(img[j].unsqueeze(0))    # (1, feature, H, W)
                label_feature = self.label_transformation(label[j].unsqueeze(0))  # (1, feature, H, W)
                # print(f" frame_feature.shape: {frame_feature.shape}")

                if adapt_TeacherForcing:
                    pred_frame = img[j-1].unsqueeze(0)  
                else:
                    pred_frame = current_frame
                
                # VAE sampling
                z, mu, logvar = self.Gaussian_Predictor(frame_feature, label_feature) 
                kld += kl_criterion(mu, logvar, B)
                
                pred_frame = self.frame_transformation(pred_frame).detach()  #(1, feature, H, W)
                # print(f" pred_frame.shape: {pred_frame.shape}")

                # 融合 + 預測
                fused_feature = self.Decoder_Fusion(pred_frame, label_feature, z)  # (1, C, H, W)
                current_frame = self.Generator(fused_feature)
                # print(f"fused current_frame.shape: {current_frame.shape}")

                # MSE Loss + PSNR（逐 sample）
                rec_loss += self.mse_criterion(current_frame, img[j].unsqueeze(0))
                
            loss = rec_loss + beta * kld
            
            total_loss += loss
            self.optim.zero_grad()
            loss.backward()
            self.optimizer_step()
                
        total_loss /= B

        return total_loss

    
    def val_one_step(self, img_batch, label_batch):
        # img_batch: (B, T, C, H, W)
        B, T, C, H, W = img_batch.shape
        beta = self.kl_annealing.get_beta()
        psnr_list = []

        '''v1 : batah level '''
        # total_rec_loss = 0.0
        # total_kl_loss = 0.0
        # current_frame = img_batch[:, 0]  # (B, C, H, W)

        # for t in range(1, T):
        #     # 對應第 t 幀的 label 和 ground truth frame
        #     label_t = label_batch[:, t]       # (B, C, H, W)
        #     target_frame = img_batch[:, t]    # (B, C, H, W)
        #     # 編碼
        #     frame_feature = self.frame_transformation(target_frame)    # (B, F)
        #     label_feature = self.label_transformation(label_t)         # (B, F)
        #     # VAE samplin
        #     z, mu, logvar = self.Gaussian_Predictor(frame_feature, label_feature)  # (B, F)
        #     kl_loss = kl_criterion(mu, logvar, B)
        #     # 取得前一幀（用上一幀預測）
        #     prev_feature = self.frame_transformation(current_frame).detach()
        #     # 融合 + 預測
        #     fused_feature = self.Decoder_Fusion(prev_feature, label_feature, z)
        #     pred_frame = self.Generator(fused_feature)  # (B, C, H, W)
        #     # MSE Loss + PSNR（逐 sample）
        #     rec_loss = self.mse_criterion(pred_frame, target_frame)
        #     total_rec_loss += rec_loss
        #     total_kl_loss += kl_loss

        #     # 計算 PSNR
        #     psnr_batch = Generate_PSNR(pred_frame, target_frame)  # shape: (B,)
        #     psnr_list.append(psnr_batch.detach().cpu().item())

        #     current_frame = pred_frame  # 更新輸入幀為預測結果

        # total_loss = total_rec_loss / (T - 1) + beta * total_kl_loss / (T - 1)
        
        ''' v2 : frame level '''
        total_loss = 0.0
        for i in range(B):
            img = img_batch[i]
            label = label_batch[i]
            rec_loss = 0
            kld = 0
            current_frame = img[0].unsqueeze(0)
            
            for j in range(1, T):
                frame_feature = self.frame_transformation(img[j].unsqueeze(0))
                label_feature = self.label_transformation(label[j].unsqueeze(0))
                # VAE sampling
                z , mu, logvar = self.Gaussian_Predictor(frame_feature, label_feature)
                kld += kl_criterion(mu, logvar, B)
                
                pred_frame = current_frame.detach()
                pred_frame = self.frame_transformation(pred_frame).detach()
                # 融合 + 預測
                fused_feature = self.Decoder_Fusion(pred_frame, label_feature, z)
                current_frame = self.Generator(fused_feature)
                # MSE Loss + PSNR（逐 sample）
                rec_loss += self.mse_criterion(current_frame, img[j].unsqueeze(0))
                psnr_list.append(Generate_PSNR(current_frame, img[j].unsqueeze(0)).detach().cpu().item())
                
                # print(f"psnr_batch: {psnr_batch}")
            loss = rec_loss + beta * kld
            total_loss += loss
        
        total_loss /= B
                   
        return total_loss, np.mean(psnr_list),psnr_list

               
    def make_gif(self, images_list, img_name):
        new_list = []
        for img in images_list:
            new_list.append(transforms.ToPILImage()(img))
            
        new_list[0].save(img_name, format="GIF", append_images=new_list,
                    save_all=True, duration=40, loop=0)
    
    def train_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize((self.args.frame_H, self.args.frame_W)),
            transforms.ToTensor()
        ])

        dataset = Dataset_Dance(root=self.args.DR, transform=transform, mode='train', video_len=self.train_vi_len, \
                                                partial=args.fast_partial if self.args.fast_train else args.partial)
        if self.current_epoch > self.args.fast_train_epoch:
            self.args.fast_train = False
            
        train_loader = DataLoader(dataset,
                                  batch_size=self.batch_size,
                                  num_workers=self.args.num_workers,
                                  drop_last=True,
                                  shuffle=False)  
        return train_loader
    
    def val_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize((self.args.frame_H, self.args.frame_W)),
            transforms.ToTensor()
        ])
        dataset = Dataset_Dance(root=self.args.DR, transform=transform, mode='val', video_len=self.val_vi_len, partial=1.0)  
        val_loader = DataLoader(dataset,
                                  batch_size=1,
                                  num_workers=self.args.num_workers,
                                  drop_last=True,
                                  shuffle=False)  
        return val_loader
    
    def teacher_forcing_ratio_update(self):
        # TODO
        # 當 epoch 大於或等於 tfr_sde 時，每個 epoch 下降 tfr_d_step，最低不低於 0
        if self.current_epoch >= self.tfr_sde  and self.current_epoch % self.tfr_sde == 0:
            self.tfr = max(0, self.tfr - self.tfr_d_step)   
                     
    def tqdm_bar(self, mode, pbar, loss, lr):
        pbar.set_description(f"({mode}) Epoch {self.current_epoch}, lr:{lr}" , refresh=False)
        pbar.set_postfix(loss=float(loss), refresh=False)
        pbar.refresh()
        
    def save(self, path):
        torch.save({
            "state_dict": self.state_dict(),
            "optimizer": self.optim.state_dict(),  
            # 原本沒加這三行可以Load回來train沒問題 只是沒有optimizer 跟 scheduler 的state_dict
            "scheduler"   : self.scheduler.state_dict(),  # ← 加這行

            "lr"        : self.scheduler.get_last_lr()[0],
            "tfr"       :   self.tfr,
            "last_epoch": self.current_epoch
        }, path)
        print(f"save ckpt to {path}")

    def load_checkpoint(self):
        if self.args.ckpt_path != None:
            checkpoint = torch.load(self.args.ckpt_path)
            self.load_state_dict(checkpoint['state_dict'], strict=True) 
            self.args.lr = checkpoint['lr']
            self.tfr = checkpoint['tfr']
            
            self.optim      = optim.Adam(self.parameters(), lr=self.args.lr)
            self.scheduler  = optim.lr_scheduler.MultiStepLR(self.optim, milestones=[2, 4], gamma=0.1)
            
            self.optim.load_state_dict(checkpoint['optimizer'])# ← 修正了這行
            self.scheduler.load_state_dict(checkpoint['scheduler'])  # ← 修正了這行

            self.kl_annealing = kl_annealing(self.args, current_epoch=checkpoint['last_epoch'])
            self.current_epoch = checkpoint['last_epoch']

    def optimizer_step(self):
        nn.utils.clip_grad_norm_(self.parameters(), 1.)
        self.optim.step()



def main(args):
    
    os.makedirs(args.save_root, exist_ok=True)
    model = VAE_Model(args).to(args.device)
    model.load_checkpoint()
    if args.test:
        model.eval()
    else:
        model.training_stage()




if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--batch_size',    type=int,    default=2)
    parser.add_argument('--lr',            type=float,  default=0.001,     help="initial learning rate")
    parser.add_argument('--device',        type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument('--optim',         type=str, choices=["Adam", "AdamW"], default="Adam")
    parser.add_argument('--gpu',           type=int, default=1)
    parser.add_argument('--test',          action='store_true')
    parser.add_argument('--store_visualization',      action='store_true', help="If you want to see the result while training")
    parser.add_argument('--DR',            type=str, required=True,  help="Your Dataset Path")
    parser.add_argument('--save_root',     type=str, required=True,  help="The path to save your data")
    parser.add_argument('--num_workers',   type=int, default=4)
    parser.add_argument('--num_epoch',     type=int, default=1000,     help="number of total epoch")
    parser.add_argument('--per_save',      type=int, default=10,      help="Save checkpoint every seted epoch")
    parser.add_argument('--partial',       type=float, default=1.0,  help="Part of the training dataset to be trained")
    parser.add_argument('--train_vi_len',  type=int, default=16,     help="Training video length")
    parser.add_argument('--val_vi_len',    type=int, default=630,    help="valdation video length")
    parser.add_argument('--frame_H',       type=int, default=32,     help="Height input image to be resize")
    parser.add_argument('--frame_W',       type=int, default=64,     help="Width input image to be resize")
    
    
    # Module parameters setting
    parser.add_argument('--F_dim',         type=int, default=128,    help="Dimension of feature human frame")
    parser.add_argument('--L_dim',         type=int, default=32,     help="Dimension of feature label frame")
    parser.add_argument('--N_dim',         type=int, default=12,     help="Dimension of the Noise")
    parser.add_argument('--D_out_dim',     type=int, default=192,    help="Dimension of the output in Decoder_Fusion")
    
    # Teacher Forcing strategy
    parser.add_argument('--tfr',           type=float, default=0.8,  help="The initial teacher forcing ratio")
    parser.add_argument('--tfr_sde',       type=int,   default=10,   help="The epoch that teacher forcing ratio start to decay")
    parser.add_argument('--tfr_d_step',    type=float, default=0.1,  help="Decay step that teacher forcing ratio adopted")
    parser.add_argument('--ckpt_path',     type=str,    default=None,help="The path of your checkpoints")   
    
    # Training Strategy
    parser.add_argument('--fast_train',         action='store_true')
    parser.add_argument('--fast_partial',       type=float, default=0.4,    help="Use part of the training data to fasten the convergence")
    parser.add_argument('--fast_train_epoch',   type=int, default=5,        help="Number of epoch to use fast train mode")
    
    # Kl annealing stratedy arguments
    parser.add_argument('--kl_anneal_type',     type=str, default='Cyclical',       help="")#  可設定 "Cyclical", "Monotonic","Without"
    parser.add_argument('--kl_anneal_cycle',    type=int, default=10,               help="")
    parser.add_argument('--kl_anneal_ratio',    type=float, default=1,              help="")
    

    

    args = parser.parse_args()
    
    main(args)
