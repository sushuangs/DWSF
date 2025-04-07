import os
import json
import hashlib
import numpy as np
from pathlib import Path
import torch
from torch import nn
import numpy as np
from thop import profile
from PIL import Image
import shutil

import kornia
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from networks.models.Encoder import Encoder
from networks.models.Decoder import Decoder
from networks.models.network_scunet import SCUNet


def psnr_ssim_acc(image, H_img, L_img):
    # psnr
    H_psnr = kornia.metrics.psnr(
        ((image + 1) / 2).clamp(0, 1),
        ((H_img.detach() + 1) / 2).clamp(0, 1),
        1,
    )
    L_psnr = kornia.metrics.psnr(
        ((image + 1) / 2).clamp(0, 1),
        ((L_img.detach() + 1) / 2).clamp(0, 1),
        1,
    )
    # ssim
    H_ssim = kornia.metrics.ssim(
        ((image + 1) / 2).clamp(0, 1),
        ((H_img.detach() + 1) / 2).clamp(0, 1),
        window_size=11,
    ).mean()
    L_ssim = kornia.metrics.ssim(
        ((image + 1) / 2).clamp(0, 1),
        ((L_img.detach() + 1) / 2).clamp(0, 1),
        window_size=11,
    ).mean()
    return H_psnr, L_psnr, H_ssim, L_ssim


class ImageProcessingDataset(Dataset):
    def __init__(self, root_dir):
        self.root = root_dir
        self.image_paths = []
        self.rel_dirs = []

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        

        for root, _, files in os.walk(root_dir):
            rel_dir = os.path.relpath(root, root_dir)
            for f in files:
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    self.image_paths.append(os.path.join(root, f))
                    self.rel_dirs.append(rel_dir)

    def __len__(self):
        return len(self.image_paths)
    
    def generate_binary_seed(self, seed_str: str) -> int:
        seed_str = seed_str.lower().replace("\\", "/").split('/')[-1]
        seed_str = seed_str.split('.')[0]
        hash_bytes = hashlib.sha256(seed_str.encode("utf-8")).digest()
        return int.from_bytes(hash_bytes[:4], byteorder="big")

    def generate_binary_data(self, seed: int, length: int = 30) -> list:
        rng = np.random.RandomState(seed)
        return rng.randint(0, 2, size=length).tolist()

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            img = Image.open(img_path).convert('RGB')
            seed = self.generate_binary_seed(img_path)
            binary_data = self.generate_binary_data(seed, 30)
            return self.transform(img), idx, torch.tensor(binary_data, dtype=torch.float32)
        except Exception as e:
            print(f"Error loading {img_path}: {str(e)}")
            return None, idx



if __name__ == "__main__":

    input_root = "/data/experiment/data/gtos128_all/val"
    batch_size = 32
    num_workers = 4

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    '''
    model
    '''
    encoder = Encoder(H=128, W=128, message_length=30)
    decoder = Decoder(message_length=30)
    e_checkpoint = torch.load('/data/experiment/model/DWSF/runs/gtos_I_psnr/pth/encoder_best_psnr-40.716888427734375.pth')
    d_checkpoint = torch.load('/data/experiment/model/DWSF/runs/gtos_I_psnr/pth/decoder_best_psnr-40.716888427734375.pth')
    encoder.load_state_dict(e_checkpoint)
    decoder.load_state_dict(d_checkpoint)
    encoder.to(device)
    decoder.to(device)
    encoder.eval()
    decoder.eval()
    '''
    model
    '''

    scunet = SCUNet(in_nc=3,config=[4,4,4,4,4,4,4],dim=64)

    scunet.load_state_dict(torch.load('/data/experiment/model/SCUNet/runs/gtos_HiDDeN_I_35-2025-04-02-21:36-train/checkpoint/gtos_GN_75--epoch-5.pth')['network'], strict=True)
    scunet.to(device)
    scunet.eval()

    dataset = ImageProcessingDataset(input_root)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    gaussian_blur = transforms.GaussianBlur(
        kernel_size=7,
        sigma=1.0
    )
    sigma = [0, 15, 25, 50, 75]
    clean = 0
    for n in sigma:
        bitwise_avg_err_n_history = []
        bitwise_avg_err_r_history = []
        bitwise_avg_err_g_history = []
        oH_psnrs = []
        oL_psnrs = []
        iH_psnrs = []
        iL_psnrs = []
        g_psnrs = []
        N_psnrs = []
        with torch.no_grad():
            for data in dataloader:
                inputs, indices, message = data

                noise = torch.Tensor(np.random.normal(0, n, inputs.shape)/128.).to(device)
                
                inputs = inputs.to(device)

                message = message.to(device)

                output_img = encoder(inputs, message)
                output_img_n = output_img + noise
                output_img_g = gaussian_blur(output_img_n)
                output_img_r = scunet(output_img_n)

                decoded_messages_n = decoder(output_img_n)
                decoded_messages_r = decoder(output_img_r)
                decoded_messages_g = decoder(output_img_g)
                
                decoded_rounded_n = decoded_messages_n.detach().cpu().numpy().round().clip(0, 1)
                bitwise_avg_err_n = np.sum(np.abs(decoded_rounded_n - message.detach().cpu().numpy())) / (
                        batch_size * 30)

                decoded_rounded_r = decoded_messages_r.detach().cpu().numpy().round().clip(0, 1)
                bitwise_avg_err_r = np.sum(np.abs(decoded_rounded_r - message.detach().cpu().numpy())) / (
                        batch_size * 30)
                
                decoded_rounded_g = decoded_messages_g.detach().cpu().numpy().round().clip(0, 1)
                bitwise_avg_err_g = np.sum(np.abs(decoded_rounded_g - message.detach().cpu().numpy())) / (
                        batch_size * 30)
                oH_psnr, oL_psnr, _ , _ = psnr_ssim_acc(output_img.cpu(), output_img_r.cpu(), output_img_n.cpu())
                iH_psnr, iL_psnr, _ , _ = psnr_ssim_acc(inputs.cpu(), output_img_r.cpu(), output_img_n.cpu())
                N_psnr, _, _ , _ = psnr_ssim_acc(inputs.cpu(), (noise + inputs).cpu(), output_img_n.cpu())
                g_psnr, _, _ , _ = psnr_ssim_acc(output_img.cpu(), output_img_g.cpu(), output_img_n.cpu())
                oH_psnrs.append(oH_psnr)
                oL_psnrs.append(oL_psnr)
                iH_psnrs.append(iH_psnr)
                iL_psnrs.append(iL_psnr)
                g_psnrs.append(g_psnr)
                N_psnrs.append(N_psnr)
                bitwise_avg_err_n_history.append(bitwise_avg_err_n)
                bitwise_avg_err_r_history.append(bitwise_avg_err_r)
                bitwise_avg_err_g_history.append(bitwise_avg_err_g)
        noise = 1 - np.mean(bitwise_avg_err_n_history)
        recover = 1 - np.mean(bitwise_avg_err_r_history)
        blur = 1 - np.mean(bitwise_avg_err_g_history)
        if n == 0:
            clean = 1 - np.mean(bitwise_avg_err_n_history)
        else:
            revover_rate = (recover - noise) / (clean - noise)
            print('in sigma {}, recovery rate {:.4f}'.format(n, revover_rate * 100))
        imporve_rate = (recover - noise) / noise
        print('in sigma {}, increase rate {:.4f}'.format(n, imporve_rate * 100))
        print('in sigma {}, nosie image accuracy {:.4f}'.format(n, noise * 100))
        print('in sigma {}, blur image accuracy {:.4f}'.format(n, blur * 100))
        print('in sigma {}, recover image accuracy {:.4f}'.format(n, recover * 100))
        print('in sigma {}, H_psnr_wm_to_r {:.4f}'.format(n, np.mean(oH_psnrs)))
        print('in sigma {}, L_psnr_wm_to_n {:.4f}'.format(n, np.mean(oL_psnrs)))
        print('in sigma {}, H_psnr_ori_to_r {:.4f}'.format(n, np.mean(iH_psnrs)))
        print('in sigma {}, L_psnr_ori_to_n {:.4f}'.format(n, np.mean(iL_psnrs)))
        print('in sigma {}, psnr GaussianBlur {:.4f}'.format(n, np.mean(g_psnrs)))
        print('in sigma {}, N_psnr {:.4f}'.format(n, np.mean(N_psnrs)))