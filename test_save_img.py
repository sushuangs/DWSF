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
    ############
    #your model#
    #   start  #
    ############
from networks.models.Encoder import Encoder
from networks.models.Decoder import Decoder
    ############
    #your model#
    #    end   #
    ############

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

    input_root = "/data/experiment/model/DWSF/DWSF_40_gtos/val" # your path
    batch_size = 32
    num_workers = 4

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    ############
    #your model#
    #   start  #
    ############
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

    ############
    #your model#
    #    end   #
    ############

    dataset = ImageProcessingDataset(input_root)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    bitwise_avg_err_history = []
    with torch.no_grad():
        for data in dataloader:
            inputs, indices, message = data
            
            inputs = inputs.to(device)

            message = message.to(device)

            output_img = encoder(inputs, message)

            decoded_messages = decoder(output_img)

            
            decoded_rounded = decoded_messages.detach().cpu().numpy().round().clip(0, 1)
            bitwise_avg_err = np.sum(np.abs(decoded_rounded - message.detach().cpu().numpy())) / (
                    batch_size * 30)
            
            bitwise_avg_err_history.append(bitwise_avg_err)

    acc = 1 - np.mean(bitwise_avg_err_history)
    print('acc {:.4f}'.format(acc))