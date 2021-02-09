import os
import torch
from torch.utils.data import DataLoader
import torchvision
import numpy as np
import cv2
from tqdm import tqdm
import pickle
import argparse

from architectures.old_autoencoder import OldAutoEncoder
from architectures.gan import Generator
from utils.checkpoint import Checkpoint
from dataset import ImageDataset, RenderDataset
import utils.noise as noise
from utils.measures import ssim, PSNR


parser = argparse.ArgumentParser(description="Evaluation parser")

parser.add_argument(
    "--dataset",
    default="",
    type=str,
    help="dataset type",
)

parser.add_argument(
    "--noise",
    default="all",
    type=str,
    help="noise type",
)

args = parser.parse_args()

if args.dataset == "coco":
    path = "coco/test"
    if args.noise == "all":
        save_name = "coco_gaussian"
        g_min = 0.05
        g_max = 0.10
        p_min = 0.10
        p_max = 0.40
        s_min = 0.05
        s_max = 0.20
    elif args.noise == "gaussian":
        save_name = "coco_all"
        g_min = 0.05
        g_max = 0.10
        p_min = 0.0
        p_max = 0.0
        s_min = 0.0
        s_max = 0.0
    else:
        raise AssertionError("Noise tyoe not valid.")

    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    dataset = ImageDataset(
        path,
        transform=transform,
        g_min=g_max,
        g_max=g_max,
        p_min=p_max,
        p_max=p_max,
        s_min=s_max,
        s_max=s_max,
    )

    loader = DataLoader(dataset, batch_size=8, shuffle=False)
elif args.dataset == "render":
    path = "data_rend/train"
    save_name = "render"
    transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(), torchvision.transforms.Resize(256, 256)]
    )
    dataset = RenderDataset(path, transform=transform)
    loader = DataLoader(dataset, batch_size=8, shuffle=False)
else:
    raise AssertionError("Dataset type not valid.")


device = "cuda" if torch.cuda.is_available() else "cpu"

generator = Generator().to(device)

gan_checkpoint_path = "best_models/300_gan_pixar.pth"
gan_checkpoint = torch.load(
    gan_checkpoint_path, map_location=lambda storage, loc: storage
)
generator.load_state_dict(gan_checkpoint["model_state_dict"])

autoencoder = OldAutoEncoder().to(device)

encoder_checkpoint_path = "best_models/200_tconv.pth"
encoder_checkpoint = torch.load(
    encoder_checkpoint_path, map_location=lambda storage, loc: storage
)
autoencoder.load_state_dict(encoder_checkpoint["model_state_dict"])


psnr = PSNR()

gan_acc = {
    "psnr": {
        "noise": 0,
        "fake": 0,
    },
    "ssim": {
        "noise": 0,
        "fake": 0,
    },
}

encoder_acc = {
    "psnr": {
        "noise": 0,
        "fake": 0,
    },
    "ssim": {
        "noise": 0,
        "fake": 0,
    },
}

generator.eval()
autoencoder.eval()
for noise, clean in tqdm(loader):
    noise = noise.to(device)
    clean = clean.to(device)

    fake_gan = generator(noise)
    fake_encoder = encoder(noise)

    clean = clean * 255
    noise = noise * 255
    fake_gan = fake_gan * 255
    fake_encoder = fake_encoder * 255

    gan_acc["psnr"]["noise"] += psnr(noise, clean)
    gan_acc["psnr"]["fake"] += psnr(fake_gan, clean)
    gan_acc["ssim"]["noise"] += ssim(noise, clean)
    gan_acc["ssim"]["fake"] += ssim(fake_gan, clean)

    encoder_acc["psnr"]["noise"] += psnr(noise, clean)
    encoder_acc["psnr"]["fake"] += psnr(fake_encoder, clean)
    encoder_acc["ssim"]["noise"] += ssim(noise, clean)
    encoder_acc["ssim"]["fake"] += ssim(fake_encoder, clean)

os.makedirs("evaluations", exist_ok=True)

with open("evaluations/gan_acc_{}.pkl".format(save_name), "wb") as f:
    pickle.dump(gan_acc, f, protocol=pickle.HIGHEST_PROTOCOL)

with open("evaluations/encoder_acc_{}.pkl".format(save_name), "wb") as f:
    pickle.dump(encoder_acc, f, protocol=pickle.HIGHEST_PROTOCOL)