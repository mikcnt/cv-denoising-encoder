import os
import torch
from torch.utils.data import DataLoader
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import pickle
import argparse

from architectures.old_autoencoder import OldAutoEncoder
from architectures.gan import Generator
from utils.checkpoint import Checkpoint
from dataset import ImageDataset, RenderDataset
import utils.noise as noise


parser = argparse.ArgumentParser(description="Evaluation parser")


parser.add_argument(
    "--model",
    default="",
    type=str,
    help="model type",
)
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
parser.add_argument(
    "--use_drive",
    dest="use_drive",
    action="store_true",
    help="use this flag to load checkpoints and save stats on drive",
)
parser.add_argument("--batch_size", default=8, type=int, help="batch size (default: 8)")
parser.add_argument("--num_images", default=0, type=int, help="number of images (default: 0)")

args = parser.parse_args()

if args.dataset == "coco":
    path = "coco/test"
    if args.noise == "all":
        dataset_name = "coco_all"
        g_min = 0.08
        g_max = 0.12
        p_min = 0.05
        p_max = 0.08
        s_min = 0.03
        s_max = 0.05
    elif args.noise == "gaussian":
        dataset_name = "coco_gaussian"
        g_min = 0.08
        g_max = 0.12
        p_min = 0.0
        p_max = 0.0
        s_min = 0.0
        s_max = 0.0
    else:
        raise AssertionError("Noise type not valid.")

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
        seed=True
    )

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
elif args.dataset == "render":
    path = "data_rend/train"
    dataset_name = "render"
    transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(), torchvision.transforms.Resize((256, 256))]
    )
    dataset = RenderDataset(path, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
else:
    raise AssertionError("Dataset type not valid.")

print("Dataset loaded.")

device = "cuda" if torch.cuda.is_available() else "cpu"

if args.model == "gan":

    gan_checkpoint_path = "best_models/300_gan_pixar.pth"

    if args.use_drive:
        gan_checkpoint_path = os.path.join(
            "/content/drive/MyDrive", gan_checkpoint_path
        )

    model = Generator().to(device)

    gan_checkpoint = torch.load(
        gan_checkpoint_path, map_location=lambda storage, loc: storage
    )
    model.load_state_dict(gan_checkpoint["model_state_dict"])

    print("GAN checkpoint loaded.")
elif args.model == "encoder":
    encoder_checkpoint_path = "best_models/200_tconv.pth"
    if args.use_drive:
        encoder_checkpoint_path = os.path.join(
            "/content/drive/MyDrive", encoder_checkpoint_path
        )

    model = OldAutoEncoder().to(device)

    encoder_checkpoint = torch.load(
        encoder_checkpoint_path, map_location=lambda storage, loc: storage
    )
    model.load_state_dict(encoder_checkpoint["model_state_dict"])

    print("Autoencoder checkpoint loaded.")
else:
    raise AssertionError("Model type not valid.")

results_path = os.path.join("results", f"{args.model}_{dataset_name}")

if args.use_drive:
    results_path = os.path.join("/content/drive/MyDrive", results_path)

os.makedirs(results_path, exist_ok=True)

model.eval()
i = 1
for noise, clean in tqdm(loader):
    noise = noise.to(device)
    clean = clean.to(device)
    fake = model(noise)
    for clean_img, noise_img, fake_img in zip(clean, noise, fake):
        clean_path = os.path.join(results_path, "{}_clean.png".format(str(i).zfill(3)))
        noise_path = os.path.join(results_path, "{}_noise.png".format(str(i).zfill(3)))
        fake_path = os.path.join(results_path, "{}_fake.png".format(str(i).zfill(3)))
        clean_img = clean_img.cpu().detach().permute(1, 2, 0).numpy()
        noise_img = noise_img.cpu().detach().permute(1, 2, 0).numpy()
        fake_img = np.clip(fake_img.cpu().detach().permute(1, 2, 0).numpy(), 0, 1)
        
        plt.imsave(clean_path, clean_img)
        plt.imsave(noise_path, noise_img)
        plt.imsave(fake_path, fake_img)
        i += 1
        if i == args.num_images:
            exit()