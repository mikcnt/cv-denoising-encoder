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
from utils.measures import ssim, psnr


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

args = parser.parse_args()

if args.dataset == "coco":
    path = "coco/test"
    if args.noise == "all":
        save_name = "coco_all"
        g_min = 0.05
        g_max = 0.10
        p_min = 0.10
        p_max = 0.40
        s_min = 0.05
        s_max = 0.20
    elif args.noise == "gaussian":
        save_name = "coco_gaussian"
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

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
elif args.dataset == "render":
    path = "data_rend/train"
    save_name = "render"
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
    model = Generator().to(device)

    gan_checkpoint_path = "best_models/300_gan_pixar.pth"

    if args.use_drive:
        gan_checkpoint_path = os.path.join(
            "/content/drive/MyDrive", gan_checkpoint_path
        )

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


model_acc = {
    "psnr": {
        "noise": [],
        "fake": [],
    },
    "ssim": {
        "noise": [],
        "fake": [],
    },
}

model.eval()
for noise, clean in tqdm(loader):
    noise = noise.to(device)
    clean = clean.to(device)

    fake = model(noise)

    clean = clean * 255
    noise = noise * 255
    fake = fake * 255

    model_acc["psnr"]["noise"] += psnr(noise, clean).item()
    model_acc["psnr"]["fake"] += psnr(fake, clean).item()
    model_acc["ssim"]["noise"] += ssim(noise, clean).item()
    model_acc["ssim"]["fake"] += ssim(fake, clean).item()

def mean(item):
    return sum(item) / len(item)

for measure in model_acc:
    for key in model_acc[measure]:
        model_acc[measure][key] = mean(model_acc[measure][key])

evaluations_path = "evaluations"

if args.use_drive:
    evaluations_path = os.path.join("/content/drive/MyDrive", evaluations_path)

model_path = os.path.join(
    evaluations_path, "{}_acc_{}.pkl".format(args.model, save_name)
)

os.makedirs(evaluations_path, exist_ok=True)

with open(model_path, "wb") as f:
    pickle.dump(model_acc, f, protocol=pickle.HIGHEST_PROTOCOL)