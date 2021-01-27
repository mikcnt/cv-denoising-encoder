import os
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
import torchvision

from models import AutoEncoder, GeneratorLoss
from models import Discriminator, DiscriminatorLoss
from dataset import ImageDataset
from parser import main_parser
from utils.checkpoint import Checkpoint


def main():
    # Load arguments from the parser
    parser = main_parser()
    args = parser.parse_args()

    # Hyperparameters
    RESUME_LAST = args.resume_last
    GENERATOR_CHECKPOINT = args.generator_checkpoint
    DISCRIMINATOR_CHECKPOINT = args.discriminator_checkpoint
    h, w = 256, 256
    VAL_IMAGES = 40
    BATCH_SIZE = args.batch_size
    NUM_EPOCHS = args.epochs
    LEARNING_RATE = args.learning_rate
    DATA_PATH = args.data_path
    TRAIN_DATA_PATH = os.path.join(DATA_PATH, "train")
    TEST_DATA_PATH = os.path.join(DATA_PATH, "test")

    # Select device for training (gpu if available)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Starting epoch
    epoch = 0

    # Resume last checkpoints

    checkpoints_path = {
        "discriminator": "checkpoints/discriminator",
        "generator": "checkpoints/generator",
    }

    if RESUME_LAST:
        try:
            gen_path = sorted(os.listdir(checkpoints_path["generator"]))[-1]
            dis_path = sorted(os.listdir(checkpoints_path["discriminator"]))[-1]
            gen_path = os.path.join(checkpoints_path["generator"], gen_path)
            dis_path = os.path.join(checkpoints_path["discriminator"], dis_path)
        except IndexError:
            print("One of the checkpoint doesn't exists")
            exit(1)

    if GENERATOR_CHECKPOINT:
        gen_path = GENERATOR_CHECKPOINT

    if DISCRIMINATOR_CHECKPOINT:
        dis_path = DISCRIMINATOR_CHECKPOINT

    # Instantiate losses
    train_losses = {}
    test_losses = {}
    try:
        gen_checkpoint = Checkpoint(gen_path)
        dis_checkpoint = Checkpoint(dis_path)

        generator = AutoEncoder().to(device)
        discriminator = Discriminator().to(device)

        dis_opt = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE)
        gen_opt = optim.Adam(generator.parameters(), lr=LEARNING_RATE)

        gen_criterion = GeneratorLoss()
        dis_criterion = DiscriminatorLoss()

        generator, gen_opt, epoch, gen_train_losses, gen_test_losses = gen_checkpoint.load(generator, gen_opt)
        discriminator, dis_opt, epoch, dis_train_losses, dis_test_losses = dis_checkpoint.load(discriminator, dis_opt)
        print("Models loaded from checkpoints!")
    except RuntimeError:
        print("No checkpoints, so the models are new!")

    # Load data
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    g_min = 0.05
    g_max = 0.08
    p_min = 0.1
    p_max = 0.2
    s_min = 0.03
    s_max = 0.06

    train_dataset = ImageDataset(
        TRAIN_DATA_PATH,
        transform=transform,
        g_min=g_min,
        g_max=g_max,
        p_min=p_min,
        p_max=p_max,
        s_min=s_min,
        s_max=s_max,
    )
    test_dataset = ImageDataset(
        TEST_DATA_PATH,
        transform=transform,
        g_min=g_max,
        g_max=g_max,
        p_min=p_max,
        p_max=p_max,
        s_min=s_max,
        s_max=s_max,
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    for epoch in range(epoch + 1, NUM_EPOCHS + 1):
        generator.train()
        discriminator.train()

        gen_train_loss_epoch = 0
        dis_train_loss_epoch = 0
        gen_test_loss_epoch = 0
        dis_test_loss_epoch = 0

        for noise, clean in tqdm(train_loader, ncols=70, desc="Epoch {}".format(epoch)):
            noise = noise.to(device)
            clean = clean.to(device)

            fake = generator(noise)
            prediction_real = discriminator(clean)
            prediction_fake = discriminator(fake)

            ones = torch.ones_like(prediction_real)
            zeros = torch.zeros_like(prediction_fake)

            # Train Discriminator
            dis_loss = dis_criterion(prediction_real, ones, prediction_fake, zeros)
            dis_loss.backward()

            dis_opt.zero_grad()
            dis_opt.step()

            # Train generator
            gen_loss = gen_criterion(prediction_fake, ones)
            gen_loss.backward()

            gen_opt.zero_grad()
            gen_opt.step()

            # Storing the losses of the epoch
            gen_train_loss_epoch += gen_loss.item()
            dis_train_loss_epoch += dis_loss.item()

        with torch.no_grad():
            noise_images = []
            clean_images = []
            gen_images = []
            num_batches = VAL_IMAGES // BATCH_SIZE + 1

            for batch_idx, (noise_test, clean_test) in enumerate(
                tqdm(test_loader, ncols=70, desc="Validation")
            ):
                x_test = x_test.to(device)
                t_test = t_test.to(device)
                y_test = model(x_test)

                loss_test = criterion(y_test, t_test)
                test_loss_epoch += loss_test.item()

                if batch_idx < num_batches:
                    noise_images.append(x_test)
                    clean_images.append(t_test)
                    gen_images.append(y_test)

        train_losses[epoch] = train_loss_epoch
        test_losses[epoch] = test_loss_epoch

        print(
            "Train loss = {:.4f} \t Test loss = {:.4f}".format(
                train_loss_epoch, test_loss_epoch
            )
        )

        noise_path = "outputs/noise.png"
        real_path = "outputs/real.png"
        gen_path = "outputs/{}_fake.png".format(str(epoch).zfill(3))

        noise_images = torch.cat(noise_images, dim=0)[:VAL_IMAGES, ...]
        clean_images = torch.cat(clean_images, dim=0)[:VAL_IMAGES, ...]
        gen_images = torch.cat(gen_images, dim=0)[:VAL_IMAGES, ...]

        img_grid_noise = torchvision.utils.make_grid(noise_images, nrow=8)
        img_grid_clean = torchvision.utils.make_grid(clean_images, nrow=8)
        img_grid_gen = torchvision.utils.make_grid(gen_images, nrow=8)

        torchvision.utils.save_image(img_grid_noise, noise_path)
        torchvision.utils.save_image(img_grid_clean, real_path)
        torchvision.utils.save_image(img_grid_gen, gen_path)

        check_path = "checkpoints/{}.pth".format(str(epoch).zfill(3))

        save_checkpoint(model, optimizer, epoch, train_losses, test_losses, check_path)


if __name__ == "__main__":
    main()
