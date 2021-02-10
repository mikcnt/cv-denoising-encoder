import os
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torchvision.datasets import ImageFolder

from models import AutoEncoder
from dataset import ImageDataset
from parser import main_parser
from utils.checkpoint import Checkpoint
from utils.log import Output


def main():
    # Arguments from the parser
    parser = main_parser()
    args = parser.parse_args()

    # Hyperparameters
    RESUME_LAST = args.resume_last
    MODEL_CHECKPOINT = args.model_checkpoint
    h, w = 256, 256
    VAL_IMAGES = 40
    BATCH_SIZE = args.batch_size
    NUM_EPOCHS = args.epochs
    LEARNING_RATE = args.learning_rate
    DATA_PATH = args.data_path
    USE_DRIVE = args.use_drive
    TRAIN_DATA_PATH = os.path.join(DATA_PATH, "train")
    TEST_DATA_PATH = os.path.join(DATA_PATH, "test")
    cached = False

    # Device selection (gpu if available)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Data loading
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    # Noise parameters
    g_min = 0.08
    g_max = 0.12
    p_min = 0.05
    p_max = 0.08
    s_min = 0.03
    s_max = 0.05

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

    # Logging prep
    noise_output = Output("outputs/noise.png", VAL_IMAGES, overwrite=False)
    clean_output = Output("outputs/clean.png", VAL_IMAGES, overwrite=False)
    gen_output = Output("outputs", VAL_IMAGES, overwrite=True)

    # Resume model checkpoint if given. Otherwise start training from scratch.
    # Resume last if `resume_last` flag is True, otherwise manual checkpoint selection
    checkpoint = Checkpoint("checkpoints", RESUME_LAST)

    # Initialization of model, optimizer and criterion
    model = AutoEncoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCELoss()

    # Checkpoint loading
    try:
        model, optimizer, epoch, train_losses, test_losses = checkpoint.load(
            model, optimizer, MODEL_CHECKPOINT
        )
        print("Model loaded from checkpoint.")
        print("Starting training from epoch {}.".format(epoch))
    except RuntimeError:
        print("No valid checkpoint is given, starting training from scratch.")
        epoch = 0
        train_losses = {}
        test_losses = {}

    for epoch in range(epoch + 1, NUM_EPOCHS + 1):
        model.train()
        train_loss_epoch = 0
        test_loss_epoch = 0
        for noise, clean in tqdm(
            train_loader, ncols=70, desc="Epoch {}".format(epoch), leave=False
        ):
            noise = noise.to(device)
            clean = clean.to(device)
            fake = model(noise)

            # Training step
            optimizer.zero_grad()
            loss = criterion(fake, clean)
            loss.backward()
            optimizer.step()

            # Updating epoch loss
            train_loss_epoch += loss.item()

        # Evaluation step
        model.eval()
        with torch.no_grad():
            num_batches = VAL_IMAGES // BATCH_SIZE + 1

            if not cached:
                cache = []
                for batch_idx, (noise_test, clean_test) in enumerate(
                    tqdm(test_loader, ncols=70, desc="Test", leave=False)
                ):
                    cache.append((noise_test, clean_test))
                    noise_test = noise_test.to(device)
                    clean_test = clean_test.to(device)
                    fake_test = model(noise_test)

                    loss_test = criterion(fake_test, clean_test)
                    test_loss_epoch += loss_test.item()

                    if batch_idx < num_batches:
                        noise_output.append(noise_test)
                        clean_output.append(clean_test)
                        gen_output.append(fake_test)

                cached = True
            else:
                for batch_idx, (noise_test, clean_test) in enumerate(
                    tqdm(cache, ncols=70, desc="Validation", leave=False)
                ):
                    noise_test = noise_test.to(device)
                    clean_test = clean_test.to(device)
                    fake_test = model(noise_test)

                    loss_test = criterion(fake_test, clean_test)
                    test_loss_epoch += loss_test.item()

                    if batch_idx < num_batches:
                        noise_output.append(noise_test)
                        clean_output.append(clean_test)
                        gen_output.append(fake_test)

        # Store losses of the epoch in the appropriate dictionaries
        train_losses[epoch] = train_loss_epoch
        test_losses[epoch] = test_loss_epoch

        print(
            f"\n\nEpoch: {epoch}\n"
            f"Train loss: {train_loss_epoch:.4f}\n"
            f"Validation loss: {test_loss_epoch:.4f}\n"
        )
        # Save output images
        noise_output.save()
        clean_output.save()
        gen_output.save(filename="{}_fake.png".format(str(epoch).zfill(3)))

        # Save checkpoint
        if USE_DRIVE:
            checkpoint.save_drive(model, optimizer, epoch, train_losses, test_losses)
        else:
            checkpoint.save(model, optimizer, epoch, train_losses, test_losses)


if __name__ == "__main__":
    main()
