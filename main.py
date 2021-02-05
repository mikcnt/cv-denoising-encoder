import os
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import albumentations as A

from models import AutoEncoder
from dataset import RenderDataset
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
    TRAIN_DATA_PATH = os.path.join(DATA_PATH, "train")
    TEST_DATA_PATH = os.path.join(DATA_PATH, "test")

    USE_DRIVE = args.use_drive

    # Device selection (gpu if available)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Data loading
    transform = A.Compose(
        [
            A.Resize(width=350, height=350),
            A.RandomCrop(width=256, height=256),
            A.Rotate(limit=40, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=1),
            A.ChannelShuffle(p=0.2),
            A.CLAHE(p=0.5),
            A.ToGray(p=0.2),
            A.OneOf(
                [
                    A.Blur(blur_limit=3, p=0.5),
                    A.ColorJitter(p=0.5),
                ],
                p=1.0,
            ),
            A.ElasticTransform(p=0.3),
            A.RandomBrightness(),
        ],
        additional_targets={"image0": "image", "image1": "image"},
    )

    train_dataset = RenderDataset(images_folder=TRAIN_DATA_PATH, transform=transform)
    # test_dataset = RenderDataset(images_folder=TEST_DATA_PATH, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Logging prep
    # noise_output = Output("outputs/noise.png", VAL_IMAGES, overwrite=False)
    # clean_output = Output("outputs/clean.png", VAL_IMAGES, overwrite=False)
    # gen_output = Output("outputs", VAL_IMAGES, overwrite=True)

    # Resume model checkpoint if given. Otherwise start training from scratch.
    # Resume last if `resume_last` flag is True, otherwise manual checkpoint selection
    checkpoint = Checkpoint("checkpoints", RESUME_LAST)

    # Initialization of model, optimizer and criterion
    model = AutoEncoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

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
        for noise, clean in tqdm(train_loader, ncols=70, desc="Epoch {}".format(epoch)):
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

        # # Evaluation step
        # model.eval()
        # with torch.no_grad():
        #     num_batches = VAL_IMAGES // BATCH_SIZE + 1

        #     for batch_idx, (noise_test, clean_test) in enumerate(
        #         tqdm(test_loader, ncols=70, desc="Validation")
        #     ):
        #         noise_test = noise_test.to(device)
        #         clean_test = clean_test.to(device)
        #         fake_test = model(noise_test)

        #         loss_test = criterion(fake_test, clean_test)
        #         test_loss_epoch += loss_test.item()

        #         if batch_idx < num_batches:
        #             noise_output.append(noise_test)
        #             clean_output.append(clean_test)
        #             gen_output.append(fake_test)

        # Store losses of the epoch in the appropriate dictionaries
        train_losses[epoch] = train_loss_epoch
        test_losses[epoch] = test_loss_epoch

        print(
            "Train loss = {:.4f} \t Test loss = {:.4f}".format(
                train_loss_epoch, test_loss_epoch
            )
        )

        # Save output images
        # noise_output.save()
        # clean_output.save()
        # gen_output.save(filename="{}_fake.png".format(str(epoch).zfill(3)))

        # Save checkpoint
        if USE_DRIVE:
            checkpoint.save_drive(model, optimizer, epoch, train_losses, test_losses)
        else:
            checkpoint.save(model, optimizer, epoch, train_losses, test_losses)


if __name__ == "__main__":
    main()
