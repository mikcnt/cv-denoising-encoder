import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision

class Checkpoint:
    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(path))

    def load(self, model, optimizer):
        self.checkpoint = torch.load(self.path, map_location=lambda storage, loc: storage)
        if self.checkpoint == None:
            raise RuntimeError("Checkpoint empty!")
        epoch = self.checkpoint["epoch"]
        model.load_state_dict(self.checkpoint["model_state_dict"])
        optimizer.load_state_dict(self.checkpoint["optimizer_state_dict"])
        train_loss = self.checkpoint["train_loss"]
        test_loss = self.checkpoint["test_loss"]
        return (model, optimizer, epoch, train_loss, test_loss)

    def save(self, model, optimizer, epoch, train_loss, test_loss):
        model_checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "train_loss": train_loss,
            "test_loss": test_loss,
        }
        torch.save(model_checkpoint, self.path)

