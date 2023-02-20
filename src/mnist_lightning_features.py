import os
from argparse import ArgumentParser

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.datasets import MNIST
from torch.optim import Adam

from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping


class Net(LightningModule):
    def __init__(self, batch_size, hidden_size, learning_rate, **kwargs):
        super(Net, self).__init__()
        self.save_hyperparameters()

        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.fc1 = nn.Linear(28 * 28, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 10)

    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Net")
        parser.add_argument("--batch_size", type=int, default=64)
        parser.add_argument("--hidden_size", type=int, default=128)
        parser.add_argument("--learning_rate", type=float, default=1e-3)
        return parent_parser

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

    def train_dataloader(self):
        mnist_train = MNIST(
            os.getcwd(), train=True, download=True, transform=transforms.ToTensor()
        )
        return DataLoader(mnist_train, batch_size=self.batch_size)

    def val_dataloader(self):
        mnist_train = MNIST(
            os.getcwd(), train=False, download=True, transform=transforms.ToTensor()
        )
        return DataLoader(mnist_train, batch_size=self.batch_size)

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams, {"val/accuracy": 0})

    def training_step(self, batch, batch_idx):
        data, target = batch
        output = self.forward(data)
        loss = F.nll_loss(output, target)
        self.log("train/loss", loss)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        data, target = batch
        output = self.forward(data)
        loss = F.nll_loss(output, target)
        pred = output.argmax(dim=1, keepdim=True)
        correct = pred.squeeze(1).eq(target).sum().item()
        self.log("val/loss", loss)
        return {"loss": loss, "correct": correct, "total": len(target)}

    def validation_epoch_end(self, outs):
        num_correct = sum(map(lambda x: x[f"correct"], outs), 0)
        num_total = sum(map(lambda x: x[f"total"], outs), 0)
        self.log("val/accuracy", num_correct / num_total)


if __name__ == "__main__":
    seed_everything(42)

    parser = ArgumentParser()
    parser = Net.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    logger = TensorBoardLogger(save_dir=".", default_hp_metric=False)

    net = Net(**vars(args))

    checkpoint_callback = ModelCheckpoint(
        monitor="val/accuracy", mode="max", verbose=True
    )
    early_stopping_callback = EarlyStopping(
        monitor="val/accuracy", mode="max", patience=2
    )
    trainer = Trainer.from_argparse_args(
        args,
        callbacks=[checkpoint_callback, early_stopping_callback],
        accelerator="gpu",
        max_epochs=10,
        logger=logger,
    )
    trainer.fit(net)
