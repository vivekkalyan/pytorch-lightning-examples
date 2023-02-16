import os
from argparse import ArgumentParser

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.datasets import MNIST
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from pytorch_lightning import LightningModule, Trainer, seed_everything


class Net(LightningModule):
    def __init__(self, batch_size, hidden_size, learning_rate):
        super(Net, self).__init__()

        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 10)

    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Net")
        parser.add_argument("--batch_size", type=int, default=64)
        parser.add_argument("--hidden_size", type=int, default=128)
        parser.add_argument("--learning_rate", type=float, default=1e-3)
        return parent_parser

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
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

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.learning_rate)
        scheduler = StepLR(optimizer, step_size=1)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def training_step(self, batch, batch_idx):
        data, target = batch
        output = self.forward(data)
        loss = F.nll_loss(output, target)
        return {"loss": loss}


if __name__ == "__main__":
    seed_everything(42)

    parser = ArgumentParser()
    parser = Net.add_model_specific_args(parser)
    args = parser.parse_args()

    net = Net(**vars(args))
    trainer = Trainer(accelerator="gpu", max_epochs=10)
    trainer.fit(net)
