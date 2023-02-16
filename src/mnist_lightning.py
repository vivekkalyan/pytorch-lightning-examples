import os

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.datasets import MNIST
from torch.optim import Adam

from pytorch_lightning import LightningModule, Trainer


class Net(LightningModule):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)

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
        return DataLoader(mnist_train, batch_size=64)

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        data, target = batch
        output = self.forward(data)
        loss = F.nll_loss(output, target)
        return {"loss": loss}


if __name__ == "__main__":
    net = Net()
    trainer = Trainer(accelerator="gpu", max_epochs=10)
    trainer.fit(net)
