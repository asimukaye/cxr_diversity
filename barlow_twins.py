from functools import partial
from typing import Sequence, Tuple, Union

import pytorch_lightning as L
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models.resnet import resnet18, ResNet18_Weights

import numpy as np

class BarlowTwinsTransform:
    def __init__(self, train=True, input_height=224, gaussian_blur=True, jitter_strength=1.0, normalize=None):
        self.input_height = input_height
        self.gaussian_blur = gaussian_blur
        self.jitter_strength = jitter_strength
        self.normalize = normalize
        self.train = train

        color_jitter = transforms.ColorJitter(
            0.8 * self.jitter_strength,
            0.8 * self.jitter_strength,
            0.8 * self.jitter_strength,
            0.2 * self.jitter_strength,
        )

        color_transform = [transforms.RandomApply([color_jitter], p=0.8), transforms.RandomGrayscale(p=0.2)]

        if self.gaussian_blur:
            kernel_size = int(0.1 * self.input_height)
            if kernel_size % 2 == 0:
                kernel_size += 1

            color_transform.append(transforms.RandomApply([transforms.GaussianBlur(kernel_size=kernel_size)], p=0.5))

        self.color_transform = transforms.Compose(color_transform)

        # if normalize is None:
        #     self.final_transform = transforms.ToTensor()
        # else:
        #     self.final_transform = transforms.Compose([transforms.ToTensor(), normalize])
        if train:
            self.transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(self.input_height),
                    transforms.RandomHorizontalFlip(p=0.5),
                    self.color_transform,
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ]
            )
        else:
            self.transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __call__(self, sample):
        return self.transform(sample), self.transform(sample)


class BarlowTwinsLoss(nn.Module):
    def __init__(self, batch_size, lambda_coeff=5e-3, z_dim=128, mode='train'):
        super().__init__()

        self.z_dim = z_dim
        self.batch_size = batch_size
        self.lambda_coeff = lambda_coeff

    def off_diagonal_ele_batch(self, x):
        # taken from: https://github.com/facebookresearch/barlowtwins/blob/main/main.py
        # return a flattened view of the off-diagonal elements of a square matrix
        res = x.clone()
        res.diagonal(dim1=-1, dim2=-2).zero_()
        return res
    

    def off_diagonal_ele(self, x):
        # taken from: https://github.com/facebookresearch/barlowtwins/blob/main/main.py
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def forward(self, z1, z2):
        # N x D, where N is the batch size and D is output dim of projection head
        z1_norm = (z1 - torch.mean(z1, dim=0)) / torch.std(z1, dim=0)
        z2_norm = (z2 - torch.mean(z2, dim=0)) / torch.std(z2, dim=0)


        z1_norm = z1_norm[:, :, None]
        z2_norm = z2_norm[:, None, :]

        cross_corr = torch.matmul(z1_norm, z2_norm) 
        # cross_corr = torch.matmul(z1.T, z2) / self.batch_size
        # print(cross_corr.shape)

        on_diag = torch.diagonal(cross_corr).add_(-1).pow_(2).sum()
        # off_diag = self.off_diagonal_ele(cross_corr).pow_(2).sum()
        on_diag = torch.diagonal(cross_corr, dim1=-2, dim2=-1).add_(-1).pow_(2).sum(dim=1)
        # print(on_diag.shape)
        off_diag = self.off_diagonal_ele_batch(cross_corr).pow_(2).sum(dim=(1,2))

        # print(off_diag.shape)
        # print(on_diag)


        return on_diag + self.lambda_coeff * off_diag
    
class ProjectionHead(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=2048, output_dim=128):
        super().__init__()

        encoder = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        # for CIFAR10, replace the first 7x7 conv with smaller 3x3 conv and remove the first maxpool
        # encoder.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # encoder.maxpool = nn.MaxPool2d(kernel_size=1, stride=1)

        # replace classification fc layer of Resnet to obtain representations from the backbone
        encoder.fc = nn.Identity()

        self.encoder = encoder

        self.projection_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=True),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim, bias=False),
        )

    def forward(self, x):
        x = self.encoder(x)
        return self.projection_head(x)
 
 
def fn(warmup_steps, step):
    if step < warmup_steps:
        return float(step) / float(max(1, warmup_steps))
    else:
        return 1.0


def linear_warmup_decay(warmup_steps):
    return partial(fn, warmup_steps)  

class BarlowTwins(L.LightningModule):
    def __init__(
        self,
        encoder_out_dim,
        num_training_samples,
        batch_size,
        lambda_coeff=5e-3,
        z_dim=128,
        learning_rate=1e-4,
        warmup_epochs=10,
        max_epochs=200,
        out_dir = 'barlow'
    ):
        super().__init__()
        self.save_hyperparameters()

        self.encoder = ProjectionHead(input_dim=encoder_out_dim, hidden_dim=encoder_out_dim, output_dim=z_dim)
        self.loss_fn = BarlowTwinsLoss(batch_size=batch_size, lambda_coeff=lambda_coeff, z_dim=z_dim)

        self.learning_rate = learning_rate
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs

        self.train_iters_per_epoch = num_training_samples // batch_size
        self.batch_size = batch_size
        self.out_dir = out_dir

    def forward(self, x):
        # return self.encoder(x)
        return self.shared_step(x)
    

    def shared_step(self, batch):
        x1, x2 = batch

        z1 = self.encoder(x1)
        z2 = self.encoder(x2)

        return self.loss_fn(z1, z2)

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log("val_loss", loss, on_step=False, on_epoch=True)

    
    def test_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        agg_loss = loss.sum()/self.batch_size
        self.log("test_loss", agg_loss, on_step=False, on_epoch=True)
        return loss

    def test_epoch_end(self, outputs) -> None:

        out_barlow =  torch.cat(outputs).cpu().numpy()
        np.savetxt(self.out_dir +'_barlow_out.csv', out_barlow)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        warmup_steps = self.train_iters_per_epoch * self.warmup_epochs

        scheduler = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                linear_warmup_decay(warmup_steps),
            ),
            "interval": "step",
            "frequency": 1,
        }

        return [optimizer], [scheduler]