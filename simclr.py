# -*- coding: utf-8 -*-
"""AI Summer SimCLR Resnet18 STL10.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1LhKx8FPwn8GvLbiaOp3m2wZOPZE44tgZ

# SimCLR in STL10 with Resnet18 AI Summer tutorial

## Imports, basic utils, augmentations and Contrastive loss
"""

import torch
import numpy as np

import torch.nn as nn
import torch.nn.functional as F

from torch.multiprocessing import cpu_count
import torchvision.transforms as T


from torchvision.models import resnet18, ResNet18_Weights

import pytorch_lightning as pl

from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

def default(val, def_val):
    return def_val if val is None else val

def reproducibility(config):
    SEED = int(config.seed)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)
    if (config.cuda):
        torch.cuda.manual_seed(SEED)


def device_as(t1, t2):
    """
    Moves t1 to the device of t2
    """
    return t1.to(t2.device)

# From https://github.com/PyTorchLightning/pytorch-lightning/issues/924
def weights_update(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in checkpoint['state_dict'].items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print(f'Checkpoint {checkpoint_path} was loaded')
    return model


class SimCLRTrainTransforms:
    """
    A stochastic data augmentation module
    Transforms any given data example randomly
    resulting in two correlated views of the same example,
    denoted x ̃i and x ̃j, which we consider as a positive pair.
    """

    def __init__(self, img_size=224, s=1):
        color_jitter = T.ColorJitter(
            0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s
        )
        # 10% of the image
        blur = T.GaussianBlur((3, 3), (0.1, 2.0))

        # self.train_transform = T.Compose(
        self.train_transform = nn.Sequential(

            # [
            # T.Resize((img_size, img_size)),
            # T.ToTensor(),
            T.RandomResizedCrop(size=img_size),
            T.RandomHorizontalFlip(p=0.5),  # with 0.5 probability
            T.RandomApply([color_jitter], p=0.8),
            T.RandomApply([blur], p=0.5),
            T.RandomGrayscale(p=0.2),
            # imagenet stats
            # T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            # ]
        )

    def __call__(self, x):
        return self.train_transform(x), self.train_transform(x)

class SimCLRTestTransforms:

    def __init__(self, img_size=224):

        self.test_transform = T.Compose(
            [
                # T.Resize((img_size, img_size)),
                # T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __call__(self, x):
        return self.test_transform(x), self.test_transform(x)



class ContrastiveLoss(nn.Module):
    """
    Vanilla Contrastive loss, also called InfoNceLoss as in SimCLR paper
    """
    def __init__(self, batch_size, temperature=0.5):
        super().__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.mask = (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float()

    def calc_similarity_batch(self, a, b):
        representations = torch.cat([a, b], dim=0)
        return F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)

    def forward(self, proj_1, proj_2):
        """
        proj_1 and proj_2 are batched embeddings [batch, embedding_dim]
        where corresponding indices are pairs
        z_i, z_j in the SimCLR paper
        """
        batch_size = proj_1.shape[0]
        z_i = F.normalize(proj_1, p=2, dim=1)
        z_j = F.normalize(proj_2, p=2, dim=1)

        similarity_matrix = self.calc_similarity_batch(z_i, z_j)

        sim_ij = torch.diag(similarity_matrix, batch_size)
        sim_ji = torch.diag(similarity_matrix, -batch_size)

        positives = torch.cat([sim_ij, sim_ji], dim=0)

        nominator = torch.exp(positives / self.temperature)

        denominator = device_as(self.mask, similarity_matrix) * torch.exp(similarity_matrix / self.temperature)

        all_losses = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(all_losses) / (2 * self.batch_size)
        return loss



class AddProjection(nn.Module):
    def __init__(self, model=None, embed_dim=128, mlp_dim=512):
        super(AddProjection, self).__init__()
        embedding_size = embed_dim
        self.backbone = model
        # mlp_dim = default(mlp_dim, self.backbone.fc.in_features)

        print('Dim MLP input:',mlp_dim)
        self.backbone.fc = nn.Identity()

        # add mlp projection head
        self.projection = nn.Sequential(
            nn.Linear(in_features=mlp_dim, out_features=mlp_dim),
            nn.BatchNorm1d(mlp_dim),
            nn.ReLU(),
            nn.Linear(in_features=mlp_dim, out_features=embedding_size),
            nn.BatchNorm1d(embedding_size),
        )

    def forward(self, x, return_embedding=False):
        embedding = self.backbone(x)
        if return_embedding:
            return embedding
        return self.projection(embedding)


class SimCLR_pl(pl.LightningModule):
    def __init__(self, batch_size, embed_dim = 128, hidden_dim=512, lr=3e-4, temperature=0.5, weight_decay=1e-6, max_epochs=300):
        super().__init__()
        self.save_hyperparameters()
        
        model_base = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.model = AddProjection(model=model_base, embed_dim=embed_dim, mlp_dim=hidden_dim)
        # se
        # self.model.fc = nn.Sequential(
        #     self.convnet.fc,  # Linear(ResNet output, 4*hidden_dim)
        #     nn.ReLU(inplace=True),
        #     nn.Linear(4 * hidden_dim, hidden_dim),
        # )

        # self.loss = ContrastiveLoss(batch_size, temperature=temperature)

    # def forward(self, X):
    #     return self.model(X)

    def info_nce_loss(self, batch, mode="train"):
        imgs = batch
        imgs = torch.cat(imgs, dim=0)

        # Encode all images
        feats = self.model(imgs)
        # Calculate cosine similarity
        cos_sim = F.cosine_similarity(feats[:, None, :], feats[None, :, :], dim=-1)
        # Mask out cosine similarity to itself
        self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
        cos_sim.masked_fill_(self_mask, -9e15)
        # Find positive example -> batch_size//2 away from the original example
        pos_mask = self_mask.roll(shifts=cos_sim.shape[0] // 2, dims=0)
        # InfoNCE loss
        cos_sim = cos_sim / self.hparams.temperature
        nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
        nll = nll.mean()

        # Logging loss
        self.log(mode + "_loss", nll)
        # Get ranking position of positive example
        comb_sim = torch.cat(
            [cos_sim[pos_mask][:, None], cos_sim.masked_fill(pos_mask, -9e15)],  # First position positive example
            dim=-1,
        )
        sim_argsort = comb_sim.argsort(dim=-1, descending=True).argmin(dim=-1)
        # Logging ranking metrics
        self.log(mode + "_acc_top1", (sim_argsort == 0).float().mean())
        self.log(mode + "_acc_top5", (sim_argsort < 5).float().mean())
        self.log(mode + "_acc_mean_pos", 1 + sim_argsort.float().mean())

        return nll
    

    def training_step(self, batch, batch_idx):
        # x1, x2 = batch
        # z1 = self.model(x1)
        # z2 = self.model(x2)
        # loss = self.loss(z1, z2)
        # self.log('Contrastive_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return self.info_nce_loss(batch, mode="train")
        # return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self.info_nce_loss(batch, mode="val")
        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)

        scheduler_warmup = CosineAnnealingLR(optimizer, T_max=self.hparams.max_epochs, eta_min=self.hparams.lr / 50)

        return [optimizer], [scheduler_warmup]