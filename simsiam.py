from argparse import ArgumentParser
from copy import deepcopy
from typing import Any, Dict, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule, Trainer
from torch import Tensor
from torchvision.models import resnet18, ResNet18_Weights

from utils import MLP, SiameseArm
from optimizer_utils import LinearWarmupCosineAnnealingLR
import numpy as np

class SimSiam(LightningModule):
    """PyTorch Lightning implementation of Exploring Simple Siamese Representation Learning (SimSiam_)_

    Paper authors: Xinlei Chen, Kaiming He.

    Args:
        learning_rate (float, optional): optimizer leaning rate. Defaults to 0.05.
        weight_decay (float, optional): optimizer weight decay. Defaults to 1e-4.
        momentum (float, optional): optimizer momentum. Defaults to 0.9.
        warmup_epochs (int, optional): number of epochs for scheduler warmup. Defaults to 10.
        max_epochs (int, optional): maximum number of epochs for scheduler. Defaults to 100.
        base_encoder (Union[str, nn.Module], optional): base encoder architecture. Defaults to "resnet50".
        encoder_out_dim (int, optional): base encoder output dimension. Defaults to 2048.
        projector_hidden_dim (int, optional): projector MLP hidden dimension. Defaults to 2048.
        projector_out_dim (int, optional): project MLP output dimension. Defaults to 2048.
        predictor_hidden_dim (int, optional): predictor MLP hidden dimension. Defaults to 512.
        exclude_bn_bias (bool, optional): option to exclude batchnorm and bias terms from weight decay.
            Defaults to False.

    Model implemented by:
        - `Zvi Lapp <https://github.com/zlapp>`_

    Example::

        model = SimSiam()

        dm = CIFAR10DataModule(num_workers=0)
        dm.train_transforms = SimCLRTrainDataTransform(32)
        dm.val_transforms = SimCLREvalDataTransform(32)

        trainer = Trainer()
        trainer.fit(model, datamodule=dm)

    CLI command::

        # cifar10
        python simsiam_module.py --gpus 1

        # imagenet
        python simsiam_module.py
            --gpus 8
            --dataset imagenet2012
            --meta_dir /path/to/folder/with/meta.bin/
            --batch_size 32

    .. _SimSiam: https://arxiv.org/pdf/2011.10566v1.pdf
    """

    def __init__(
        self,
        learning_rate: float = 0.05,
        weight_decay: float = 1e-4,
        momentum: float = 0.9,
        warmup_epochs: int = 10,
        max_epochs: int = 100,
        base_encoder: nn.Module = resnet18,
        encoder_out_dim: int = 2048,
        projector_hidden_dim: int = 2048,
        projector_out_dim: int = 2048,
        predictor_hidden_dim: int = 512,
        exclude_bn_bias: bool = False,
        out_dir = 'simsiam_',
        **kwargs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore="base_encoder")

        self.online_network = SiameseArm(base_encoder, encoder_out_dim, projector_hidden_dim, projector_out_dim)
        self.target_network = deepcopy(self.online_network)
        self.predictor = MLP(projector_out_dim, predictor_hidden_dim, projector_out_dim)
        self.out_dir = out_dir

    def forward(self, x: Tensor) -> Tensor:
        """Returns encoded representation of a view."""
        return self.online_network.encode(x)

    def training_step(self, batch: Any, batch_idx: int) -> Tensor:
        """Complete training loop."""
        return self._shared_step(batch, batch_idx, "train")

    def validation_step(self, batch: Any, batch_idx: int) -> Tensor:
        """Complete validation loop."""
        return self._shared_step(batch, batch_idx, "val")

    def _shared_step(self, batch: Any, batch_idx: int, step: str) -> Tensor:
        """Shared evaluation step for training and validation loops."""
        imgs = batch
        img1, img2 = imgs

        # Calculate similarity loss in each direction
        loss_12 = self.calculate_loss(img1, img2)
        loss_21 = self.calculate_loss(img2, img1)

        # Calculate total loss
        total_loss = loss_12 + loss_21

        # Log loss
        if step == "train":
            self.log_dict({"train_loss_12": loss_12, "train_loss_21": loss_21, "train_loss": total_loss})
        elif step == "val":
            self.log_dict({"val_loss_12": loss_12, "val_loss_21": loss_21, "val_loss": total_loss})
        else:
            raise ValueError(f"Step '{step}' is invalid. Must be 'train' or 'val'.")

        return total_loss

    def calculate_loss_vector(self, v_online: Tensor, v_target: Tensor) -> Tensor:
        """Calculates similarity loss between the online network prediction of target network projection.

        Args:
            v_online (Tensor): Online network view
            v_target (Tensor): Target network view
        """
        _, z1 = self.online_network(v_online)
        h1 = self.predictor(z1)
        with torch.no_grad():
            _, z2 = self.target_network(v_target)
        loss_vector = -0.5 * F.cosine_similarity(h1, z2)
        return loss_vector
    
    def calculate_loss(self, v_online: Tensor, v_target: Tensor) -> Tensor:
        """Calculates similarity loss between the online network prediction of target network projection.

        Args:
            v_online (Tensor): Online network view
            v_target (Tensor): Target network view
        """
        _, z1 = self.online_network(v_online)
        h1 = self.predictor(z1)
        with torch.no_grad():
            _, z2 = self.target_network(v_target)
        loss = -0.5 * F.cosine_similarity(h1, z2).mean()
        return loss
    
    def test_step(self, batch, batch_idx):
        img1, img2 = batch

        # Calculate similarity loss in each direction
        loss_12 = self.calculate_loss_vector(img1, img2)
        loss_21 = self.calculate_loss_vector(img2, img1)

        # Calculate total loss
        total_loss = loss_12 + loss_21

        return total_loss

    def test_epoch_end(self, outputs) -> None:

        out_barlow =  torch.cat(outputs).cpu().numpy()
        np.savetxt(self.out_dir+'_simsiam_out.csv', out_barlow)


    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        if self.hparams.exclude_bn_bias:
            params = self.exclude_from_weight_decay(self.named_parameters(), weight_decay=self.hparams.weight_decay)
        else:
            params = self.parameters()

        optimizer = torch.optim.SGD(
            params,
            lr=self.hparams.learning_rate,
            momentum=self.hparams.momentum,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer, warmup_epochs=self.hparams.warmup_epochs, max_epochs=self.hparams.max_epochs
        )

        return [optimizer], [scheduler]

    @staticmethod
    def exclude_from_weight_decay(named_params, weight_decay, skip_list=("bias", "bn")) -> List[Dict]:
        """Exclude parameters from weight decay."""
        params = []
        excluded_params = []

        for name, param in named_params:
            if not param.requires_grad:
                continue
            elif param.ndim == 1 or name in skip_list:
                excluded_params.append(param)
            else:
                params.append(param)

        return [
            {"params": params, "weight_decay": weight_decay},
            {"params": excluded_params, "weight_decay": 0.0},
        ]

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        args = parser.parse_args([])

        if "max_epochs" in args:
            parser.set_defaults(max_epochs=100)
        else:
            parser.add_argument("--max_epochs", type=int, default=100)

        parser.add_argument("--learning_rate", default=0.05, type=float, help="base learning rate")
        parser.add_argument("--weight_decay", default=1e-4, type=float, help="weight decay")
        parser.add_argument("--momentum", default=0.9, type=float, help="momentum")
        parser.add_argument("--base_encoder", default="resnet50", type=str, help="encoder backbone")
        parser.add_argument("--warmup_epochs", default=10, type=int, help="number of warmup epochs")

        return parser
