from pytorch_lightning import Trainer
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor


from dataset import MIMIC_CXR_Dataset
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule

from simsiam import SimSiam
from simclr import SimCLRTrainTransforms, SimCLRTestTransforms


from torchvision.models import resnet18, ResNet18_Weights


ROOT = '/home/asim.ukaye/ml_proj/mimic_cxr_pa_resized/'
EPOCHS = 1000
BATCH_SIZE = 128
NUM_WORKERS = 24

CKPT_LOAD = False
CKPT_PATH ='/home/asim.ukaye/ml_proj/simclr_cxr/lightning_logs/version_32/checkpoints/last.ckpt'

seed_everything(1256)

transforms = SimCLRTrainTransforms(224)
train_set = MIMIC_CXR_Dataset(ROOT, split='train',augmentations=transforms)
val_set = MIMIC_CXR_Dataset(ROOT, split='val', augmentations=transforms)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)


base_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

model_pl = SimSiam(base_encoder=base_model, encoder_out_dim=512, max_epochs=EPOCHS)

checkpoint_callback = ModelCheckpoint(every_n_epochs=50, save_top_k=-1, save_last=True)
trainer = Trainer(callbacks=[checkpoint_callback, LearningRateMonitor("epoch")], accelerator="auto", devices=1,  max_epochs=EPOCHS)


# trainer.fit(model_pl, datamodule=dm)
trainer.fit(model_pl, train_loader, val_loader, ckpt_path=CKPT_PATH)


# def cli_main():
#     from pl_bolts.callbacks.ssl_online import SSLOnlineEvaluator
#     from pl_bolts.datamodules import CIFAR10DataModule, ImagenetDataModule, STL10DataModule
#     from pl_bolts.models.self_supervised.simclr import SimCLREvalDataTransform, SimCLRTrainDataTransform

#     seed_everything(1234)

#     parser = ArgumentParser()

#     parser = Trainer.add_argparse_args(parser)
#     parser = SimSiam.add_model_specific_args(parser)
#     parser = CIFAR10DataModule.add_dataset_specific_args(parser)
#     parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "imagenet2012", "stl10"])

#     args = parser.parse_args()

#     # Initialize datamodule
#     if args.dataset == "cifar10":
#         dm = CIFAR10DataModule.from_argparse_args(args)
#         dm.train_transforms = SimCLRTrainDataTransform(32)
#         dm.val_transforms = SimCLREvalDataTransform(32)
#         args.num_classes = dm.num_classes
#     elif args.dataset == "stl10":
#         dm = STL10DataModule.from_argparse_args(args)
#         dm.train_dataloader = dm.train_dataloader_mixed
#         dm.val_dataloader = dm.val_dataloader_mixed

#         (c, h, w) = dm.dims
#         dm.train_transforms = SimCLRTrainDataTransform(h)
#         dm.val_transforms = SimCLREvalDataTransform(h)
#         args.num_classes = dm.num_classes
#     elif args.dataset == "imagenet2012":
#         dm = ImagenetDataModule.from_argparse_args(args, image_size=196)
#         (c, h, w) = dm.dims
#         dm.train_transforms = SimCLRTrainDataTransform(h)
#         dm.val_transforms = SimCLREvalDataTransform(h)
#         args.num_classes = dm.num_classes
#     else:
#         raise ValueError(
#             f"{args.dataset} is not a valid dataset. Dataset must be 'cifar10', 'stl10', or 'imagenet2012'."
#         )

#     # Initialize SimSiam module
#     model = SimSiam(**vars(args))

#     # Finetune in real-time
#     online_eval = SSLOnlineEvaluator(dataset=args.dataset, z_dim=2048, num_classes=dm.num_classes)

#     trainer = Trainer.from_argparse_args(args, callbacks=[online_eval])

#     trainer.fit(model, datamodule=dm)


# if __name__ == "__main__":
#     cli_main()