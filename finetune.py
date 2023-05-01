
import pytorch_lightning as pl
import torch
from torch.optim import SGD
"""## Pretraining main logic"""

from pytorch_lightning import Trainer
import os
from pytorch_lightning.callbacks import GradientAccumulationScheduler
from pytorch_lightning.callbacks import ModelCheckpoint
from torchvision.models import  resnet18


class SimCLR_eval(pl.LightningModule):
    def __init__(self, lr, model=None, linear_eval=False):
        super().__init__()
        self.lr = lr
        self.linear_eval = linear_eval
        if self.linear_eval:
          model.eval()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(512,10),
            # torch.nn.ReLU(),
            # torch.nn.Dropout(0.1),
            # torch.nn.Linear(128, 10)
        )

        self.model = torch.nn.Sequential(
            model, self.mlp
        )
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, X):
        return self.model(X)

    def training_step(self, batch, batch_idx):
        x, y = batch
        z = self.forward(x)
        loss = self.loss(z, y)
        self.log('Cross Entropy loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        predicted = z.argmax(1)
        acc = (predicted == y).sum().item() / y.size(0)
        self.log('Train Acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        z = self.forward(x)
        loss = self.loss(z, y)
        self.log('Val CE loss', loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)

        predicted = z.argmax(1)
        acc = (predicted == y).sum().item() / y.size(0)
        self.log('Val Accuracy', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def configure_optimizers(self):
        if self.linear_eval:
          print(f"\n\n Attention! Linear evaluation \n")
          optimizer = SGD(self.mlp.parameters(), lr=self.lr, momentum=0.9)
        else:
          optimizer = SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
        return [optimizer]


class Hparams:
    def __init__(self):
        self.epochs = 100 # number of training epochs
        self.seed = 77777 # randomness seed
        self.cuda = True # use nvidia gpu
        self.img_size = 96 #image shape
        self.save = "./saved_models/" # save checkpoint
        self.gradient_accumulation_steps = 1 # gradient accumulation steps
        self.batch_size = 128
        self.lr = 1e-3 
        self.embedding_size= 128 # papers value is 128
        self.temperature = 0.5 # 0.1 or 0.5


# general stuff
available_gpus = len([torch.cuda.device(i) for i in range(torch.cuda.device_count())])
train_config = Hparams()
save_model_path = os.path.join(os.getcwd(), "saved_models/")
print('available_gpus:', available_gpus)
filename = 'SimCLR_ResNet18_finetune_'
reproducibility(train_config)
save_name = filename + '_Final.ckpt'

# load resnet backbone
backbone = models.resnet18(pretrained=False)
backbone.fc = nn.Identity()
checkpoint = torch.load('resnet18_backbone_weights.ckpt')
backbone.load_state_dict(checkpoint['model_state_dict'])
model = SimCLR_eval(train_config.lr, model=backbone, linear_eval=False)

# preprocessing and data loaders
transform_preprocess = Augment(train_config.img_size).test_transform
data_loader = get_stl_dataloader(train_config.batch_size, transform=transform_preprocess,split='train')
data_loader_test = get_stl_dataloader(train_config.batch_size, transform=transform_preprocess,split='test')


# callbacks and trainer
accumulator = GradientAccumulationScheduler(scheduling={0: train_config.gradient_accumulation_steps})

checkpoint_callback = ModelCheckpoint(filename=filename, dirpath=save_model_path,save_last=True,save_top_k=2,
                                       monitor='Val Accuracy_epoch', mode='max')

trainer = Trainer(callbacks=[checkpoint_callback,accumulator],
                  gpus=available_gpus,
                  max_epochs=train_config.epochs)

trainer.fit(model, data_loader,data_loader_test)
trainer.save_checkpoint(save_name)

"""# Finetune from Imageget pretraining"""

# load model
resnet = models.resnet18(pretrained=False)
resnet.fc = nn.Identity()
print('imagenet weights, no pretraining')
model = SimCLR_eval(train_config.lr, model=resnet, linear_eval=False)

# preprocessing and data loaders
transform_preprocess = Augment(train_config.img_size).test_transform
data_loader = get_stl_dataloader(128, transform=transform_preprocess,split='train')
data_loader_test = get_stl_dataloader(128, transform=transform_preprocess,split='test')

checkpoint_callback = ModelCheckpoint(filename=filename, dirpath=save_model_path)

trainer = Trainer(callbacks=[checkpoint_callback],
                  gpus=available_gpus,
                  max_epochs=train_config.epochs)

trainer.fit(model, data_loader, data_loader_test)
trainer.save_checkpoint(save_name)

# Commented out IPython magic to ensure Python compatibility.
# %load_ext tensorboard
# %tensorboard --logdir ./lightning_logs/ --port 6010