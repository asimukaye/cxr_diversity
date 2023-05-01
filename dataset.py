import os
import numpy as np 

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import pandas as pd
import cv2
# from PIL import Image
from pytorch_lightning import LightningDataModule
from torchvision import transforms
import torch
import albumentations as A
class ToTensor():
    def __init__(self, dtype='float32'):
        self.dtype = dtype
    
    def __call__(self, x, **kwargs):
        return torch.from_numpy(x.transpose(2, 0, 1).astype(self.dtype))
    

class MIMIC_CXR_Dataset(Dataset):
    def __init__(
        self,
        root=None,
        split="train",
        augmentations=None

    ):
        self.augmentations = augmentations
        super().__init__()
        df_data = pd.read_csv(os.path.join(root, 'mimic-cxr-2.0.0-dataloader.csv'), header=0, sep=',')

        if split == 'train':
            df_final = df_data.loc[df_data['split'] == 'train']
        elif split == 'val':
            df_final = df_data.loc[df_data['split'] == 'validate']
        elif split == 'test':
            df_final = df_data.loc[df_data['split'] == 'test']
        elif split == 'unlabeled':
            df_final = df_data.loc[df_data['split'] != 'test']
        else:
            raise KeyError

        df_path = root+'files/p' \
                + df_final['subject_id'].floordiv(1000000).astype(str)\
                + '/p' + df_final['subject_id'].astype(str)\
                + '/s'+ df_final['study_id'].astype(str)\
                + '/' + df_final['dicom_id'] +'.jpg'

        self.image_files = df_path.to_list()

        # mean=(0.485, 0.456, 0.406),
        # std=(0.229, 0.224, 0.225),
        self.preprocessing = ToTensor()
        # self.preprocessing = A.Compose([
        #     A.Resize(256, 256, always_apply=True),
        #     # A.Normalize(mean, std, max_pixel_value=255.0, always_apply=True),
        #     A.Lambda(image=ToTensor(), always_apply=True)
        # ])
    

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Get fname index and trace name
        # img = Image.open(self.image_files[idx])
        img = cv2.imread(self.image_files[idx])

        # img = self.preprocessing(image=img)['image']
        img = self.preprocessing(img)

        # Transforms
        if self.augmentations is not None:
            outs = self.augmentations(img)

        return outs



class CXR_DataModule(LightningDataModule):
        def __init__(self, root, batch_size, num_workers = 4, mode='unlabeled', transforms=None):
            super().__init__()
            self.root = root
            self.batch_size = batch_size
            self.mode = mode
            self.transforms = transforms
            self.num_workers = num_workers
           
        def setup(self, stage):
            # make assignments here (val/train/test split)
            # called on every process in DDP
            if stage == "fit":
                if self.mode =='unlabeled':
                    self.train_set = MIMIC_CXR_Dataset(self.root, split='unlabeled', augmentations=self.transforms)
                else:
                    self.train_set = MIMIC_CXR_Dataset(self.root, split='train',augmentations=self.transfroms)

                self.val_set = MIMIC_CXR_Dataset(self.root, split='val', augmentations=self.transforms)

            if stage == 'test':
  
                self.test_split = MIMIC_CXR_Dataset(self.root, split='test', augmentations=self.transforms)


        def train_dataloader(self):
            return DataLoader(self.train_set, batch_size=self.batch_size, num_workers=self.num_workers)
        def val_dataloader(self):
            return DataLoader(self.val_set, batch_size=self.batch_size, num_workers=self.num_workers)
        def test_dataloader(self):
            return DataLoader(self.test_split, batch_size=self.batch_size, num_workers=self.num_workers)
        

# root = '/home/asim.ukaye/physionet.org/files/mimic-cxr-jpg/2.0.0/'
# dataset = MIMIC_CXR_Dataset(root, split='train')


# import cv2
# # import matplotlib.pyplot as plt

# for i in range(100):
#     cv2.imshow('CXR',dataset[i])
#     cv2.waitKey(0)
# cv2.destroyAllWindows()
# fig, ax = plt.subplots(1,2)
# img1 = dataset[0]
# img2 = dataset[1]

# ax[0].imshow(img1)
# ax[1].imshow(img2)
