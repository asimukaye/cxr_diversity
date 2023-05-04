import os
import numpy as np 

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from timm.data import ImageDataset
import pandas as pd
import cv2
# from PIL import Image
from pytorch_lightning import LightningDataModule
from torchvision import transforms as T
import torch
import albumentations as A
class ToTensor():
    def __init__(self, dtype='float32'):
        self.dtype = dtype
    
    def __call__(self, x, **kwargs):
        return torch.from_numpy(x.transpose(2, 0, 1).astype(self.dtype))

   

class Prompt_Set(Dataset):
    def __init__(
        self,
        root=None,
        prompt_root=None,
        split="train",
        mode = 'og-prompt',
        augmentations=None
        ):

        super().__init__()
        if mode == 'og-prompt':
            df_data = pd.read_csv(os.path.join(root, 'og_to_prompt_loader.csv'), header=0, sep=',')
        elif mode == 'intra-prompt':
            df_data = pd.read_csv(os.path.join(root, 'intra_prompt.csv'), header=0, sep=',')
        elif mode == 'inter-prompt':
            df_data = pd.read_csv(os.path.join(root, 'inter_prompt.csv'), header=0, sep=',')


        # if split == 'train':
        #     df_final = df_data.loc[df_data['split'] == 'train']
        # elif split == 'val':
        #     df_final = df_data.loc[df_data['split'] == 'validate']
        # elif split == 'test':
        #     df_final = df_data.loc[df_data['split'] == 'test']
        # elif split == 'unlabeled':
        #     df_final = df_data.loc[df_data['split'] != 'test']
        # else:
        #     raise KeyError

        df_final = df_data
        self.mode = mode
        
        if mode != 'train' and augmentations is not None:
            raise TypeError
        
        if mode == 'og-prompt':
            df_path_1 = root + 'files/p' \
                    + df_final['subject_id'].floordiv(1000000).astype(str)\
                    + '/p' + df_final['subject_id'].astype(str)\
                    + '/s'+ df_final['study_id_x'].astype(str)\
                    + '/' + df_final['dicom_id'] +'.jpg'

            df_path_2 = df_final['folder_path']+'/'+df_final['prompt_id']+'.png'

        elif mode == 'intra-prompt':
            df_path_1 = df_final['folder_path']+'/'+df_final['p1']+'.png'
            df_path_2 = df_final['folder_path']+'/'+df_final['p2']+'.png'
        elif mode == 'inter-prompt':
            df_path_1 = df_final['folder_path']+'/'+df_final['prompt_id']+'.png'
            df_path_2 = df_final['folder_path_new']+'/'+df_final['p_new']+'.png'
        
        self.image_files_1 = df_path_1.to_list()
        self.image_files_2 = df_path_2.to_list()

        print("Data Size: ", len(self.image_files_1))
        # print(self.image_files_1[:3])

        self.preprocessing = T.Compose(
            [   T.ToTensor(),
                # T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),    
            ])
        
    def __len__(self):
        return len(self.image_files_1)

    def __getitem__(self, idx):
        # Get fname index and trace name
        # img = Image.open(self.image_files[idx])

        img1 = cv2.imread(self.image_files_1[idx])
        img2 = cv2.imread(self.image_files_2[idx])


        # img = self.preprocessing(image=img)['image']
        im1 = self.preprocessing(img1)
        im2 = self.preprocessing(img2)

        return im1, im2

    

class OOD_Loader(Dataset):
    def __init__(
        self,
        root=None,
        split="train",
        mode = 'train',
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

        self.mode = mode
    

        df_path = root+'files/p' \
                + df_final['subject_id'].floordiv(1000000).astype(str)\
                + '/p' + df_final['subject_id'].astype(str)\
                + '/s'+ df_final['study_id'].astype(str)\
                + '/' + df_final['dicom_id'] +'.jpg'

        self.image_files = df_path.to_list()

        # mean=(0.485, 0.456, 0.406),
        # std=(0.229, 0.224, 0.225),
        if mode == 'train':
            self.preprocessing = ToTensor()
        else:
            self.preprocessing = T.Compose(
            [
                T.ToTensor(),
                # T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        
        if mode == 'ood':
            self.other_dataset = ImageDataset('/home/asim.ukaye/ml_proj/imagenette2-320/train')

            print("Loaded Imagenet:", len(self.other_dataset))


    def get_second_image(self, idx):
        idx = idx % len(self.other_dataset)
        img, _ = self.other_dataset[idx]
        # print(type(img))
        # img.save()
        img = img.resize((256,256))
        return img

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Get fname index and trace name
        # img = Image.open(self.image_files[idx])
        img = cv2.imread(self.image_files[idx])

        # img = self.preprocessing(image=img)['image']
        img1 = self.preprocessing(img)


        img2 = self.get_second_image(idx)
        img2 = self.preprocessing(img2)
        outs = (img1, img2)
        
          # Transforms
        if self.augmentations is not None:
            outs = self.augmentations(outs)

        return outs

# ROOT = '/home/asim.ukaye/ml_proj/mimic_cxr_pa_resized/'
# eval_set = OOD_Loader(root=ROOT, mode='ood')

# outs = eval_set[10]

# print(outs[1].numpy().transpose(1, 2, 0).shape)
# cv2.imwrite("test.png", outs[1].numpy().transpose(1, 2, 0))
# print(outs)