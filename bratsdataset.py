import os
import numpy as np
import torch
from torch.utils.data import Dataset
import glob
import nibabel as nib
import models


class BratsDataset(Dataset):

    def __init__(self, folder_dir):

        self.folder_dir = folder_dir
        self.image_files = sorted(glob.glob((os.path.join(self.folder_dir, '**/*flair.nii.gz')), recursive = True))
        self.mask_files = sorted(glob.glob((os.path.join(self.folder_dir, '**/*seg.nii.gz')), recursive= True))

    def __len__(self):

        return len(self.image_files)

    def __getitem__(self, idx):
        
        img_path = self.image_files[idx]
        mask_path = self.mask_files[idx]

        image = nib.load(img_path)
        mask = nib.load(mask_path)

        image = np.asanyarray(image.dataobj, dtype=np.float32)
        mask = np.asanyarray(mask.dataobj, dtype=np.float32)

        #padding so that the unet concat works
        image = np.pad(image, ((0, 0), (0, 0), (2, 3)), mode='constant', constant_values=0)
        mask = np.pad(mask, ((0, 0), (0, 0), (2, 3)), mode='constant', constant_values=0)

        #changing labels 1, 2 and 4 into 1
        mask = mask > 0
        mask = mask.astype(np.float32)

        #windowing and z score normalizing
        image = (image - image.mean()) / image.std()
        p1 = np.percentile(image, 1)
        p99 = np.percentile(image, 99)
        image = np.clip(image, p1, p99)

        image = np.expand_dims(image, axis=0)
        mask = np.expand_dims(mask, axis=0)

        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask)

        return image, mask