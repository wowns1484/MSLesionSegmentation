import glob
from lib import augment3D
import os

import torch
import numpy as np
from scipy.ndimage.measurements import label
from torch.utils.data import Dataset

import nibabel as nib

from tqdm.contrib import tzip

import lib.utils as utils
from lib.medloaders import medical_image_process as img_loader
from lib.medloaders.medical_loader_utils import create_sub_volumes, find_random_crop_dim
from lib.medloaders.medical_loader_utils import get_viz_set
from lib.medloaders.medical_image_process import crop_img, normalize_intensity


class MICCAI2018BrainTumor(Dataset):
    def __init__(self, args, mode, dataset_path='../datasets', classes=1, crop_dim=(144, 144, 144), split_idx=0, samples=100,
                 load=False):
        self.mode = mode
        self.root = dataset_path
        self.classes = classes
        dataset_name = "miccai_2018_brain_tumor" + str(classes)
        self.samples = samples
        self.list = []
        self.full_vol_size = (240, 240, 155) # width height slice
        self.crop_dim = crop_dim
        self.threshold = 0.1
        self.normalization = args.normalization if 'normalization' in args else None
        self.augmentation = args.augmentation if 'augmentation' in args else True
        self.list_labels = []
        self.list_mris = []
        self.full_volume = None
        self.save_name = self.root + '/miccai_2018_brain_tumor-classes-' + str(
            classes) + '-list-' + mode + '-samples-' + str(
            samples) + '.txt'

        self.cuda = args.cuda

        if self.augmentation:
            self.transform = augment3D.RandomChoice(
                transforms=[
                            augment3D.GaussianNoise(mean=0,std=0.01), 
                            augment3D.RandomFlip(),
                            augment3D.ElasticTransform()]
                , p=0.5
            )

        if load:
            ## load pre-generated data

            mri_path = sorted(glob.glob(os.path.join(self.root, 'imagesTr','*.nii.gz')))[0]
            self.list = utils.load_list(self.save_name)
            self.affine = img_loader.load_affine_matrix(mri_path)
            return

        subvol = '_vol_' + str(crop_dim[0]) + 'x' + str(crop_dim[1]) + 'x' + str(crop_dim[2])
        self.sub_vol_path = self.root + '/generated/' + mode + subvol + '/'
        utils.make_dirs(self.sub_vol_path)

        mri_paths = sorted(glob.glob(os.path.join(self.root, 'imagesTr','*.nii.gz')))
        label_paths = sorted(glob.glob(os.path.join(self.root,'labelsTr', '*.nii.gz')))


        self.affine = img_loader.load_affine_matrix(mri_paths[0])

        if mode == 'train':
            mri_paths = mri_paths[:split_idx]
            label_paths = label_paths[:split_idx]

        elif mode == 'val':
            mri_paths = mri_paths[split_idx:]
            label_paths = label_paths[split_idx:]

        for i,(mri_path,label_path) in enumerate(tzip(mri_paths,label_paths, desc=f"{mode} : mri t2 flair and label .nii.gz to .npy", total=len(mri_paths))):
            # 1. nii.gz load
            mri_nii = nib.load(mri_path)
            label_nii = nib.load(label_path)
            
            # 2. nii.gz to numpy float32
            mri_nii = np.squeeze(mri_nii.get_fdata(dtype=np.float32))
            label_nii = np.squeeze(label_nii.get_fdata(dtype=np.float32))

            # 3. 백그라운드를 제외한 모든 마스크를 통일합니다.
            """
            "labels": {
                 "0": "background",
                 "1": "edema",  부종
                 "2": "non-enhancing tumor",
                 "3": "enhancing tumour"}
             """
            label_nii[label_nii > 0] = 1.0

            # 4. MRI 영상 중 T2 Flair만 가져옵니다.
            """
            "modality": { 
                "0": "FLAIR", 
                "1": "T1w", 
                "2": "t1gd",
                "3": "T2w"
            },
            """  
            mri_nii = mri_nii[:,:,:,0]

            # 5. numpy to tensor
            mri_tensor = torch.from_numpy(mri_nii)
            label_tensor = torch.from_numpy(label_nii)

            # 6. intensity normalization
            MEAN, STD = mri_tensor.mean(), mri_tensor.std()
            MAX, MIN = mri_tensor.max(), mri_tensor.min()
            normalization='full_volume_mean'
            mri_tensor = normalize_intensity(mri_tensor, normalization=normalization, norm_values=(MEAN, STD, MAX, MIN))
            
            # 5. tiling
            crop_coordinate = find_random_crop_dim(self.full_vol_size, crop_dim)
            mri_tensor = crop_img(mri_tensor,crop_size=crop_dim, crop=crop_coordinate)
            label_tensor = crop_img(label_tensor,crop_size=crop_dim, crop=crop_coordinate)

            # 5. save file
            mri_filename = f'{self.sub_vol_path}id_s_{str(i)}_modality_0.npy'
            label_filename = f'{self.sub_vol_path}id_s_{str(i)}_modality_0_seg.npy'

            np.save(mri_filename, mri_tensor)
            np.save(label_filename, label_tensor)

            self.list.append(tuple([mri_filename,label_filename]))


        utils.save_list(self.save_name, self.list)

    def __len__(self):
        return len(self.list)

    def __getitem__(self, index):
        mri_path, label_path = self.list[index]

        image_mri , image_label = np.load(mri_path) , np.load(label_path)

        if self.mode =='train' and self.augmentation:
            [image_mri], image_label = self.transform([image_mri],image_label)
            return torch.FloatTensor(image_mri.copy()).unsqueeze(0), torch.FloatTensor(image_label.copy()).unsqueeze(0)

        elif self.mode =='val':
            return torch.FloatTensor(image_mri.copy()).unsqueeze(0), torch.FloatTensor(image_label.copy()).unsqueeze(0)

        return image_mri , image_label
