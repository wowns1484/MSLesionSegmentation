from torch.utils.data import Dataset
import torch
import os

from torchio.data.image import ScalarImage
from lib import utils
import glob
from tqdm.notebook import tqdm, trange

import torchio as tio

class CORNEL_MS_LESIONS(Dataset):

    def __init__(self, train_mode, dataset_path, crop_dim,classes=1, sample_per_image=1, split=(), transform=None):
        self.train_mode = train_mode
        self.dataset_path = dataset_path
        self.classes = classes
        self.full_vol_size = (144,144,144) # 임시 값

        self.sample_per_image = sample_per_image
        self.transform = transform

        self.crop_dim = crop_dim

        # 전처리와 데이터 증강이 된 데이터셋이 저장된 배열
        self.list = []

        self.t1s = []
        self.t2s = []
        self.flairs = []
        self.masks = []

        self.save_name = f'{self.dataset_path}/cornel_ms_lesions-classes-{str(self.classes)}-list-{self.train_mode}-sampler per image-{self.sample_per_image}-{str(crop_dim[0])}x{str(crop_dim[1])}x{str(crop_dim[2])}'
        self.is_cuda = torch.cuda.is_available()

        # 전처리된 데이터 로드
        if os.path.exists(self.save_name):
            self.list = utils.load_list(self.save_name)
            return
        
        self.sub_vol_path = f'{self.dataset_path}/generated/cornel_ms_lesions-classes-{str(self.classes)}-list-{self.train_mode}-sampler per image-{self.sample_per_image}-{str(crop_dim[0])}x{str(crop_dim[1])}x{str(crop_dim[2])}/'
        utils.make_dirs(self.sub_vol_path)

        # MRI 데이터 로드
        flair_paths = sorted(glob.glob(f'{self.dataset_path}/*/*_T2FLAIR_to_MAG.nii.gz'))
        t2_paths = sorted(glob.glob(f'{self.dataset_path}/*/*_T2_to_MAG.nii.gz'))
        t1_paths = sorted(glob.glob(f'{self.dataset_path}/*/*_T1_to_MAG.nii.gz'))
        mask_paths = sorted(glob.glob(f'{self.dataset_path}/*/*_T2FLAIR_to_MAG_ROI.nii.gz'))

        # 
        total_data = len(mask_paths)

        train_split_idx = int(sum(split[:1]) * total_data) # 100개면 0.1 -> 10 -> 0~9가 되겠금 -1
        val_split_idx = int(sum(split[:2]) * total_data)

        # 훈련 데이터와 검증 데이터의 분할
        if self.train_mode == 'train':
            flair_paths  = flair_paths[:train_split_idx]
            t2_paths = t2_paths[:train_split_idx]
            t1_paths = t1_paths[:train_split_idx]
            mask_paths = mask_paths[:train_split_idx]
        
        elif self.train_mode == 'val':
            flair_paths  = flair_paths[train_split_idx:val_split_idx]
            t2_paths = t2_paths[train_split_idx:val_split_idx]
            t1_paths = t1_paths[train_split_idx:val_split_idx]
            mask_paths = mask_paths[train_split_idx:val_split_idx]

        elif self.train_mode == 'test':
            flair_paths  = flair_paths[val_split_idx:]
            t2_paths = t2_paths[val_split_idx:]
            t1_paths = t1_paths[val_split_idx:]
            mask_paths = mask_paths[val_split_idx:]

        print(total_data, train_split_idx, val_split_idx)

        paths = enumerate(tqdm(zip(flair_paths,t2_paths,t1_paths,mask_paths), 
                                desc=f"{self.train_mode} : flair, t2, t1, and label .nii.gz to .pt", 
                                total=len(flair_paths),))
                                
        for idx, path in paths:
            flair_path = path[0]
            t2_path = path[1]
            t1_path = path[2]
            mask_path = path[3]

            for j in trange(self.sample_per_image, leave=False,desc="sampling"):
                
                subject = tio.Subject(
                    flair = tio.ScalarImage(flair_path),
                    t2 = tio.ScalarImage(t2_path),
                    t1 = tio.ScalarImage(t1_path),
                    mask = tio.LabelMap(mask_path)
                )

                if self.transform != None:
                    subject = self.transform(subject)
                
                 # 8. save file
                flair_filename = f'{self.sub_vol_path}id_s_{str(idx)}_{str(j)}_flair.pt'
                t2_filename = f'{self.sub_vol_path}id_s_{str(idx)}_{str(j)}_t2.pt'
                t1_filename = f'{self.sub_vol_path}id_s_{str(idx)}_{str(j)}_t1.pt'
                mask_filename = f'{self.sub_vol_path}id_s_{str(idx)}_{str(j)}_mask.pt'

                torch.save(subject.flair.data,flair_filename )
                torch.save(subject.t2.data,t2_filename )
                torch.save(subject.t1.data,t1_filename )
                torch.save(subject.mask.data,mask_filename )
            
                # 9. append list
                self.list.append(tuple([flair_filename,
                                            t2_filename,
                                            t1_filename,
                                            mask_filename]))

        utils.save_list(self.save_name, self.list)
    
    
    def __len__(self):
        return len(self.list);

    def __getitem__(self, index):
        flair_path,t2_path,t1_path, mask_path = self.list[index]

        flair = torch.load(flair_path)
        t2 = torch.load(t2_path)
        t1 = torch.load(t1_path)
        mask = torch.load(mask_path).float()

        return flair,mask