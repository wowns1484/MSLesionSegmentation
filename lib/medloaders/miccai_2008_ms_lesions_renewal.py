import glob
from re import split, sub
from typing import Dict

from torch.utils.data import Dataset

import torchio as tio
from easydict import EasyDict

from tqdm.contrib import tzip

from tqdm.notebook import tqdm, trange

from lib.medloaders import medical_image_process as img_loader
from lib.medloaders.medical_loader_utils import create_sub_volumes, find_random_crop_dim
from lib.medloaders.medical_loader_utils import get_viz_set
from lib.medloaders.medical_image_process import crop_img, normalize_intensity, medical_image_transform

from sklearn.model_selection import train_test_split

CHANNELS = EasyDict({
    "FLAIR" : 0,
    "MPRAGE" : 1,
    "PDW" : 2,
    "T2W" : 3,
    "MASK" : 4
})

class MICCAI2008MSLESIONS(Dataset):
    def __init__(self, train_mode, dataset_path, channel, classes=1, crop_dim=(144,144,144), sample_per_image=1, split=1.0, is_normalization=False, transform=None):
        
        # train_mode : 'train', 'val' 두 개가 제공되며, split을 통해, train dataset과 validation dataset의 비율을 정할 수 있음.
        # dataset_path : dataset이 저장된 최상위 경로
        # claases : dataset에서 제공하는 class의 종류로 이진 분류라면 1로 설정해야함.
        # full_vol_size : dataset이 제공하는 mri 데이터의 해상도
        # crop_dim : 데이터 증강을 위한 cropping size 
        # threshold : 
        # is_normalization : 
        # sample_per_image : 데이터 증강 횟수
        # transform : 데이터 증가시 사용할 함수로, Spatial Transform : LabelMap에도 똑같이 적용하고, Intensity Transform : LabelMap에는 적용x
        # list : 증강된 데이터의 경로를 저장한 배열로 tuple을 원소로 가짐 (flair,mprage,pdw,t2w,mask)
        # flairs : 전처리된 flair 영상은 가진 배열
        # mprages : 전처리된 mprages 영상은 가진 배열
        # pdws : 전처리된 pdws 영상은 가진 배열
        # t2ws : 전처리된 t2ws 영상은 가진 배열
        # masks : 전처리된 masks 영상은 가진 배열
        # save_name : list를 저장할 이름
        # is_cuda : GPU 연산 적용 여부
        # sub_vol_path : 전처리된 데이터(flairs, mprages, pdws, t2ws, masks)들이 저장될 경로
        # affine : mri의 affine 값으로, 데이터 변형시(elastic trasnform) 필요한 행렬

        self.train_mode = train_mode
        self.dataset_path = dataset_path
        self.classes = classes
        self.full_vol_size = (181, 217, 181)
        self.crop_dim = crop_dim
        self.threshold = 0.1
        self.is_normalization = is_normalization
        self.sample_per_image = sample_per_image
        self.channel = CHANNELS.get(channel)
        self.transform = transform

        self.cache = {}
        self.list = []

        self.flairs = []
        self.mprages = []
        self.pdws = []
        self.t2ws = []
        self.masks = []

        if self.train_mode == 'train' or self.train_mode == 'val':
            # MRI 데이터의 경로를 가져옴.
            flair_paths = sorted(glob.glob(f'{self.dataset_path}/training/*/*/*_flair_pp.nii.gz'))
            mprage_paths = sorted(glob.glob(f'{self.dataset_path}/training/*/*/*_mprage_pp.nii.gz'))
            pdw_paths = sorted(glob.glob(f'{self.dataset_path}/training/*/*/*_pd_pp.nii.gz'))
            t2w_paths = sorted(glob.glob(f'{self.dataset_path}/training/*/*/*_t2_pp.nii.gz'))
            mask_paths = sorted(glob.glob(f'{self.dataset_path}/training/*/*/*mask1.nii.gz'))

            # 전체 데이터에서 split의 비율 만큼 인덱스를 계산
            total_data = len(mask_paths)

            train_split_idx = int(split[0] * total_data) # 100개면 0.1 -> 10 -> 0~9가 되겠금 -1
            self.affine = img_loader.load_affine_matrix(flair_paths[0])

            # train datset과 validation dataset의 분할
            if self.train_mode == 'train':
                flair_paths = flair_paths[:train_split_idx]
                mprage_paths = mprage_paths[:train_split_idx]
                pdw_paths = pdw_paths[:train_split_idx]
                t2w_paths = t2w_paths[:train_split_idx]
                mask_paths = mask_paths[:train_split_idx]

            elif self.train_mode == 'val':
                flair_paths = flair_paths[train_split_idx:]
                mprage_paths = mprage_paths[train_split_idx:]
                pdw_paths = pdw_paths[train_split_idx:]
                t2w_paths = t2w_paths[train_split_idx:]
                mask_paths = mask_paths[train_split_idx:]           

            for flair_path, mprage_path, pdw_path, t2w_path, mask_path in zip(flair_paths, mprage_paths, pdw_paths, t2w_paths, mask_paths):
                # 9. append list
                for i in range(self.sample_per_image):
                    self.list.append(tuple([
                                            flair_path,
                                            mprage_path,
                                            pdw_path,
                                            t2w_path,
                                            mask_path
                                            ]))

        elif self.train_mode == 'test':
            self.flair_paths = sorted(glob.glob(f'{self.dataset_path}/testdata_website/*/*/*_flair_pp.nii'))
            self.mprage_paths = sorted(glob.glob(f'{self.dataset_path}/testdata_website/*/*/*_mprage_pp.nii'))
            self.pdw_paths = sorted(glob.glob(f'{self.dataset_path}/testdata_website/*/*/*_pd_pp.nii'))
            self.t2w_paths = sorted(glob.glob(f'{self.dataset_path}/testdata_website/*/*/*_t2_pp.nii'))

            self.test_channel_paths = {}
            self.test_channel_paths[CHANNELS.get("FLAIR")] = self.flair_paths
            self.test_channel_paths[CHANNELS.get("MPRAGE")] = self.mprage_paths
            self.test_channel_paths[CHANNELS.get("PDW")] = self.pdw_paths
            self.test_channel_paths[CHANNELS.get("T2W")] = self.t2w_paths

            for flair_path, mprage_path, pdw_path, t2w_path in zip(self.flair_paths, self.mprage_paths, self.pdw_paths, self.t2w_paths):
                # 9. append list
                for i in range(self.sample_per_image):
                    self.list.append(tuple([
                                            flair_path,
                                            mprage_path,
                                            pdw_path,
                                            t2w_path
                                            ]))

    def __len__(self):
        return len(self.list)

    def __getitem__(self, index):
        subject = self.get_item(index)

        if self.train_mode == 'train' or self.train_mode == 'val':
            if self.transform != None:
                subject = self.transform(subject)

            channel = subject.channel.data
            mask = subject.mask.data

            return channel, mask
        
        elif self.train_mode == 'test':
            # 0: channel data, 1: channel data path
            return subject[0], subject[1]
    
    def get_item(self, index):
        if index in self.cache.keys():
            return self.cache[index]
        
        data = self.load_item(index)
        self.cache[index] = data

        return data

    def load_item(self, index):
        channel_path = self.list[index][self.channel]

        if self.train_mode == 'train' or self.train_mode == 'val':
            mask_path = self.list[index][CHANNELS.get("MASK")]
            
            subject = tio.Subject(
                channel = tio.ScalarImage(channel_path),
                mask = tio.LabelMap(mask_path)
            )

            return subject

        elif self.train_mode == 'test':
            channel = tio.ScalarImage(channel_path)
            
            return channel, self.test_channel_paths[self.channel][index]