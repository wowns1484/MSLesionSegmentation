import glob
import os
from re import split

from torch._C import dtype
from lib import augment3D, utils
from torch.utils.data import Dataset
import torch

import torchio as tio

from tqdm.contrib import tzip

from tqdm.notebook import tqdm, trange

import nibabel as nib
import numpy as np

from lib.medloaders import medical_image_process as img_loader
from lib.medloaders.medical_loader_utils import create_sub_volumes, find_random_crop_dim
from lib.medloaders.medical_loader_utils import get_viz_set
from lib.medloaders.medical_image_process import crop_img, normalize_intensity, medical_image_transform

from sklearn.model_selection import train_test_split

class MICCAI2008MSLESIONS(Dataset):

    def __init__(self, train_mode, dataset_path, classes=1, crop_dim=(144,144,144),sample_per_image=1,split=1.0, is_normalization=False, transform=None):
        
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

        self.train_mode = train_mode;
        self.dataset_path = dataset_path
        self.classes = classes
        self.full_vol_size = (181, 217, 181)
        self.crop_dim = crop_dim
        self.threshold = 0.1
        self.is_normalization = is_normalization
        self.sample_per_image = sample_per_image

        self.transform = transform

        self.list = []

        self.flairs = []
        self.mprages = []
        self.pdws = []
        self.t2ws = []
        self.masks = []

        self.save_name = f'{self.dataset_path}/miccai 2008 ms lesions-classes-{str(self.classes)}-list-{self.train_mode}-sampler per image-{self.sample_per_image}-{str(crop_dim[0])}x{str(crop_dim[1])}x{str(crop_dim[2])}'
        self.is_cuda = torch.cuda.is_available()

        if self.train_mode == 'train' or self.train_mode == 'val':
            # 전처리된 데이터가 존재하는지 확인.
            if os.path.exists(self.save_name):
                flair_path = sorted(glob.glob(f'{self.dataset_path}/training/*/*/*_flair_pp.nii.gz'))[0]
                self.affine = img_loader.load_affine_matrix(flair_path)
                self.list = utils.load_list(self.save_name)
                return 

            self.sub_vol_path = f'{self.dataset_path}/generated/miccai 2008 ms lesions-classes-{str(self.classes)}-list-{self.train_mode}-sampler per image-{self.sample_per_image}-{str(crop_dim[0])}x{str(crop_dim[1])}x{str(crop_dim[2])}/'
            utils.make_dirs(self.sub_vol_path)

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

            paths = enumerate(
                    tqdm(
                        zip(flair_paths,mprage_paths,pdw_paths,t2w_paths,mask_paths), 
                        desc=f"{self.train_mode} : flair, mprage, t2w, pdw and label .nii.gz to .pt", 
                        total=len(flair_paths)
                        )
                    )
    
            for idx,path in paths:
                
                flair_path = path[0]
                mprage_path = path[1]
                pdw_path = path[2]
                t2w_path = path[3]
                mask_path = path[4]

                for j in trange(sample_per_image, leave=False,desc="sampling"):

                    subject = tio.Subject(
                        flair = tio.ScalarImage(flair_path),
                        mprage = tio.ScalarImage(mprage_path),
                        pdw = tio.ScalarImage(pdw_path),
                        t2w = tio.ScalarImage(t2w_path),
                        mask = tio.LabelMap(mask_path)
                    )

                    if self.transform != None:
                        subject = self.transform(subject)

                    # 8. save file
                    flair_filename = f'{self.sub_vol_path}id_s_{str(idx)}_{str(j)}_flair.pt'
                    mprage_filename = f'{self.sub_vol_path}id_s_{str(idx)}_{str(j)}_mprage.pt'
                    pdw_filename = f'{self.sub_vol_path}id_s_{str(idx)}_{str(j)}_pdw.pt'
                    t2w_filename = f'{self.sub_vol_path}id_s_{str(idx)}_{str(j)}_t2w.pt'
                    mask_filename = f'{self.sub_vol_path}id_s_{str(idx)}_{str(j)}_mask.pt'

                    torch.save(subject.flair.data,flair_filename )
                    torch.save(subject.mprage.data,mprage_filename )
                    torch.save(subject.pdw.data,pdw_filename )
                    torch.save(subject.t2w.data,t2w_filename )
                    torch.save(subject.mask.data,mask_filename )

                    # 9. append list
                    self.list.append(tuple([flair_filename,
                                            mprage_filename,
                                            pdw_filename,
                                            t2w_filename,
                                            mask_filename]))

            # 10. save file path lists
            utils.save_list(self.save_name, self.list) 

        else:
            self.flair_paths = sorted(glob.glob(f'{self.dataset_path}/testdata_website/*/*/*_flair_pp.nii.gz'))
            self.mprage_paths = sorted(glob.glob(f'{self.dataset_path}/testdata_website/*/*/*_mprage_pp.nii.gz'))
            self.pdw_paths = sorted(glob.glob(f'{self.dataset_path}/testdata_website/*/*/*_pd_pp.nii.gz'))
            self.t2w_paths = sorted(glob.glob(f'{self.dataset_path}/testdata_website/*/*/*_t2_pp.nii.gz'))
            
            # 전처리된 데이터가 존재하는지 확인.
            if os.path.exists(self.save_name):
                flair_path = sorted(glob.glob(f'{self.dataset_path}/testdata_website/*/*/*_flair_pp.nii.gz'))[0]
                self.affine = img_loader.load_affine_matrix(flair_path)
                self.list = utils.load_list(self.save_name)
                return 

            self.sub_vol_path = f'{self.dataset_path}/generated/miccai 2008 ms lesions-classes-{str(self.classes)}-list-{self.train_mode}-sampler per image-{self.sample_per_image}-{str(crop_dim[0])}x{str(crop_dim[1])}x{str(crop_dim[2])}/'
            utils.make_dirs(self.sub_vol_path)

            paths = enumerate(
                        tqdm(
                            zip(self.flair_paths,self.mprage_paths,self.pdw_paths,self.t2w_paths), 
                            desc=f"{self.train_mode} : flair, mprage, t2w, pdw and label .nii.gz to .pt", 
                            total=len(self.flair_paths)
                            )
                        )
        
            for idx,path in paths:
                for j in trange(sample_per_image, leave=False,desc="sampling"):
                    subject = tio.Subject(
                        flair = tio.ScalarImage(path[0]),
                        mprage = tio.ScalarImage(path[1]),
                        pdw = tio.ScalarImage(path[2]),
                        t2w = tio.ScalarImage(path[3]),
                    )

                    if self.transform != None:
                        subject = self.transform(subject)

                    # 8. save file
                    flair_filename = f'{self.sub_vol_path}id_s_{str(idx)}_{str(j)}_flair.pt'
                    mprage_filename = f'{self.sub_vol_path}id_s_{str(idx)}_{str(j)}_mprage.pt'
                    pdw_filename = f'{self.sub_vol_path}id_s_{str(idx)}_{str(j)}_pdw.pt'
                    t2w_filename = f'{self.sub_vol_path}id_s_{str(idx)}_{str(j)}_t2w.pt'

                    torch.save(subject.flair.data,flair_filename )
                    torch.save(subject.mprage.data,mprage_filename )
                    torch.save(subject.pdw.data,pdw_filename )
                    torch.save(subject.t2w.data,t2w_filename )

                    # 9. append list
                    self.list.append(tuple([
                                            flair_filename,
                                            mprage_filename,
                                            pdw_filename,
                                            t2w_filename
                                            ]))

            # 10. save file path lists
            utils.save_list(self.save_name, self.list)


    def __len__(self):
        return len(self.list)

    def __getitem__(self, index):
        if self.train_mode == 'train' or self.train_mode == 'val':
            flair_path, mprage_path, pdw_path, t2w_path, mask_path = self.list[index]
            flair = torch.load(flair_path)
            # mprage = torch.load(mprage_path)
            # pdw = torch.load(pdw_path)
            # t2w = torch.load(t2w_path)
            mask = torch.load(mask_path).float()

            # return flair,mprage,pdw,t2w,mask
            return flair, mask

        elif self.train_mode == 'test':
            flair_path, mprage_path, pdw_path, t2w_path = self.list[index]

            return torch.load(flair_path), self.flair_paths[index]