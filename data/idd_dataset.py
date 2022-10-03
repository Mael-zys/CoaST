import numpy as np


import os
import torch
import numpy as np
import scipy.misc as m
from tqdm import tqdm

from torch.utils import data
from PIL import Image

from data.augmentations import *
from data.base_dataset import BaseDataset
from data.randaugment import RandAugmentMC

import random

from torchvision import transforms

import json
def json_load(file_path):
    with open(file_path, 'r') as fp:
        return json.load(fp)

def recursive_glob(rootdir=".", suffix=""):
    """Performs recursive glob with given suffix and rootdir 
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    return [
        os.path.join(looproot, filename)
        for looproot, _, filenames in os.walk(rootdir)  #os.walk: traversal all files in rootdir and its subfolders
        for filename in filenames
        if filename.endswith(suffix)
    ]


class IDD_loader(BaseDataset):
    colors = [  # [  0,   0,   0],
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [0, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
    ]

    label_colours = dict(zip(range(19), colors))

    colors_7 = [  # [  0,   0,   0],
        [128, 64, 128],
        [70, 70, 70],
        [153, 153, 153],
        [107, 142, 35],
        [0, 130, 180],
        [220, 20, 60],
        [0, 0, 142],
    ]

    label_colours_7 = dict(zip(range(7), colors_7))

    def __init__(self, opt, logger, augmentations = None, split='train', domain_id=0):
        self.opt = opt
        if opt.src_dataset == 'idd':
            self.root = opt.src_rootpath
            self.dataset_name = self.opt.src_dataset
        else:
            self.root = opt.tgt_rootpath_list[domain_id]
            self.dataset_name = self.opt.tgt_dataset_list[domain_id]
        self.domain_id = domain_id
        self.split = split
        self.augmentations = augmentations
        self.randaug = RandAugmentMC(2, 10)
        self.n_classes = opt.n_class
        self.img_size = opt.img_size
        self.files = {}
        self.paired_files = {}
        self.ignore_index = 250
        INFO_PATH_7 = opt.root + '/data/idd_list/info7class.json'
        INFO_PATH_19 = opt.root + '/data/idd_list/info19class.json'
        if self.n_classes == 7:
            self.info = json_load(INFO_PATH_7)
            self.to7 = dict(zip(range(7), range(7)))
        else:
            self.info = json_load(INFO_PATH_19)
            self.to19 = dict(zip(range(19), range(19)))
        
        self.images_base = os.path.join(self.root, "leftImg8bit", self.split)
        self.annotations_base = os.path.join(
            self.root, "gtFine", self.split
        )
        
        sorted_paths = sorted(recursive_glob(rootdir=self.images_base, suffix=".png")) #find all files from rootdir and subfolders with suffix = ".png"

        self.files = list(sorted_paths)
        if opt.train_iters is not None and 'train' in split and not opt.norepeat:
            self.files = self.files * int(np.ceil(float(opt.train_iters) / len(self.files)))
            

        self.mapping = np.array(self.info['label2train'], dtype=np.int)
        self.map_vector = np.zeros((self.mapping.shape[0],), dtype=np.uint8)
        for source_label, target_label in self.mapping:
            self.map_vector[source_label] = target_label

    def get_metadata(self, name):
        img_file = self.root / 'leftImg8bit' / self.set / name
        if self.label_folder is not '':
            label_file = self.label_folder + '/' + (name.split('/')[1])
        else:
            label_name = name.replace("leftImg8bit", "gtFine_labelids")
            label_file = self.root / 'gtFine' / self.set / label_name
        return img_file, label_file

    def __len__(self):
        """__len__"""
        return len(self.files)

    def map_labels(self, input_):
        return self.map_vector[input_.astype(np.uint8, copy=False)]

    def __getitem__(self, index):
        img_path = self.files[index].rstrip()
        lbl_path = os.path.join(
            self.annotations_base,
            img_path.split(os.sep)[-2],
            os.path.basename(img_path)[:-15] + "gtFine_labelids.png",
        )

        img = Image.open(img_path)
        lbl = Image.open(lbl_path)
        img = img.resize(self.img_size, Image.BILINEAR)
        lbl = lbl.resize(self.img_size, Image.NEAREST)
        
        img = np.array(img, dtype=np.uint8)
        lbl = np.array(lbl, dtype=np.uint8)

        lbl = self.map_labels(lbl).copy()

        img_full = img.copy().astype(np.float64)
        # img_full -= self.mean
        img_full = img_full.astype(float) / 255.0
        img_full = img_full.transpose(2, 0, 1)

        lp, lpsoft, weak_params = None, None, None
        if self.opt.src_dataset != 'idd' and self.split == 'train' and self.opt.used_save_pseudo:
            lpsoft = np.load(os.path.join(self.opt.path_soft, self.dataset_name, os.path.basename(img_path).replace('.png', '.npy')))


        input_dict = {}
        if self.augmentations!=None:

            img, lbl, lp, lpsoft, weak_params = self.augmentations(img, lbl, lp, lpsoft)
            img_strong, params = self.randaug(Image.fromarray(img))
            img_strong, _, _ = self.transform(img_strong, lbl)
            input_dict['img_strong'] = img_strong
            input_dict['params'] = params


        img, lbl_, lp = self.transform(img, lbl, lp)
                
        input_dict['img'] = img
        input_dict['img_full'] = torch.from_numpy(img_full).float()
        input_dict['label'] = lbl_
        input_dict['lp'] = lp
        input_dict['lpsoft'] = lpsoft
        input_dict['weak_params'] = weak_params  #full2weak
        input_dict['img_path'] = self.files[index]

        input_dict = {k:v for k, v in input_dict.items() if v is not None}
        return input_dict

    def transform(self, img, lbl, lp=None, check=True):
        """transform
        :param img:
        :param lbl:
        """
        # img = m.imresize(
        #     img, (self.img_size[0], self.img_size[1])
        # )  # uint8 with RGB mode
        img = np.array(img)
        # img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float64)
        # img -= self.mean
        img = img.astype(float) / 255.0
        # NHWC -> NCHW
        img = img.transpose(2, 0, 1)

        classes = np.unique(lbl)
        lbl = np.array(lbl)
        lbl = lbl.astype(float)
        # lbl = m.imresize(lbl, (self.img_size[0], self.img_size[1]), "nearest", mode="F")
        lbl = lbl.astype(int)

        if not np.all(classes == np.unique(lbl)):
            print("WARN: resizing labels yielded fewer classes")    #TODO: compare the original and processed ones

        if check and not np.all(np.unique(lbl[lbl != self.ignore_index]) < self.n_classes):   #todo: understanding the meaning 
            print("after det", classes, np.unique(lbl))
            raise ValueError("Segmentation map contained invalid class values")

        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()

        if lp is not None:
            classes = np.unique(lp)
            lp = np.array(lp)

            lp = torch.from_numpy(lp).long()

        return img, lbl, lp
    
    def decode_segmap(self, temp):
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        if self.n_classes == 19:
            for l in range(0, self.n_classes):
                r[temp == l] = self.label_colours[self.to19[l]][0]
                g[temp == l] = self.label_colours[self.to19[l]][1]
                b[temp == l] = self.label_colours[self.to19[l]][2]
        elif self.n_classes == 7:
            for l in range(0, self.n_classes):
                r[temp == l] = self.label_colours_7[self.to7[l]][0]
                g[temp == l] = self.label_colours_7[self.to7[l]][1]
                b[temp == l] = self.label_colours_7[self.to7[l]][2]
        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        return rgb