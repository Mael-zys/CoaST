import json
import warnings
from pathlib import Path

from torch.utils import data
from torchvision import transforms
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
import pdb


def json_load(file_path):
    with open(file_path, 'r') as fp:
        return json.load(fp)

class Mapillary_loader(BaseDataset):

    classes_ids = {'flat': 0,
                   'construction': 1,
                   'object': 2,
                   'nature': 3,
                   'sky': 4,
                   'human': 5,
                   'vehicle': 6,
                   'other': 250}

    classes_mappings_mapillary_to_cityscapes = {'bird': 'other',
                                                'ground animal': 'other',
                                                'curb': 'construction',
                                                'fence': 'construction',
                                                'guard rail': 'construction',
                                                'barrier': 'construction',
                                                'wall': 'construction',
                                                'bike lane': 'flat',
                                                'crosswalk - plain': 'flat',
                                                'curb cut': 'flat',
                                                'parking': 'flat',
                                                'pedestrian area': 'flat',
                                                'rail track': 'flat',
                                                'road': 'flat',
                                                'service lane': 'flat',
                                                'sidewalk': 'flat',
                                                'bridge': 'construction',
                                                'building': 'construction',
                                                'tunnel': 'construction',
                                                'person': 'human',
                                                'bicyclist': 'human',
                                                'motorcyclist': 'human',
                                                'other rider': 'human',
                                                'lane marking - crosswalk': 'flat',
                                                'lane marking - general': 'flat',
                                                'mountain': 'other',
                                                'sand': 'other',
                                                'sky': 'sky',
                                                'snow': 'other',
                                                'terrain': 'flat',
                                                'vegetation': 'nature',
                                                'water': 'other',
                                                'banner': 'other',
                                                'bench': 'other',
                                                'bike rack': 'other',
                                                'billboard': 'other',
                                                'catch basin': 'other',
                                                'cctv camera': 'other',
                                                'fire hydrant': 'other',
                                                'junction box': 'other',
                                                'mailbox': 'other',
                                                'manhole': 'other',
                                                'phone booth': 'other',
                                                'pothole': 'object',
                                                'street light': 'object',
                                                'pole': 'object',
                                                'traffic sign frame': 'object',
                                                'utility pole': 'object',
                                                'traffic light': 'object',
                                                'traffic sign (back)': 'object',
                                                'traffic sign (front)': 'object',
                                                'trash can': 'other',
                                                'bicycle': 'vehicle',
                                                'boat': 'vehicle',
                                                'bus': 'vehicle',
                                                'car': 'vehicle',
                                                'caravan': 'vehicle',
                                                'motorcycle': 'vehicle',
                                                'on rails': 'vehicle',
                                                'other vehicle': 'vehicle',
                                                'trailer': 'vehicle',
                                                'truck': 'vehicle',
                                                'wheeled slow': 'vehicle',
                                                'car mount': 'other',
                                                'ego vehicle': 'other',
                                                'unlabeled': 'other'}
    classes_ids_19 = {'road': 0,
                   'sidewalk': 1,
                   'building': 2,
                   'wall': 3,
                   'fence': 4,
                   'pole': 5,
                   'traffic light': 6,
                   'traffic sign': 7,
                   'vegetation': 8,
                   'terrain': 9,
                   'sky': 10,
                   'person': 11,
                   'rider': 12,
                   'car': 13,
                   'truck': 14,
                   'bus': 15,
                   'train': 16,
                   'motorcycle': 17,
                   'bicycle': 18,
                   'other': 250}

    classes_mappings_mapillary_to_cityscapes_19 = {'bird': 'other',
                                                'ground animal': 'other',
                                                'curb': 'other',
                                                'fence': 'fence',
                                                'guard rail': 'other',
                                                'barrier': 'other',
                                                'wall': 'wall',
                                                'bike lane': 'other',
                                                'crosswalk - plain': 'other',
                                                'curb cut': 'other',
                                                'parking': 'other',
                                                'pedestrian area': 'other',
                                                'rail track': 'other',
                                                'road': 'road',
                                                'service lane': 'other',
                                                'sidewalk': 'sidewalk',
                                                'bridge': 'other',
                                                'building': 'building',
                                                'tunnel': 'other',
                                                'person': 'person',
                                                'bicyclist': 'rider',
                                                'motorcyclist': 'rider',
                                                'other rider': 'rider',
                                                'lane marking - crosswalk': 'other',
                                                'lane marking - general': 'other',
                                                'mountain': 'other',
                                                'sand': 'other',
                                                'sky': 'sky',
                                                'snow': 'other',
                                                'terrain': 'terrain',
                                                'vegetation': 'vegetation',
                                                'water': 'other',
                                                'banner': 'other',
                                                'bench': 'other',
                                                'bike rack': 'other',
                                                'billboard': 'other',
                                                'catch basin': 'other',
                                                'cctv camera': 'other',
                                                'fire hydrant': 'other',
                                                'junction box': 'other',
                                                'mailbox': 'other',
                                                'manhole': 'other',
                                                'phone booth': 'other',
                                                'pothole': 'other',
                                                'street light': 'other',
                                                'pole': 'pole',
                                                'traffic sign frame': 'other',
                                                'utility pole': 'other',
                                                'traffic light': 'traffic light',
                                                'traffic sign (back)': 'traffic sign',
                                                'traffic sign (front)': 'traffic sign',
                                                'trash can': 'other',
                                                'bicycle': 'bicycle',
                                                'boat': 'other',
                                                'bus': 'bus',
                                                'car': 'car',
                                                'caravan': 'other',
                                                'motorcycle': 'motorcycle',
                                                'on rails': 'train',
                                                'other vehicle': 'other',
                                                'trailer': 'other',
                                                'truck': 'truck',
                                                'wheeled slow': 'other',
                                                'car mount': 'other',
                                                'ego vehicle': 'other',
                                                'unlabeled': 'other'}

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
        # root = Path(download('mapillary'))
        self.opt = opt
        if opt.src_dataset == 'mapillary':
            self.root = opt.src_rootpath
            self.dataset_name = self.opt.src_dataset
        else:
            self.root = opt.tgt_rootpath_list[domain_id]
            self.dataset_name = self.opt.tgt_dataset_list[domain_id]
        self.domain_id = domain_id
        if 'val' in split:
            split = 'validation'
        self.split = split
        self.augmentations = augmentations
        self.randaug = RandAugmentMC(2, 10)
        self.n_classes = opt.n_class
        self.img_size = opt.img_size

        self.files = {}
        self.paired_files = {}

        self.path = Path(self.root) / split
        
        self.ignore_index = 250

        self.set = split
        sorted_paths = map(lambda x: sorted((self.path / x).iterdir()),
                           ('images', 'labels', 'instances'))

        self.data_paths = list(zip(*sorted_paths))
        # if set == 'train':
        #     self.data_paths = list(zip(*sorted_paths))[:900]
        # else:
        #     self.data_paths = list(zip(*sorted_paths))

        if opt.train_iters is not None and 'train' in split and not opt.norepeat:
            self.data_paths = self.data_paths * int(np.ceil(float(opt.train_iters) / len(self.data_paths)))

        
        self.labels = json.loads((Path(self.root) / 'config.json').read_text())['labels']

        self.vector_mappings = None

        if self.n_classes == 7:
            class_mappings=self.classes_mappings_mapillary_to_cityscapes
            model_classes=self.classes_ids
            self.to7 = dict(zip(range(7), range(7)))
        else:
            class_mappings=self.classes_mappings_mapillary_to_cityscapes_19
            model_classes=self.classes_ids_19
            self.to19 = dict(zip(range(19), range(19)))

        if class_mappings is not None:
            dataset_classes = [label['readable'] for label in self.labels]
            self.vector_mappings = array_from_class_mappings(dataset_classes,
                                                            class_mappings,
                                                            model_classes)

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, index):
        image_path, labels_path, instances_path = self.data_paths[index]

        image_array = Image.open(image_path)  # 3D  #double-check if this is RGB
        labels_array = Image.open(labels_path)  # 2D
        img = image_array.resize(self.img_size, Image.BILINEAR)
        lbl = labels_array.resize(self.img_size, Image.NEAREST)

        if self.vector_mappings is not None:
            # we have to remap the labels on new classes.
            lbl = self.vector_mappings[lbl]

        img = np.array(img, dtype=np.uint8)
        lbl = np.array(lbl, dtype=np.uint8)

        img_full = img.copy().astype(np.float64)
        # img_full -= self.mean
        img_full = img_full.astype(float) / 255.0
        img_full = img_full.transpose(2, 0, 1)

        lp, lpsoft, weak_params = None, None, None
        if self.opt.src_dataset != 'mapillary' and self.split == 'train' and self.opt.used_save_pseudo:
            lpsoft = np.load(os.path.join(self.opt.path_soft, self.dataset_name, os.path.basename(str(image_path)).replace('.jpg', '.npy')))


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
        input_dict['img_path'] = str(image_path)

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
            # if not np.all(np.unique(lp[lp != self.ignore_index]) < self.n_classes):
            #     raise ValueError("lp Segmentation map contained invalid class values")
        
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


def label_mapping_mapilliary(input, mapping):
    output = np.copy(input)
    for ind,val in enumerate(mapping):
        output[input == ind] = val
    # return np.array(output, dtype=np.int64)
    return np.array(output, dtype=np.uint8)

def array_from_class_mappings(dataset_classes, class_mappings, model_classes):
    """
    :param dataset_classes: list or dict. Mapping between indexes and name of classes.
                            If using a list, it's equivalent
                            to {x: i for i, x in enumerate(dataset_classes)}
    :param class_mappings: Dictionary mapping names of the dataset to
                           names of classes of the model.
    :param model_classes:  list or dict. Same as dataset_classes,
                           but for the model classes.
    :return: A numpy array representing the mapping to be done.
    """
    # Assert all classes are different.
    assert len(model_classes) == len(set(model_classes))

    # to generate the template to fill the dictionary for class_mappings
    # uncomment this code.
    """
    for x in dataset_classes:
        print((' ' * 20) + f'\'{name}\': \'\',')
    """

    # Not case sensitive to make it easier to write.
    if isinstance(dataset_classes, list):
        dataset_classes = {x: i for i, x in enumerate(dataset_classes)}
    dataset_classes = {k.lower(): v for k, v in dataset_classes.items()}
    class_mappings = {k.lower(): v.lower() for k, v in class_mappings.items()}
    if isinstance(model_classes, list):
        model_classes = {x: i for i, x in enumerate(model_classes)}
    model_classes = {k.lower(): v for k, v in model_classes.items()}

    result = np.zeros((max(dataset_classes.values()) + 1,), dtype=np.uint8)
    for dataset_class_name, i in dataset_classes.items():
        result[i] = model_classes[class_mappings[dataset_class_name]]
    return result


def resize_with_pad(target_size, image, resize_type, fill_value=0):
    if target_size is None:
        return np.array(image)
    # find which size to fit to the target size
    target_ratio = target_size[0] / target_size[1]
    image_ratio = image.size[0] / image.size[1]

    if image_ratio > target_ratio:
        resize_ratio = target_size[0] / image.size[0]
        new_image_shape = (target_size[0], int(image.size[1] * resize_ratio))
    else:
        resize_ratio = target_size[1] / image.size[1]
        new_image_shape = (int(image.size[0] * resize_ratio), target_size[1])

    image_resized = image.resize(new_image_shape, resize_type)

    image_resized = np.array(image_resized)
    if image_resized.ndim == 2:
        image_resized = image_resized[:, :, None]
    tmp = target_size[::-1] + [image_resized.shape[2],]
    result = np.ones(tmp, image_resized.dtype) * fill_value
    assert image_resized.shape[0] <= result.shape[0]
    assert image_resized.shape[1] <= result.shape[1]
    placeholder = result[:image_resized.shape[0], :image_resized.shape[1]]
    placeholder[:] = image_resized
    return result

def pad_with_fixed_AS(target_ratio, image, fill_value=0):
    dimW = float(image.size[0])
    dimH = float(image.size[1])
    image_ratio = dimW/dimH
    if target_ratio > image_ratio:
        dimW = target_ratio*dimH
    elif target_ratio < image_ratio:
        dimH = dimW/target_ratio
    else:
        return np.array(image)
    image = np.array(image)
    result = np.ones((int(dimH), int(dimW)), image.dtype) * fill_value
    placeholder = result[:image.shape[0], :image.shape[1]]
    placeholder[:] = image
    return result