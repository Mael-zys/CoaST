import os
import logging
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from parser_train import parser_, relative_path_to_absolute_path
import cv2

from tqdm import tqdm
from data import create_dataset
from models import adaptation_modelv2
from utils import fliplr
from models.cnsn import instance_norm_mix, calc_ins_mean_std
def test(opt, logger):
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    np.random.seed(opt.seed)
    random.seed(opt.seed)
    ## create dataset
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    datasets = create_dataset(opt, logger) 

    if opt.model_name == 'deeplabv2':
        checkpoint = torch.load(opt.resume_path)['ResNet101']["model_state"]
        model = adaptation_modelv2.CustomModel(opt, logger)
        model.BaseNet.load_state_dict(checkpoint)

    validation(model, logger, datasets, device, opt)

def validation(model, logger, datasets, device, opt):
    _k = -1
    model.eval(logger=logger)
    torch.cuda.empty_cache()
    with torch.no_grad():
        validate(datasets.target_valid_loader_list, device, model, opt)

def label2rgb(func, label):
    rgbs = []
    for k in range(label.shape[0]):
        rgb = func(label[k, 0].cpu().numpy())
        rgbs.append(torch.from_numpy(rgb).permute(2, 0, 1))
    rgbs = torch.stack(rgbs, dim=0).float()
    return rgbs

def validate(valid_loader_list, device, model, opt):
    
    sm = torch.nn.Softmax(dim=1)
    
    model.eval()
   
    
    num_target = len(opt.tgt_dataset_list)
    with torch.no_grad():
        data_target_list = []
        
        for i in range(model.num_target):
            ori_LP = os.path.join(opt.root, 'Code/ProDA', opt.save_path, opt.name, opt.tgt_dataset_list[i])
            if not os.path.exists(ori_LP):
                os.makedirs(ori_LP)

            valid_loader = valid_loader_list[i]
            for data_i in tqdm(valid_loader):
                model.domain_id = i
                target_image = data_i['img'].to(device)

                filename = data_i['img_path']
                label = data_i['label'].to(device)
                
                out_list = model.BaseNet_DP(target_image, [num_target])

                out_agno = out_list[0]
                
        
                out_agno['out'] = F.interpolate(out_agno['out'], size=target_image.shape[2:], mode='bilinear', align_corners=True)
                
                confidence_agno, pseudo_agno = out_agno['out'].max(1, keepdim=True)
                pseudo_rgb_agno = label2rgb(valid_loader_list[i].dataset.decode_segmap, pseudo_agno).float() * 255
                
                gt_rgb = label2rgb(valid_loader_list[i].dataset.decode_segmap, label.unsqueeze(1)).float() * 255
                gt_rgb = gt_rgb * (label.unsqueeze(1).cpu().numpy() !=250)
                target_image *= 255
                for k in range(target_image.shape[0]):
                    name = os.path.basename(filename[k])
                    Image.fromarray(pseudo_rgb_agno[k].permute(1,2,0).cpu().numpy().astype(np.uint8)).save(os.path.join(ori_LP, name[:-4] + '_pre.png'))
                    Image.fromarray(target_image[k].permute(1,2,0).cpu().numpy().astype(np.uint8)).save(os.path.join(ori_LP, name[:-4] + '_img.png'))
                    Image.fromarray(gt_rgb[k].permute(1,2,0).cpu().numpy().astype(np.uint8)).save(os.path.join(ori_LP, name[:-4] + '_gt.png'))
                    

def get_logger(logdir):
    logger = logging.getLogger('ptsemseg')
    file_path = os.path.join(logdir, 'run.log')
    hdlr = logging.FileHandler(file_path)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr) 
    logger.setLevel(logging.INFO)
    return logger

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument('--save_path', type=str, default='./Pseudo', help='pseudo label update thred')
    parser.add_argument('--soft', action='store_true', help='save soft pseudo label')

    parser = parser_(parser)
    opt = parser.parse_args()

    opt = relative_path_to_absolute_path(opt)
    opt.logdir = opt.logdir.replace(opt.name, 'debug')
    opt.noaug = True
    opt.noshuffle = True

    print('RUNDIR: {}'.format(opt.logdir))
    if not os.path.exists(opt.logdir):
        os.makedirs(opt.logdir)

    logger = get_logger(opt.logdir)

    test(opt, logger)