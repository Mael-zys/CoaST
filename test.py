import os
import sys
import time
import torch
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from parser_train import parser_, relative_path_to_absolute_path
import copy
from tqdm import tqdm
from data import create_dataset
from utils import get_logger
from models import adaptation_modelv2
from metrics import runningScore, averageMeter
import warnings
import torch.backends.cudnn as cudnn
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")
def test(opt, logger):
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    np.random.seed(opt.seed)
    random.seed(opt.seed)
    cudnn.benchmark = True
    cudnn.enabled = True
    ## create dataset
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    

    if opt.model_name == 'deeplabv2':
        model = adaptation_modelv2.CustomModel(opt, logger)

    # Setup Metrics
    running_metrics_val_list = []
    time_meter = averageMeter()
    for i in range(len(opt.tgt_dataset_list)):
        running_metrics_val_list.append(runningScore(opt.n_class))


    datasets = create_dataset(opt, logger)

    validation(model, logger, datasets, device, running_metrics_val_list, opt=opt)


def validation(model, logger, datasets, device, running_metrics_val_list, opt=None):
    _k = -1
    
    model.eval(logger=logger)
    
    current_mIoU = 0
    for i_target in range(len(opt.tgt_dataset_list)):
        val_datset = datasets.target_valid_loader_list[i_target]
        running_metrics_val = running_metrics_val_list[i_target]

        with torch.no_grad():
            validate(val_datset, device, model, running_metrics_val, len(opt.tgt_dataset_list))
            # validate(val_datset, device, model, running_metrics_val, i_target)
            score, class_iou = running_metrics_val.get_scores()
            for k, v in score.items():
                print(k, v)
                logger.info('{}: {}'.format(k, v))

            for k, v in class_iou.items():
                logger.info('{}: {}'.format(k, v))

            current_mIoU += score["Mean IoU : \t"]
            
            running_metrics_val.reset()

    current_mIoU /= len(opt.tgt_dataset_list)
    print('Avg mIoU {}'.format(current_mIoU))
    logger.info('Avg mIoU {}'.format(current_mIoU))
            


def validate(valid_loader, device, model, running_metrics_val, domain_id=0):
    for data_i in tqdm(valid_loader):

        images_val = data_i['img'].to(device)
        labels_val = data_i['label'].to(device)

        out = model.BaseNet_DP(images_val, [domain_id])
        out = out[0]

        outputs = F.interpolate(out['out'], size=images_val.size()[2:], mode='bilinear', align_corners=True)

        pred = outputs.data.max(1)[1].cpu().numpy()
        gt = labels_val.data.cpu().numpy()
        running_metrics_val.update(gt, pred)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="config")
    parser = parser_(parser)
    opt = parser.parse_args()

    opt = relative_path_to_absolute_path(opt)
    opt.logdir = opt.logdir.replace(opt.name, 'test')

    print('RUNDIR: {}'.format(opt.logdir))
    if not os.path.exists(opt.logdir):
        os.makedirs(opt.logdir)

    logger = get_logger(opt.logdir)

    test(opt, logger)
