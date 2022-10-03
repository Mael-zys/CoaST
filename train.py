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
def train(opt, logger, opt_pl):
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    np.random.seed(opt.seed)
    random.seed(opt.seed)
    cudnn.benchmark = True
    cudnn.enabled = True
    ## create dataset
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    
    if opt.stage == 'stage1':
        print('create datasets to generate pseudo labels')
        datasets_pl = create_dataset(opt_pl, logger)

    if opt.model_name == 'deeplabv2':
        model = adaptation_modelv2.CustomModel(opt, logger)

    # Setup Metrics
    running_metrics_val_list = []
    time_meter = averageMeter()
    for i in range(len(opt.tgt_dataset_list)):
        running_metrics_val_list.append(runningScore(opt.n_class))

    if opt.stage == 'stage1':
        generate_pl(model, logger, datasets_pl, device, opt_pl)


    datasets = create_dataset(opt, logger)


    # begin training
    save_path = os.path.join(opt.logdir,"from_{}_to_{}_on_{}_current_model.pkl".format(opt.src_dataset, len(opt.tgt_dataset_list), opt.model_name))
    
    model.iter = 0
    start_epoch = 0

    for i in range(opt.train_iters):
        if opt.stage == 'stage1' and opt.rectify and (i+1) % 10000 == 0 and i != opt.train_iters-1:
            generate_pl(model, logger, datasets_pl, device, opt_pl)

        source_data = datasets.source_train_loader.next()

        data_target_list = []
        model.iter += 1
        # i = model.iter
        for i_target in range(len(opt.tgt_dataset_list)):
            data_target_list.append(datasets.target_train_loader_list[i_target].next())

        start_ts = time.time()

        model.train(logger=logger)
        if opt.freeze_bn:
            model.freeze_bn_apply()
        model.optimizer_zerograd()

       
        if opt.stage == 'warm_up':
            loss_GTA, loss_G, loss_D = model.step_adv(source_data, data_target_list, device)
        else:
            loss, loss_CTS, loss_kd, loss_CTS_transfer = model.step(source_data, data_target_list, device)

        time_meter.update(time.time() - start_ts)

        #print(i)
        if (i + 1) % opt.print_interval == 0:
            if opt.stage == 'warm_up':
                fmt_str = "Iter [{:d}/{:d}]  loss_GTA: {:.4f}  loss_G: {:.4f}  loss_D: {:.4f} Time/Image: {:.4f}"
                print_str = fmt_str.format(i + 1, opt.train_iters, loss_GTA, loss_G, loss_D, time_meter.avg / opt.bs)
            elif opt.stage == 'stage1':
                fmt_str = "Iter [{:d}/{:d}]  loss: {:.4f}  loss_CTS: {:.4f}  loss_kd: {:.4f}  loss_CTS_transfer: {:.4f} Time/Image: {:.4f}"
                print_str = fmt_str.format(i + 1, opt.train_iters, loss, loss_CTS, loss_kd, loss_CTS_transfer, time_meter.avg / opt.bs)
            print(print_str)
            logger.info(print_str)
            time_meter.reset()

        # evaluation
        if (i + 1) % opt.val_interval == 0:
            validation(model, logger, datasets, device, running_metrics_val_list, iters = model.iter, opt=opt)
            # logger.info('Best iou until now is {}'.format(model.best_iou))

        model.scheduler_step()


def validation(model, logger, datasets, device, running_metrics_val_list, iters, opt=None):
    iters = iters
    _k = -1
    for v in model.optimizers:
        _k += 1
        for param_group in v.param_groups:
            _learning_rate = param_group.get('lr')
        logger.info("learning rate is {} for {} net".format(_learning_rate, model.nets[_k].__class__.__name__))
     
    
    if opt.stage == 'warm_up':
        state = {}
        _k = -1
        for net in model.nets:
            _k += 1
            new_state = {
                "model_state": net.state_dict(),
            }
            state[net.__class__.__name__] = new_state
        state['iter'] = iters + 1
        save_path = os.path.join(opt.logdir,"from_{}_to_{}_on_{}_current_model.pkl".format(opt.src_dataset, len(opt.tgt_dataset_list), opt.model_name))
        torch.save(state, save_path)
    else:
        model.eval(logger=logger)
        current_mIoU = 0
        for i_target in range(len(opt.tgt_dataset_list)):
            val_datset = datasets.target_valid_loader_list[i_target]
            running_metrics_val = running_metrics_val_list[i_target]

            with torch.no_grad():

                validate(val_datset, device, model, running_metrics_val, len(opt.tgt_dataset_list))

                score, class_iou = running_metrics_val.get_scores()
                for k, v in score.items():
                    print(k, v)
                    logger.info('{}: {}'.format(k, v))

                for k, v in class_iou.items():
                    logger.info('{}: {}'.format(k, v))

                current_mIoU += score["Mean IoU : \t"]
                
                running_metrics_val.reset()


        current_mIoU /= len(opt.tgt_dataset_list)

        state = {}
        _k = -1
        for net in model.nets:
            _k += 1
            new_state = {
                "model_state": net.state_dict(),
            }
            state[net.__class__.__name__] = new_state
        state['iter'] = iters + 1
        state['best_iou'] = current_mIoU
        save_path = os.path.join(opt.logdir,"from_{}_to_{}_on_{}_current_model.pkl".format(opt.src_dataset, len(opt.tgt_dataset_list), opt.model_name))
        torch.save(state, save_path)
        logger.info('current mIoU {}'.format(current_mIoU))
        
        if current_mIoU >= model.best_iou:
            
            torch.cuda.empty_cache()
            model.best_iou = current_mIoU
            logger.info('current mIoU {}, best mIoU {}'.format(current_mIoU, model.best_iou))
            state = {}
            _k = -1
            for net in model.nets:
                _k += 1
                new_state = {
                    "model_state": net.state_dict(),             
                }
                state[net.__class__.__name__] = new_state
            state['iter'] = iters + 1
            state['best_iou'] = model.best_iou
            save_path = os.path.join(opt.logdir,"from_{}_to_{}_on_{}_best_model.pkl".format(opt.src_dataset, len(opt.tgt_dataset_list), opt.model_name))
            torch.save(state, save_path)
            # return score["Mean IoU : \t"]
        logger.info('Best iou until now is {}'.format(model.best_iou))

def validate(valid_loader, device, model, running_metrics_val, domain_id=0):
    for data_i in tqdm(valid_loader):

        images_val = data_i['img'].to(device)
        labels_val = data_i['label'].to(device)

        out = model.BaseNet_DP(images_val, [domain_id])
        out = out[0]

        outputs = F.interpolate(out['out'], size=images_val.size()[2:], mode='bilinear', align_corners=True)
        #val_loss = loss_fn(input=outputs, target=labels_val)

        pred = outputs.data.max(1)[1].cpu().numpy()
        gt = labels_val.data.cpu().numpy()
        running_metrics_val.update(gt, pred)


def generate_pl(model, logger, datasets, device, opt):
    model.eval(logger=logger)
    with torch.no_grad():
        generate(datasets.target_train_loader_list, device, model, opt)
        #validate(datasets.target_valid_loader, device, model, opt)

def generate(valid_loader_list, device, model, opt):

    num_target = len(opt.tgt_dataset_list)
    for i in range(num_target):
        ori_LP = os.path.join(opt.root, 'Code/CoaST', opt.save_path, opt.name, opt.tgt_dataset_list[i])

        if not os.path.exists(ori_LP):
            os.makedirs(ori_LP)

        sm = torch.nn.Softmax(dim=1)
        # valid_loader = valid_loader_list[i]
        for data_i in tqdm(valid_loader_list[i]):
            images_val = data_i['img'].to(device)
            labels_val = data_i['label'].to(device)
            filename = data_i['img_path']

            if opt.name == 'mtaf_test_labv2_stage2':
                out = model.BaseNet_DP(images_val, [0])[0]
            else:
                out = model.BaseNet_DP(images_val, [i])[0]

            if opt.soft:
                threshold_arg = F.softmax(out['out'], dim=1)
                for k in range(labels_val.shape[0]):
                    name = os.path.basename(filename[k])
                    np.save(os.path.join(ori_LP, name[:-4] + '.npy'), threshold_arg[k].cpu().numpy())
            else:
                confidence, pseudo = out['out'].max(1, keepdim=True)
                for k in range(labels_val.shape[0]):
                    name = os.path.basename(filename[k])
                    Image.fromarray(pseudo[k,0].cpu().numpy().astype(np.uint8)).save(os.path.join(ori_LP, name[:-4] + '.png'))
                    np.save(os.path.join(ori_LP, name[:-4] + '_conf.npy'), confidence[k, 0].cpu().numpy().astype(np.float16))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="config")
    parser = parser_(parser)
    opt = parser.parse_args()

    opt = relative_path_to_absolute_path(opt)

    print('RUNDIR: {}'.format(opt.logdir))
    if not os.path.exists(opt.logdir):
        os.makedirs(opt.logdir)

    logger = get_logger(opt.logdir)

    # only used for stage1
    opt_pl = copy.deepcopy(opt)
    opt_pl.noaug = True
    opt_pl.noshuffle = True
    opt_pl.norepeat = True
    opt_pl.soft = True
    opt_pl.no_droplast = True
    opt_pl.save_path = './Pseudo'
    opt_pl.used_save_pseudo = False
    
    temp_path = opt.path_soft.split(os.sep)
    temp_path = temp_path[:-1]
    temp_path = '/'.join(temp_path)
    temp_path += '/'+opt_pl.name
    opt_pl.path_soft = temp_path
    
    logger.info(opt)
    train(opt, logger, opt_pl)
