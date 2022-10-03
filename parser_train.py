import os
import json
from utils import project_root

def parser_(parser):
    parser.add_argument('--root', type=str, default=str(project_root), help='root path')
    parser.add_argument('--model_name', type=str, default='deeplabv2', help='deeplabv2')
    parser.add_argument('--name', type=str, default='gta2city', help='pretrain source model')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--bs', type=int, default=4)
    parser.add_argument('--freeze_bn', action='store_true')
    parser.add_argument('--train_iters', type=int, default=60000)
    parser.add_argument('--bn', type=str, default='sync_bn', help='sync_bn|bn|gn|adabn')
    #training
    parser.add_argument('--no_resume', action='store_true')
    parser.add_argument('--seed', type=int, default=1337, help='random seed')
    parser.add_argument('--stage', type=str, default='stage1', help='warm_up|stage1')
    #model
    parser.add_argument('--resume_path', type=str, default='pretrained/warmup/from_gta5_to_cityscapes_on_deeplab101_best_model_warmup.pkl', help='resume model path')

    #data
    parser.add_argument('--src_dataset', type=str, default='gta5', help='gta5|cityscapes|mapillary|idd')
    parser.add_argument('--tgt_dataset', dest='tgt_dataset_list', type=str, nargs='*', default=['cityscapes', 'idd'], help="cityscapes, mapillary, idd")
    parser.add_argument('--src_rootpath', type=str, default='Dataset/GTA5')
    parser.add_argument('--tgt_rootpath', dest='tgt_rootpath_list', type=str, nargs='*', default=['Dataset/Cityscapes', 'Dataset/IDD_Segmentation'], help="Dataset/Cityscapes, Dataset/Mapillary, Dataset/IDD_Segmentation")
    parser.add_argument('--path_soft', type=str, default='', help='soft pseudo label for rectification')
    parser.add_argument('--used_save_pseudo', action='store_true', help='if True used saved pseudo label')
    parser.add_argument('--no_droplast', action='store_true')

    parser.add_argument('--img_size', type=str, default='640,320', help='image resolution')
    parser.add_argument('--resize', type=int, default=640, help='resize long size')
    parser.add_argument('--rcrop', type=str, default='320,160', help='rondom crop size')
    parser.add_argument('--hflip', type=float, default=0.5, help='random flip probility')

    parser.add_argument('--n_class', type=int, default=7, help='7|19')
    parser.add_argument('--num_workers', type=int, default=6)
    #loss
    parser.add_argument('--gan', type=str, default='LS', help='Vanilla|LS')
    parser.add_argument('--adv', type=float, default=0.01, help='loss weight of adv loss, only use when stage=warm_up')

    #print
    parser.add_argument('--print_interval', type=int, default=20, help='print loss')
    parser.add_argument('--val_interval', type=int, default=1000, help='validate model iter')

    parser.add_argument('--noshuffle', action='store_true', help='do not use shuffle')
    parser.add_argument('--noaug', action='store_true', help='do not use data augmentation')

    parser.add_argument('--norepeat', action='store_true', help='do not repeat the data')

    parser.add_argument('--rectify', action='store_true')
    parser.add_argument("--train_thred", default=-1, type=float)
    parser.add_argument("--gamma", default=1, type=float)
    parser.add_argument("--ratio", default=1, type=float)
    return parser

def relative_path_to_absolute_path(opt):
    opt.rcrop = [int(opt.rcrop.split(',')[0]), int(opt.rcrop.split(',')[1])]
    opt.img_size = (int(opt.img_size.split(',')[0]), int(opt.img_size.split(',')[1]))
    opt.resume_path = os.path.join(opt.root, 'Code/CoaST', opt.resume_path)
    opt.src_rootpath = os.path.join(opt.root, opt.src_rootpath)
    for i in range(len(opt.tgt_dataset_list)):
        opt.tgt_rootpath_list[i] = os.path.join(opt.root, opt.tgt_rootpath_list[i])
    opt.path_soft = os.path.join(opt.root, 'Code/CoaST/Pseudo', opt.name)
    opt.logdir = os.path.join(opt.root, 'Code/CoaST', 'logs', opt.name)
    return opt