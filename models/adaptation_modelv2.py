import torch.nn as nn
import torch.nn.functional as F
import os, sys
import torch
import numpy as np
from models.sync_batchnorm import SynchronizedBatchNorm2d, DataParallelWithCallback
from models.deeplabv2 import Deeplab
from models.discriminator import FCDiscriminator
from .utils import freeze_bn, get_scheduler, cross_entropy2d
from data.randaugment import affine_sample
import pdb
from models.cnsn import instance_norm_mix, calc_ins_mean_std

def prob_2_entropy(prob):
    """ convert probabilistic prediction maps to weighted self-information maps
    """
    n, c, h, w = prob.size()
    return -torch.mul(prob, torch.log2(prob + 1e-30)) / np.log2(c)

class CrossNorm_list(nn.Module):
    """CrossNorm module"""
    def __init__(self, nb_target):
        super(CrossNorm_list, self).__init__()

        self.mean_list = [torch.tensor(0.0).to(0) for i in range(nb_target)]
        self.std_list = [torch.tensor(1.0).to(0) for i in range(nb_target)]

    def forward(self, x, current_id, transfer_id):
        size = x.size()
        content_mean, content_std = calc_ins_mean_std(x)

        normalized_feat = (x - content_mean.expand(
            size)) / content_std.expand(size)

        x = normalized_feat * self.std_list[transfer_id].expand(size) + self.mean_list[transfer_id].expand(size)

        return x
    
    def update_ins_mean_std(self, x, current_id):
        with torch.no_grad():
            mean, std = calc_ins_mean_std(x)
            self.mean_list[current_id], self.std_list[current_id] = mean.detach(), std.detach()



class CustomModel():
    def __init__(self, opt, logger, isTrain=True):
        self.opt = opt
        self.class_numbers = opt.n_class
        self.logger = logger
        self.best_iou = -100
        self.nets = []
        self.nets_DP = []
        self.default_gpu = 0
        self.num_target = len(opt.tgt_dataset_list)
        self.img_cn_list = CrossNorm_list(self.num_target)
        self.img_cn_list = self.init_device(self.img_cn_list, gpu_id=self.default_gpu, whether_DP=False)
        self.domain_id = -1 
        if opt.bn == 'sync_bn':
            BatchNorm = SynchronizedBatchNorm2d
        elif opt.bn == 'bn':
            BatchNorm = nn.BatchNorm2d
        else:
            raise NotImplementedError('batch norm choice {} is not implemented'.format(opt.bn))

        if self.opt.no_resume:
            restore_from = None
        else:
            restore_from= opt.resume_path
        self.best_iou = 0

        if self.opt.stage == 'stage1' and opt.norepeat == False:
            self.BaseNet = Deeplab(BatchNorm, num_classes=self.class_numbers, num_target=len(opt.tgt_dataset_list)+1, freeze_bn=False, restore_from=restore_from)
        else:
            self.BaseNet = Deeplab(BatchNorm, num_classes=self.class_numbers, num_target=len(opt.tgt_dataset_list), freeze_bn=False, restore_from=restore_from)
            
        logger.info('the backbone is {}'.format(opt.model_name))

        self.nets.extend([self.BaseNet])

        self.optimizers = []
        self.schedulers = []        
        optimizer_cls = torch.optim.SGD
        optimizer_params = {'lr':opt.lr, 'weight_decay':2e-4, 'momentum':0.9}

        if self.opt.stage == 'warm_up':
            self.net_D_list = []
            self.net_D_DP_list = []
            self.optimizer_D_list = []
            self.DSchedule_list = []
            for i_target in range(self.num_target):
                net_D = FCDiscriminator(inplanes=self.class_numbers)
                net_D_DP = self.init_device(net_D, gpu_id=self.default_gpu, whether_DP=False)
               

                optimizer_D = torch.optim.Adam(net_D.parameters(), lr=1e-4, betas=(0.9, 0.99))

                
                self.nets.extend([net_D])
                self.nets_DP.append(net_D_DP)
                self.net_D_list.append(net_D)
                self.net_D_DP_list.append(net_D_DP)
                
                self.optimizers.extend([optimizer_D])
                self.optimizer_D_list.append(optimizer_D)
                DSchedule = get_scheduler(optimizer_D, opt)
                self.schedulers.extend([DSchedule])
                self.DSchedule_list.append(DSchedule)

        if self.opt.stage == 'warm_up':
            self.BaseOpti = optimizer_cls([{'params':self.BaseNet.get_1x_lr_params(), 'lr':optimizer_params['lr']},
                                           {'params':self.BaseNet.get_10x_lr_params(), 'lr':optimizer_params['lr']*10}], **optimizer_params)
        else:
            self.BaseOpti = optimizer_cls([{'params':self.BaseNet.get_1x_lr_params_new(), 'lr':optimizer_params['lr']},
                                           {'params':self.BaseNet.get_10x_lr_params_new(), 'lr':optimizer_params['lr']*10}], **optimizer_params)

        
        
        self.BaseNet_DP = self.init_device(self.BaseNet, gpu_id=self.default_gpu, whether_DP=False)

        self.nets_DP.append(self.BaseNet_DP)
        
        self.optimizers.extend([self.BaseOpti])

        self.BaseSchedule = get_scheduler(self.BaseOpti, opt)
        self.schedulers.extend([self.BaseSchedule])
  
        self.adv_source_label = 0
        self.adv_target_label = 1
        if self.opt.gan == 'Vanilla':
            self.bceloss = nn.BCEWithLogitsLoss(size_average=True)
        elif self.opt.gan == 'LS':
            self.bceloss = torch.nn.MSELoss()
        
        
    def step_adv(self, source_data, data_target_list, device):
        for net_D in self.net_D_list:
            for param in net_D.parameters():
                param.requires_grad = False
        self.BaseOpti.zero_grad()
 
        source_x = source_data['img'].to(device)
        source_label = source_data['label'].to(device)


        domain_list = [i for i in range(self.num_target)]
        source_output_list = self.BaseNet_DP(source_x, domain_list, ssl=True)
        source_outputUp_list = []
        loss_GTA = 0
        for i in range(self.num_target):
            source_output = source_output_list[i]
            source_outputUp = F.interpolate(source_output['out'], size=source_x.size()[2:], mode='bilinear', align_corners=True)
            source_outputUp_list.append(source_outputUp)
            
            loss_GTA += cross_entropy2d(input=source_outputUp, target=source_label, size_average=True, reduction='mean')

        loss_GTA.backward()

        target_outputUp_list = []
        for i_target in range(self.num_target):
            target_x = data_target_list[i_target]['img'].to(device)
            
            target_output_list = self.BaseNet_DP(target_x, [i_target], ssl=True)
            target_output = target_output_list[0]
            target_outputUp = F.interpolate(target_output['out'], size=target_x.size()[2:], mode='bilinear', align_corners=True)
            target_outputUp_list.append(target_outputUp)

            # adv
            target_D_out = self.net_D_DP_list[i_target](prob_2_entropy(F.softmax(target_outputUp, dim=1)))
            loss_adv_G = self.bceloss(target_D_out, torch.FloatTensor(target_D_out.data.size()).fill_(self.adv_source_label).to(target_D_out.device)) * self.opt.adv
            loss_adv_G.backward()
        
        self.BaseOpti.step()

        for net_D in self.net_D_list:
            for param in net_D.parameters():
                param.requires_grad = True
        for optimizer_D in self.optimizer_D_list:
            optimizer_D.zero_grad()

        for i_target in range(self.num_target):
            source_D_out = self.net_D_DP_list[i_target](prob_2_entropy(F.softmax(source_outputUp_list[i_target].detach(), dim=1)))
            target_D_out = self.net_D_DP_list[i_target](prob_2_entropy(F.softmax(target_outputUp_list[i_target].detach(), dim=1)))
            loss_D = self.bceloss(source_D_out, torch.FloatTensor(source_D_out.data.size()).fill_(self.adv_source_label).to(source_D_out.device)) + \
                        self.bceloss(target_D_out, torch.FloatTensor(target_D_out.data.size()).fill_(self.adv_target_label).to(target_D_out.device))
            loss_D.backward()
        
        for optimizer_D in self.optimizer_D_list:
            optimizer_D.step()

        return loss_GTA.item(), loss_adv_G.item(), loss_D.item()


    def step(self, source_data, data_target_list, device):
  
        source_x = source_data['img'].to(device)
        source_label = source_data['label'].to(device)

        domain_list = [i for i in range(self.num_target+1)]
        source_output_list = self.BaseNet_DP(source_x, domain_list, ssl=True)
        source_outputUp_list = []
        loss_GTA = 0
        for i in range(self.num_target+1):
            source_output = source_output_list[i]
            source_outputUp = F.interpolate(source_output['out'], size=source_x.size()[2:], mode='bilinear', align_corners=True)
            source_outputUp_list.append(source_outputUp)
            
            loss_GTA += cross_entropy2d(input=source_outputUp, target=source_label, size_average=True, reduction='mean')

        loss_GTA.backward()

        
        for i in range(self.num_target):
            self.domain_id = i
            data_i = data_target_list[i]
            target_image = data_i['img'].to(device)
            target_imageS = data_i['img_strong'].to(device)
            target_params = data_i['params']
            target_image_full = data_i['img_full'].to(device)
            target_weak_params = data_i['weak_params']


            target_lpsoft = data_i['lpsoft'].to(device) if 'lpsoft' in data_i.keys() else None
            threshold_arg = F.interpolate(target_lpsoft, scale_factor=0.25, mode='bilinear', align_corners=True)
            

            rectified = threshold_arg
            # rectified = weights * threshold_arg
            threshold_arg = rectified.max(1, keepdim=True)[1]
            rectified = rectified / rectified.sum(1, keepdim=True)
            argmax = rectified.max(1, keepdim=True)[0]
            threshold_arg[argmax < self.opt.train_thred] = 250

            batch, _, w, h = threshold_arg.shape

            # normal forward
            self.img_cn_list.update_ins_mean_std(target_imageS.clone(), i)
            self.BaseNet_DP.update_id(i, i)
            self.BaseNet_DP._enable_update()
            target_out_list = self.BaseNet_DP(target_imageS, [i, self.num_target])
            self.BaseNet_DP._disable_cross_norm()


            target_out = target_out_list[0]
            targetS_out_agnostic = target_out_list[1]
            target_out['out'] = F.interpolate(target_out['out'], size=threshold_arg.shape[2:], mode='bilinear', align_corners=True)
            # target_out['feat'] = F.interpolate(target_out['feat'], size=threshold_arg.shape[2:], mode='bilinear', align_corners=True)
            targetS_out_agnostic['out'] = F.interpolate(targetS_out_agnostic['out'], size=threshold_arg.shape[2:], mode='bilinear', align_corners=True)


            loss = torch.Tensor([0]).to(self.default_gpu)


            threshold_argS = self.label_strong_T(threshold_arg.clone().float(), target_params, padding=250, scale=4).to(torch.int64)
            threshold_arg = threshold_argS

            maskS = (threshold_arg != 250).float()

            # style transfer
            loss_CTS_all = 0
            loss_CTS_transfered_all = 0
            variance_all = 0
            weights_all = 0
            for ii in range(self.num_target):
                if ii == i:
                    continue
                
                transfered_domain = ii
                transfered = self.img_cn_list(target_imageS, i, transfered_domain)           
            
                self.BaseNet_DP._enable_cross_norm()
                self.BaseNet_DP.update_id(i, transfered_domain)
                target_out_transfer_list = self.BaseNet_DP(transfered, [transfered_domain]) 
                self.BaseNet_DP._disable_cross_norm()

                target_out_transfer = target_out_transfer_list[0]
                target_out_transfer['out'] = F.interpolate(target_out_transfer['out'], size=threshold_arg.shape[2:], mode='bilinear', align_corners=True)
                
                # calculate predictive variance
                variance1 = torch.sum(F.kl_div(F.log_softmax(target_out_transfer['out'], dim=1),F.softmax(target_out['out'].detach(), dim=1), reduction='none'), dim=1) 
                variance_all += variance1
                exp_variance1 = torch.exp(-self.opt.gamma*variance1)
                weights_all += exp_variance1.detach()

                loss_CTS_transfered = cross_entropy2d(input=target_out_transfer['out'], target=threshold_arg.reshape([batch, w, h]).detach(),reduction='none')

                loss_CTS_transfered_all += loss_CTS_transfered


            weights_all /= self.num_target-1

            if self.opt.rectify:
                loss_CTS_transfered_all = (((loss_CTS_transfered_all * weights_all + variance_all)*maskS).sum() / maskS.sum()) / (self.num_target-1)
            else:
                loss_CTS_transfered_all = (((loss_CTS_transfered_all + variance_all)*maskS).sum() / maskS.sum()) / (self.num_target-1)
            
            
            loss_CTS_all = cross_entropy2d(input=target_out['out'], target=threshold_arg.reshape([batch, w, h]).detach(), reduction='none')
            if self.opt.rectify:
                loss_CTS_all = (loss_CTS_all * weights_all * maskS).sum() / maskS.sum()
            else:
                loss_CTS_all = (loss_CTS_all * maskS).sum() / maskS.sum()

            # agnostic kd
            student = F.log_softmax(targetS_out_agnostic['out'], dim=1)  
            teacher = F.softmax(target_out['out'].detach(), dim=1)

            loss_kd = F.kl_div(student, teacher, reduction='none')
            mask = (teacher != 250).float()
            loss_kd = (loss_kd * mask).sum() / mask.sum()
            loss_kd /= self.num_target
            

            loss += loss_CTS_all + loss_kd + self.opt.ratio * loss_CTS_transfered_all

            
            loss.backward()
        
        self.BaseOpti.step()
        self.BaseOpti.zero_grad()

        return loss.item(), loss_CTS_all.item(), loss_kd.item(), loss_CTS_transfered_all.item()

    def label_strong_T(self, label, params, padding, scale=1):
        label = label + 1
        for i in range(label.shape[0]):
            for (Tform, param) in params.items():
                if Tform == 'Hflip' and param[i].item() == 1:
                    label[i] = label[i].clone().flip(-1)
                elif (Tform == 'ShearX' or Tform == 'ShearY' or Tform == 'TranslateX' or Tform == 'TranslateY' or Tform == 'Rotate') and param[i].item() != 1e4:
                    v = int(param[i].item() // scale) if Tform == 'TranslateX' or Tform == 'TranslateY' else param[i].item()
                    label[i:i+1] = affine_sample(label[i:i+1].clone(), v, Tform)
                elif Tform == 'CutoutAbs' and isinstance(param, list):
                    x0 = int(param[0][i].item() // scale)
                    y0 = int(param[1][i].item() // scale)
                    x1 = int(param[2][i].item() // scale)
                    y1 = int(param[3][i].item() // scale)
                    label[i, :, y0:y1, x0:x1] = 0
        label[label == 0] = padding + 1  # for strong augmentation, constant padding
        label = label - 1
        return label

    def freeze_bn_apply(self):
        for net in self.nets:
            net.apply(freeze_bn)
        for net in self.nets_DP:
            net.apply(freeze_bn)

    def scheduler_step(self):
        for scheduler in self.schedulers:
            scheduler.step()
    
    def optimizer_zerograd(self):
        for optimizer in self.optimizers:
            optimizer.zero_grad()
    

    def init_device(self, net, gpu_id=None, whether_DP=False):
        gpu_id = gpu_id or self.default_gpu
        device = torch.device("cuda:{}".format(gpu_id) if torch.cuda.is_available() else 'cpu')
        net = net.to(device)
        # if torch.cuda.is_available():
        if whether_DP:
            #net = DataParallelWithCallback(net, device_ids=[0])
            net = DataParallelWithCallback(net, device_ids=range(torch.cuda.device_count()))
        return net
    
    def eval(self, net=None, logger=None):
        """Make specific models eval mode during test time"""
        # if issubclass(net, nn.Module) or issubclass(net, BaseModel):
        if net == None:
            for net in self.nets:
                net.eval()
            for net in self.nets_DP:
                net.eval()
            if logger!=None:    
                logger.info("Successfully set the model eval mode") 
        else:
            net.eval()
            if logger!=None:    
                logger("Successfully set {} eval mode".format(net.__class__.__name__))
        return

    def train(self, net=None, logger=None):
        if net==None:
            for net in self.nets:
                net.train()
            for net in self.nets_DP:
                net.train()
        else:
            net.train()
        return