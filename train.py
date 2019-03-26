#!/usr/bin/python
# -*- encoding: utf-8 -*-

import time
import logging
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from backbone import Network_D
from sphere_loss import SphereLoss, OhemSphereLoss
from market1501 import Market1501
from balanced_sampler import BalancedSampler


## logging
if not os.path.exists('./res/'): os.makedirs('./res/')
logfile = 'sphere_reid-{}.log'.format(time.strftime('%Y-%m-%d-%H-%M-%S'))
logfile = os.path.join('res', logfile)
FORMAT = '%(levelname)s %(filename)s(%(lineno)d): %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, filename=logfile)
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())


#  start_lr = 1e-2

def lr_scheduler(epoch, optimizer):
    warmup_epoch = 20
    warmup_lr = 1e-5
    lr_steps = [90, 130]
    start_lr = 1e-3
    lr_factor = 0.1

    if epoch <= warmup_epoch:  # lr warmup
        warmup_scale = (start_lr / warmup_lr) ** (1.0 / warmup_epoch)
        lr = warmup_lr * (warmup_scale ** epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        optimizer.defaults['lr'] = lr
    else:  # lr jump
        for i, el in enumerate(lr_steps):
            if epoch == el:
                lr = start_lr * (lr_factor ** (i + 1))
                logger.info('====> LR is set to: {}'.format(lr))
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                optimizer.defaults['lr'] = lr
    lrs = [round(el['lr'], 6) for el in optimizer.param_groups]
    return optimizer, lrs


def train():
    ## data
    logger.info('creating dataloader')
    dataset = Market1501('./dataset/Market-1501-v15.09.15/bounding_box_train',
            is_train = True)
    num_classes = dataset.get_num_classes()
    sampler = BalancedSampler(dataset, 16, 4)
    dl = DataLoader(dataset,
            batch_sampler = sampler,
            num_workers = 8)

    ## network and loss
    logger.info('setup model and loss')
    #  criteria = SphereLoss(1024, num_classes)
    criteria = OhemSphereLoss(1024, num_classes, thresh=0.8)
    criteria.cuda()
    net = Network_D()
    net.train()
    net.cuda()

    ## optimizer
    logger.info('creating optimizer')
    params = list(net.parameters())
    params += list(criteria.parameters())
    optim = torch.optim.Adam(params, lr = 1e-3)

    ## training
    logger.info('start training')
    t_start = time.time()
    loss_it = []
    for ep in range(150):
        optim, lrs = lr_scheduler(ep, optim)
        for it, (imgs, lbs, ids) in enumerate(dl):
            imgs = imgs.cuda()
            lbs = lbs.cuda()

            embs = net(imgs)
            loss = criteria(embs, lbs)
            optim.zero_grad()
            loss.backward()
            optim.step()

            loss_it.append(loss.detach().cpu().numpy())
        ## print log
        t_end = time.time()
        t_interval = t_end - t_start
        log_loss = sum(loss_it) / len(loss_it)
        msg = 'epoch: {}, iter: {}, loss: {:.4f}, lr: {}, time: {:.4f}'.format(ep,
                it, log_loss, lrs, t_interval)
        logger.info(msg)
        loss_it = []
        t_start = t_end

    ## save model
    torch.save(net.state_dict(), './res/model_final.pkl')
    logger.info('\nTraining done, model saved to {}\n\n'.format('./res/model_final.pkl'))


if __name__ == '__main__':
    train()
