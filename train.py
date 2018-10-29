#!/usr/bin/python
# -*- encoding: utf-8 -*-


import logging
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from backbone import Network_D
from sphere_loss import SphereLoss
from market1501 import Market1501
from balanced_sampler import BalancedSampler



def train():
    ## logging
    FORMAT = '%(levelname)s %(filename)s:%(lineno)4d: %(message)s'
    logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
    logger = logging.getLogger(__name__)

    ## data
    dataset = Market1501('./dataset/Market-1501-v15.09.15/bounding_box_train')
    sampler = BalancedSampler(dataset, 16, 4)
    dl = DataLoader(dataset, batch_sampler = sampler, num_workers = 4)
    num_classes = dataset.get_num_classes()

    ## network and loss
    sphereloss = SphereLoss(1024, num_classes)
    net = Network_D()
    net.train()
    net = nn.DataParallel(net)
    net.cuda()

    ## optimizer
    params = list(net.parameters())
    params += list(sphereloss.parameters())
    optim = torch.optim.Adam(params, lr = 1e-3, weight_decay = 5e-4)


if __name__ == '__main__':
    train()
