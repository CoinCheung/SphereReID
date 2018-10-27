#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.utils.model_zoo as model_zoo


resnet50_url = 'https://download.pytorch.org/models/resnet50-19c8e357.pth'


class Network_D(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Network_D, self).__init__(*args, **kwargs)
        resnet50 = torchvision.models.resnet50()

        self.conv1 = resnet50.conv1
        self.bn1 = resnet50.bn1
        self.relu = resnet50.relu
        self.maxpool = resnet50.maxpool
        self.layer1 = resnet50.layer1
        self.layer2 = resnet50.layer2
        self.layer3 = resnet50.layer3
        self.layer4 = resnet50.layer4
        self.bn2 = nn.BatchNorm1d(2048)
        self.dp = nn.Dropout(0.5)
        self.fc = nn.Linear(in_features = 2048, out_features = 1024, bias = True)
        self.bn3 = nn.BatchNorm1d(1024)

        state = model_zoo.load_url(resnet50_url)
        for k, v in state.items():
            if k == 'fc':
                print(k)
                continue
            self.state_dict().update({k: v})


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.avg_pool2d(x, x.size()[2:]).view(x.size()[:2])
        x = self.bn2(x)
        x = self.dp(x)
        x = self.fc(x)
        embd = self.bn3(x)
        ## TODO: what is L2Norm in the paper ?
        return embd



if __name__ == '__main__':
    intensor = torch.randn(10, 3, 256, 128)
    net = Network_D()
    out = net(intensor)
    print(out.shape)
