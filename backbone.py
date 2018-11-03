#!/usr/bin/python
# -*- encoding: utf-8 -*-


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
        self.layer4 = create_layer(1024, 512, stride=1)
        self.bn2 = nn.BatchNorm1d(2048)
        self.dp = nn.Dropout(0.5)
        self.fc = nn.Linear(in_features=2048, out_features=1024, bias=True)
        self.bn3 = nn.BatchNorm1d(1024)

        # load pretrained weights and initialize added weight
        pretrained_state = model_zoo.load_url(resnet50_url)
        state_dict = self.state_dict()
        for k, v in pretrained_state.items():
            if 'fc' in k:
                continue
            state_dict.update({k: v})
        self.load_state_dict(state_dict)
        nn.init.kaiming_normal_(self.fc.weight, a=1)
        nn.init.constant_(self.fc.bias, 0)

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
        if not self.training:
            embd_norm = torch.norm(embd, 2, 1, True).clamp(min=1e-12).expand_as(embd)
            embd = embd / embd_norm
        return embd


class Bottleneck(nn.Module):
    def __init__(self, in_chan, mid_chan, stride=1, *args, **kwargs):
        super(Bottleneck, self).__init__(*args, **kwargs)

        out_chan = 4 * mid_chan
        self.conv1 = nn.Conv2d(in_chan, mid_chan, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_chan)
        self.conv2 = nn.Conv2d(mid_chan, mid_chan, kernel_size=3, stride=stride,
                padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_chan)
        self.conv3 = nn.Conv2d(mid_chan, out_chan, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = None
        if in_chan != out_chan or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_chan, out_chan, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_chan))

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample == None:
            residual = x
        else:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


def create_layer(in_chan, mid_chan, stride):
    out_chan = in_chan * 2
    return nn.Sequential(
        Bottleneck(in_chan, mid_chan, stride=stride),
        Bottleneck(out_chan, mid_chan, stride=1),
        Bottleneck(out_chan, mid_chan, stride=1))



if __name__ == '__main__':
    intensor = torch.randn(10, 3, 256, 128)
    net = Network_D()
    out = net(intensor)
    print(out.shape)
    #  for el in net.parameters():
    #      print(el)
    #      break
    #  print(net.parameters())

    params = list(net.parameters())
    optim = torch.optim.Adam(params, lr = 1e-3, weight_decay = 5e-4)
    lr = 3
    optim.defaults['lr'] = 4
    for param_group in optim.param_groups:
        param_group['lr'] = lr
        print(param_group.keys())
        print(param_group['lr'])
    print(optim.defaults['lr'])
    print(optim.defaults.keys())
