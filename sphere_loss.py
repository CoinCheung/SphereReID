#! /usr/bin/python
# -*- encoding: utf-8 -*-



import torch
import torch.nn as nn



class SphereLoss(nn.Module):
    def __init__(self, in_feats, n_classes, scale = 14, *args, **kwargs):
        super(SphereLoss, self).__init__(*args, **kwargs)
        self.scale = 14
        self.cross_entropy = nn.CrossEntropyLoss()
        self.W = torch.nn.Parameter(torch.randn(in_feats, n_classes),
                requires_grad = True)
        #  nn.init.kaiming_uniform_(self.W, a=1)
        nn.init.kaiming_normal_(self.W, a=1)


    def forward(self, x, label):
        x_norm = torch.norm(x, 2, 1, True).clamp(min = 1e-12).expand_as(x)
        x_norm = x / x_norm
        w_norm = torch.norm(self.W, 2, 0, True).clamp(min = 1e-12).expand_as(self.W)
        w_norm = self.W / w_norm
        cos_th = torch.mm(x_norm, w_norm)
        s_cos_th = self.scale * cos_th
        loss = self.cross_entropy(s_cos_th, label)
        return loss



if __name__ == '__main__':
    Loss = SphereLoss(1024, 10)
    a = torch.randn(20, 1024)
    lb = torch.ones(20, dtype = torch.long)
    loss = Loss(a, lb)
    loss.backward()
    print(loss.detach().numpy())
    print(list(Loss.parameters())[0].shape)
    print(type(next(Loss.parameters())))
