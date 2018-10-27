#! /usr/bin/python
# -*- encoding: utf-8 -*-


import os
import os.path as osp
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset



class Market1501(Dataset):
    def __init__(self, data_pth, *args, **kwargs):
        super(Market1501, self).__init__(*args, **kwargs)

        imgs = os.listdir(data_pth)
        imgs = [im for im in imgs if osp.splitext(im)[-1] == '.jpg']
        self.im_pths = [osp.join(data_pth, im) for im in imgs]
        self.im_infos = {}
        self.person_infos = {}
        for i, im in enumerate(imgs):
            tokens = im.split('_')
            im_pth = self.im_pths[i]
            pid = int(tokens[0])
            cam = int(tokens[1][1])
            self.im_infos.update({im_pth: (pid, cam)})
            if pid in self.person_infos.keys():
                self.person_infos[pid].append(i)
            else:
                self.person_infos[pid] = [i, ]

        for pid, ids in self.person_infos.items():
            self.person_infos[pid] = np.array(ids, dtype = np.int32)


    def __getitem__(self, idx):
        im_pth = self.im_pths[idx]
        im = cv2.imread(im_pth)
        return im, self.im_infos[im_pth][0]

    def __len__(self):
        return len(self.im_pths)




if __name__ == "__main__":
    ds_train = Market1501('./dataset/Market-1501-v15.09.15/bounding_box_train')
    ds_test = Market1501('./dataset/Market-1501-v15.09.15/bounding_box_test')
    im = ds_train[1]
    cv2.imshow('img', im)
    cv2.waitKey(0)
