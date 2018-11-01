#! /usr/bin/python
# -*- encoding: utf-8 -*-


import os
import os.path as osp
import numpy as np
import cv2
from PIL import Image
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset



class Market1501(Dataset):
    def __init__(self, data_pth, is_train = True, *args, **kwargs):
        super(Market1501, self).__init__(*args, **kwargs)

        ## parse image names to generate image ids
        imgs = os.listdir(data_pth)
        imgs = [im for im in imgs if osp.splitext(im)[-1] == '.jpg']
        self.is_train = is_train
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

        self.pid_label_map = {}
        for i, (pid, ids) in enumerate(self.person_infos.items()):
            self.person_infos[pid] = np.array(ids, dtype = np.int32)
            self.pid_label_map[pid] = i

        ## preprocessing
        self.trans_train = transforms.Compose([
                transforms.Resize((288, 144)),
                transforms.RandomCrop((256, 128)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
        self.trans_no_train = transforms.Compose([
                transforms.Resize((288, 144)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])


    def __getitem__(self, idx):
        im_pth = self.im_pths[idx]
        pid = self.im_infos[im_pth][0]
        im = Image.open(im_pth)
        if self.is_train:
            im = self.trans_train(im)
        else:
            im = self.trans_no_train(im)
        return im, self.pid_label_map[pid], self.im_infos[im_pth]

    def __len__(self):
        return len(self.im_pths)


    def get_num_classes(self):
        return len(list(self.person_infos.keys()))



if __name__ == "__main__":
    ds_train = Market1501('./dataset/Market-1501-v15.09.15/bounding_box_train')
    ds_test = Market1501('./dataset/Market-1501-v15.09.15/bounding_box_test', is_train = False)
    im, lb, _ = ds_train[1]
    print(ds_train.get_num_classes())
    print(ds_test.get_num_classes())
    print(im.shape)
    #  im = im.numpy().transpose(1,2,0)
    #  cv2.imshow('img', im)
    #  cv2.waitKey(0)

    from torch.utils.data import DataLoader
    loader = DataLoader(ds_test,
                        batch_size = 4,
                        num_workers = 1,
                        drop_last = False)
    for im, _, ids in loader:
        print(im)
        print(ids)
        break
