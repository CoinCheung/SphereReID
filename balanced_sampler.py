#! /usr/bin/python
# -*- encoding: utf-8 -*-


import random
import numpy as np
import torch
from torch.utils.data.sampler import Sampler


class BalancedSampler(Sampler):
    def __init__(self, data_source, P, K, *args, **kwargs):
        super(BalancedSampler, self).__init__(data_source, *args, **kwargs)

        self.data_source = data_source
        self.P, self.K = P, K
        self.person_infos = data_source.person_infos
        self.persons = list(data_source.person_infos.keys())
        random.shuffle(self.persons)
        self.iter_num = len(self.persons) // P


    def __iter__(self):
        curr_p = 0
        for it in range(self.iter_num):
            pids = self.persons[curr_p: curr_p + self.P]
            curr_p += self.P
            ids = []
            for pid in pids:
                if len(self.person_infos[pid]) >= self.K:
                    id_sam = np.random.choice(self.person_infos[pid], self.K, False)
                    ids.extend(id_sam.tolist())
                else:
                    id_sam = np.random.choice(self.person_infos[pid], self.K, True)
                    ids.extend(id_sam.tolist())
            yield ids


    def __len__(self):
        return self.iter_num



if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from market1501 import Market1501
    import cv2
    ds = Market1501('./dataset/Market-1501-v15.09.15/bounding_box_train')
    #  im = ds[1]
    #  cv2.imshow('img', im)
    #  cv2.waitKey(0)
    sampler = BalancedSampler(ds, 16, 4)
    dl = DataLoader(ds, batch_sampler = sampler, num_workers = 1)

    for i, (imgs, lbs) in enumerate(dl):
        print(i)
        if i == 0:
            print(len(lbs))
            print(lbs)


