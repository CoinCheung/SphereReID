#!/usr/bin/python
# -*- encoding: utf-8 -*-


import sys
import os
import os.path as osp
import logging
import pickle
from tqdm import tqdm
import numpy as np
import torch
from backbone import Network_D
from torch.utils.data import DataLoader
from market1501 import Market1501



FORMAT = '%(levelname)s %(filename)s(%(lineno)d): %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)


def embed():
    ## load checkpoint
    res_pth = './res'
    mod_pth = osp.join(res_pth, 'model_final.pkl')
    net = Network_D()
    net.load_state_dict(torch.load(mod_pth))
    net.cuda()
    net.eval()

    ## data loader
    query_set = Market1501('./dataset/Market-1501-v15.09.15/query',
            is_train = False)
    gallery_set = Market1501('./dataset/Market-1501-v15.09.15/bounding_box_test',
            is_train = False)
    query_loader = DataLoader(query_set,
                        batch_size = 32,
                        num_workers = 4,
                        drop_last = False)
    gallery_loader = DataLoader(gallery_set,
                        batch_size = 32,
                        num_workers = 4,
                        drop_last = False)

    ## embed
    query_pids = []
    query_camids = []
    query_embds = []
    gallery_pids = []
    gallery_camids = []
    gallery_embds = []
    logger.info('embedding query set ...')
    for i, (im, _, ids) in enumerate(tqdm(query_loader)):
        im = im.cuda()
        pid = ids[0].numpy()
        camid = ids[1].numpy()
        embed = net(im).detach().cpu().numpy()
        query_embds.append(embed)
        query_pids.extend(pid)
        query_camids.extend(camid)

    logger.info('embedding gallery set ...')
    for i, (im, _, ids) in enumerate(tqdm(gallery_loader)):
        im = im.cuda()
        pid = ids[0].numpy()
        camid = ids[1].numpy()
        embed = net(im).detach().cpu().numpy()
        gallery_embds.append(embed)
        gallery_pids.extend(pid)
        gallery_camids.extend(camid)

    query_pids = np.array(query_pids)
    query_camids = np.array(query_camids)
    query_embds = np.vstack(query_embds)
    gallery_pids = np.array(gallery_pids)
    gallery_camids = np.array(gallery_camids)
    gallery_embds = np.vstack(gallery_embds)

    ## dump embeds results
    embd_res = (query_embds, query_pids, query_camids, gallery_embds, gallery_pids, gallery_camids)
    with open('./res/embds.pkl', 'wb') as fw:
        pickle.dump(embd_res, fw)
    logger.info('embedding done, dump to: ./res/embds.pkl')

    return embd_res


def evaluate(embd_res = None):
    if embd_res == None:
        with open('./res/embds.pkl', 'rb') as fr:
            embd_res = pickle.load(fr)

    query_embds, query_pids, query_camids, gallery_embds, gallery_pids, gallery_camids = embd_res

    ## compute distance matrix
    logger.info('compute distance matrix')
    dist_mtx = np.matmul(query_embds, gallery_embds.T)
    dist_mtx = 1.0 / (dist_mtx + 1)
    n_q, n_g = dist_mtx.shape

    logger.info('start evaluating ...')
    indices = np.argsort(dist_mtx, axis = 1)
    matches = gallery_pids[indices] == query_pids[:, np.newaxis]
    matches = matches.astype(np.int32)
    all_aps = []
    for query_idx in tqdm(range(n_q)):
        query_pid = query_pids[query_idx]
        query_camid = query_camids[query_idx]

        ## exclude same gallery pictures
        order = indices[query_idx]
        pid_diff = gallery_pids[order] != query_pid
        camid_diff = gallery_camids[order] != query_camid
        keep = np.logical_or(pid_diff, camid_diff)
        match = matches[query_idx][keep]

        if not np.any(match): continue

        num_real = match.sum()
        match_cum = match.cumsum()
        match_cum = [el / (1.0 + i) for i, el in enumerate(match_cum)]
        match_cum = np.array(match_cum) * match
        ap = match_cum.sum() / num_real
        all_aps.append(ap)

    assert len(all_aps) > 0, "NO QUERRY APPEARS IN THE GALLERY"
    mAP = sum(all_aps) / len(all_aps)

    return mAP



if __name__ == '__main__':
    embd_res = embed()
    mAP = evaluate()
    print('map is: {}'.format(mAP))

