# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

import os
import time
import math
import pickle
import argparse
import numpy as np
import pandas as pd
from os import listdir
# import cPickle as pickle
import scipy.sparse as sp
from model import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

which_gpu = "7"
os.environ["CUDA_VISIBLE_DEVICES"] = which_gpu

def normalize_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -1).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).tocoo()

def load_data(data, is_train=True):
    if is_train == True:
        data = data.sample(frac=1).reset_index(drop=True)

    data['hub_list'] = data['hub_list'].apply(lambda x: list(map(int, x.split(' '))))
    data['link_list'] = data['link_list'].apply(lambda x: list(map(int, x.split(' ')))) 
    data['hub_coherence'] = data['hub_coherence'].apply(lambda x: list(map(int, x.split(' '))))
    data['link_coherence'] = data['link_coherence'].apply(lambda x: list(map(int, x.split(' ')))) 
    data['link_distance'] = data['link_distance'].apply(lambda x: list(map(float, x.split(' '))))
    data['link_eta'] = data['link_eta'].apply(lambda x: list(map(float, x.split(' ')))) 
    
    train_num = data.shape[0]
    hub_index = list(data['hub_list'])
    link_index = list(data['link_list'])
    hubcoherence_idx = list(data['hub_coherence'])
    linkcoherence_idx = list(data['link_coherence'])
    link_distance = list(data['link_distance'])
    link_eta = list(data['link_eta'])
    labels = list(data['label'])
    distance = list(data['distance'])
    duration = list(data['duration'])
    mode = list(data['mode'])

    # one-hot user demographic attributes and weather conditions
    context_feature = ['context'+str(i) for i in range(126)]
    
    cxtfeat = torch.FloatTensor(data[context_feature].values)
    
    hubfeat = np.full((train_num, max_len), int(num_hubs), dtype=np.int64)
    linkfeat = np.full((train_num, max_len), int(num_links), dtype=np.int64)
    hubcoherence = np.full((train_num, max_len), int(num_hubs), dtype=np.int64)
    linkcoherence = np.full((train_num, max_len), int(num_links), dtype=np.int64)
    linkdistance = np.full((train_num, max_len), 0, dtype=np.float64)
    linketa = np.full((train_num, max_len), 0, dtype=np.float64)
    hubmask = np.full((train_num, max_len), 1, dtype=np.int64)
    linkmask = np.full((train_num, max_len), 1, dtype=np.int64)

    for i in range(train_num):
        for j in range(len(hub_index[i])):
            hubfeat[i, j] = hub_index[i][j]
            hubcoherence[i, j] = hubcoherence_idx[i][j]
            hubmask[i, j] = 0
    for i in range(train_num):
        for j in range(len(list(link_index[i]))):
            linkfeat[i, j] = link_index[i][j]
            linkcoherence[i, j] = linkcoherence_idx[i][j]
            linkdistance[i, j] = link_distance[i][j]
            linketa[i, j] = link_eta[i][j]
            linkmask[i, j] = 0

    num_iter = int(train_num / batch_size)
    hubfeat = torch.LongTensor(hubfeat)
    linkfeat = torch.LongTensor(linkfeat)
    hubcoherence = torch.LongTensor(hubcoherence)
    linkcoherence = torch.LongTensor(linkcoherence)
    linkdistance = torch.FloatTensor(linkdistance)
    linketa = torch.FloatTensor(linketa)
    hubmask = torch.FloatTensor(hubmask)
    linkmask = torch.FloatTensor(linkmask)
    labels = torch.FloatTensor(np.array(labels).reshape((-1, 1)))
    routedistance = torch.FloatTensor(np.array(distance).reshape((-1, 1)))
    routeduration = torch.FloatTensor(np.array(duration).reshape((-1, 1)))
    routemode = torch.LongTensor(np.array(mode).reshape((-1, 1)))
    
    return cxtfeat, hubfeat, linkfeat, linkdistance, linketa, \
           hubcoherence, linkcoherence, hubmask, linkmask, labels, routedistance, routeduration, routemode, num_iter

def get_batch(cxtfeat, hubfeat, linkfeat, linkdistance, linketa, hubcoherence, linkcoherence, hubmask, linkmask, \
              labels, distance, duration, mode, i):
    """
    splits the training dataset into small batches.
    """
    b_cxtfeat = cxtfeat[i*batch_size:i*batch_size+batch_size].cuda()
    b_hubfeat = hubfeat[i*batch_size:i*batch_size+batch_size].cuda()
    b_linkfeat = linkfeat[i*batch_size:i*batch_size+batch_size].cuda()
    b_linkdistance = linkdistance[i*batch_size:i*batch_size+batch_size].cuda()
    b_linketa = linketa[i*batch_size:i*batch_size+batch_size].cuda()
    b_hubcoherence = hubcoherence[i*batch_size:i*batch_size+batch_size].cuda()
    b_linkcoherence = linkcoherence[i*batch_size:i*batch_size+batch_size].cuda()
    b_hubmask = hubmask[i*batch_size:i*batch_size+batch_size].cuda()
    b_linkmask = linkmask[i*batch_size:i*batch_size+batch_size].cuda()
    b_label = labels[i*batch_size:i*batch_size+batch_size].cuda()
    b_distance = distance[i*batch_size:i*batch_size+batch_size].cuda()
    b_duration = duration[i*batch_size:i*batch_size+batch_size].cuda()
    b_mode = mode[i*batch_size:i*batch_size+batch_size].cuda()
    return (b_cxtfeat, b_hubfeat, b_linkfeat, b_linkdistance, b_linketa, \
            b_hubcoherence, b_linkcoherence, b_hubmask, b_linkmask, b_label, b_distance, b_duration, b_mode)

def evaluation(data):
    data.sort_values('predict', ascending=False, inplace = True)
    grouped = data.groupby('query', as_index=False)
    grouped = grouped.aggregate({'label':list, 'predict':list})
    grouped['length'] = grouped['label'].apply(lambda x: len(x))
    label = list(grouped['label'])
    predict = list(grouped['predict'])
    hit1, hit3, hit5 = 0, 0, 0
    ndcg3, ndcg5, ndcg10 = 0, 0, 0
    import math
    for i in range(len(label)):
        for idx, j in enumerate(label[i]):
            if j == 1 and idx == 0:
                hit1 += 1
            if j == 1 and idx <= 2:
                hit3 += 1
            if j == 1 and idx <= 4:
                hit5 += 1
            if idx <= 2:
                ndcg3 += (j*1.0 / math.log(idx+2, 2))
            if idx <= 4:
                ndcg5 += (j*1.0 / math.log(idx+2, 2))
            if idx <= 9:
                ndcg10 += (j*1.0 / math.log(idx+2, 2))

    print(grouped.shape, hit1, hit3, hit5)
    print('Hit@1:', hit1*1.0/grouped.shape[0], \
          'Hit@3:', hit3*1.0/grouped.shape[0], \
          'Hit@5:', hit5*1.0/grouped.shape[0], \
          'NDCG@3:', ndcg3/grouped.shape[0], \
          'NDCG@5:', ndcg5/grouped.shape[0], \
          'NDCG@10:', ndcg10/grouped.shape[0])

hour_list = ['0000', '0100', '0200', '0300', '0400', '0500', '0600', '0700', '0800', '0900', '1000', '1100', '1200',
             '1300', '1400', '1500', '1600', '1700', '1800', '1900', '2000', '2100', '2200', '2300']

def train(epoch):
    # training
    print('Training...')
    data = pd.read_csv('sample_data/sample_query.csv', sep='\t')
    print(data.shape)
    cnt = 0
    loss_total = 0
    cxtfeat, hubfeat, linkfeat, linkdistance, linketa, hubcoherence, linkcoherence, \
    hubmask, linkmask, labels, routedistance, routeduration, routemode, num_iter = load_data(data)

     # hub dynamic features at previous two hours 
    hub_traffic = pickle.load(open('sample_data/hub_traffic', 'rb'), encoding='iso-8859-1')
    hub_traffic = torch.FloatTensor(hub_traffic).cuda() # (N, T, Feature)
    # link dynamic features at previous two hours 
    link_traffic = pickle.load(open('sample_data/link_traffic', 'rb'), encoding='iso-8859-1')
    link_traffic = torch.FloatTensor(link_traffic).cuda() # (N, T, Feature)

    for i in range(num_iter+1):
        batch_data = get_batch(cxtfeat, hubfeat, linkfeat, linkdistance, linketa, \
                               hubcoherence, linkcoherence, hubmask, linkmask, labels, \
                               routedistance, routeduration, routemode, i)

        b_cxtfeat, b_hubfeat, b_linkfeat, b_linkdistance, b_linketa, b_hubcoherence, b_linkcoherence, \
        b_hubmask, b_linkmask, b_label, b_distance, b_duration, b_mode = batch_data
        
        optimizer.zero_grad()
        out_rec, vertex_loss, route_loss = model(batch_data, hub_adj_mat, link_adj_mat, \
                                                 hub_static, link_static, \
                                                 hub_traffic, link_traffic)
#         loss_a += 1
        loss_train = loss(out_rec, b_label)+0.3*vertex_loss+0.1*route_loss
        loss_total += loss_train.item()
        loss_train.backward()
        optimizer.step()

        if cnt%5 == 1:
            print('Epoch: {:04d}'.format(epoch+1),
                  'Step: {:06d}'.format(cnt+1),
                  'loss_train: {:.4f}'.format(loss_total/2)
                 )
            loss_total = 0
        cnt += 1
        
    # validation
    print('validation...')
    torch.save(model, 'hmtrl.pkl')
    #torch.save(model.state_dict(), 'parameter.pkl')
    data = pd.read_csv('sample_data/sample_query.csv', sep='\t')
    cnt = 0
    loss_val = 0
    loss_total = 0
    total_valdata = []

    cxtfeat, hubfeat, linkfeat, linkdistance, linketa, hubcoherence, linkcoherence, \
    hubmask, linkmask, labels, routedistance, routeduration, routemode, num_iter = load_data(data)

     # hub dynamic features at previous two hours 
    hub_traffic = pickle.load(open('sample_data/hub_traffic', 'rb'), encoding='iso-8859-1')
    hub_traffic = torch.FloatTensor(hub_traffic).cuda() # (N, T, Feature)
    # link dynamic features at previous two hours 
    link_traffic = pickle.load(open('sample_data/link_traffic', 'rb'), encoding='iso-8859-1')
    link_traffic = torch.FloatTensor(link_traffic).cuda() # (N, T, Feature)

    score = []
    for i in range(num_iter+1):
        cnt += 1
        batch_data = get_batch(cxtfeat, hubfeat, linkfeat, linkdistance, linketa, \
                               hubcoherence, linkcoherence, hubmask, linkmask, labels, \
                               routedistance, routeduration, routemode, i)

        b_cxtfeat, b_hubfeat, b_linkfeat, b_linkdistance, b_linketa, \
        b_hubcoherence, b_linkcoherence, b_hubmask, b_linkmask, b_label, b_distance, b_duration, b_mode = batch_data

        out_rec, vertex_loss, route_loss = model(batch_data, hub_adj_mat, link_adj_mat, \
                                                 hub_static, link_static, \
                                                 hub_traffic, link_traffic)
        loss_val = loss(out_rec, b_label)+0.3*vertex_loss+0.1*route_loss
        loss_total += loss_val.item()
        score += list(out_rec.data.cpu().numpy())
    data['predict'] = score
    print('validation loss:', loss_total/cnt)
    
    evaluation(data)
    torch.cuda.empty_cache()

T = 2
head = 8
gamma = 0.5
hid_dim = 8
hub_dim = 27
link_dim = 36
max_len = 800
embed_size = 32
batch_size = 16
learning_rate = 0.0001
# link_nodes = 420889
# station_nodes = 3218

hub_adj_mat = pickle.load(open('sample_data/hub_adj_mat', 'rb'), encoding='iso-8859-1') # hub-centric graph adjacency matrix
link_adj_mat = pickle.load(open('sample_data/link_adj_mat', 'rb'), encoding='iso-8859-1') # link-centric graph adjacency matrix

num_hubs = hub_adj_mat.shape[0]
num_links = link_adj_mat.shape[0]

link_adj_mat = link_adj_mat - sp.diags(np.array([1]*link_adj_mat.shape[0]))
link_adj_mat = link_adj_mat.tocoo()

# hub_adj_mat = normalize_adj(hub_adj_mat)
# link_adj_mat = normalize_adj(link_adj_mat)

# print(hub_adj_mat)
# print(link_adj_mat)
hub_adj_mat = sparse_mx_to_torch_sparse_tensor(hub_adj_mat)
link_adj_mat = sparse_mx_to_torch_sparse_tensor(link_adj_mat)

# hub static features
hub_static = pickle.load(open('sample_data/hub_static', 'rb'), encoding='iso-8859-1') # (N, F)
hub_static = torch.FloatTensor(hub_static).cuda()

# link static features
link_static = pickle.load(open('sample_data/link_static', 'rb'), encoding='iso-8859-1') # (N, F)
link_static = torch.FloatTensor(link_static).cuda()

model = hmtrl_model()
loss = nn.BCELoss()
# loss = nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([10.0,1.0]).cuda())
optimizer = optim.Adam(model.parameters(), lr = learning_rate)

model.cuda()

hub_adj_mat = hub_adj_mat.cuda()
link_adj_mat = link_adj_mat.cuda()

# Train model
t_start = time.time()
for epoch in range(10):
    train(epoch)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_start))
