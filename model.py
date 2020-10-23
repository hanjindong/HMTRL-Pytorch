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

import torch
import torch.nn as nn
import scipy.sparse as sp
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

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

loss_regression = nn.L1Loss()
loss_classification = nn.CrossEntropyLoss()

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

class GraphConvolution(Module):
    """
    Graph convolution layer.
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input_feat, adj):
        support = torch.mm(input_feat, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GraphSAGE(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphSAGE, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(2*in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input_feat, adj):
        # we use padding operation to align different feature spaces
#         support = torch.mm(input_feat, self.weight)
        agg_feat = torch.spmm(adj, input_feat)
        agg_feat = torch.cat((agg_feat, input_feat), 1)
        output = torch.mm(agg_feat, self.weight)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'    
    
class PathAttention(nn.Module):
    def __init__(self):
        super(PathAttention, self).__init__()
        self.w1 = Parameter(torch.FloatTensor(embed_size*4, embed_size*2))
        self.w2 = Parameter(torch.FloatTensor(head, embed_size*4))
        self.norm_layer = nn.Softmax(dim=2)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.w1.size(1))
        self.w1.data.uniform_(-stdv, stdv)
        stdv = 1. / math.sqrt(self.w2.size(1))
        self.w2.data.uniform_(-stdv, stdv)
        
    def forward(self, value, mask):
#         print(value.shape, self.w1.repeat(value.shape[0], 1, 1).shape, value.transpose(1,2).shape)
        # print(value.transpose(1,2).shape)
        try:
            x = F.relu(torch.bmm(self.w1.repeat(value.shape[0], 1, 1), value.transpose(1, 2)))
        except:
            print(value)
        x = torch.bmm(self.w2.repeat(x.shape[0], 1, 1), x)
        mask = torch.unsqueeze(mask, 1).repeat(1, head, 1)
        # print(mask.shape, x.shape)
        x = -1e9 * mask + x
        x = self.norm_layer(x)
        x = torch.bmm(x, value)
        return x

class hmtrl_model(nn.Module):
    def __init__(self):
        super(hmtrl_model, self).__init__()

        self.hub_trans = nn.Linear(3, hid_dim)
        self.link_trans = nn.Linear(3, hid_dim)

        self.GRU_hub = nn.GRUCell(hid_dim, hid_dim, bias=True)
        nn.init.xavier_uniform_(self.GRU_hub.weight_ih, gain=math.sqrt(2.0))
        nn.init.xavier_uniform_(self.GRU_hub.weight_hh, gain=math.sqrt(2.0))
        self.GRU_link = nn.GRUCell(hid_dim, hid_dim, bias=True)
        nn.init.xavier_uniform_(self.GRU_link.weight_ih, gain=math.sqrt(2.0))
        nn.init.xavier_uniform_(self.GRU_link.weight_hh, gain=math.sqrt(2.0))

        self.hub_gc1 = GraphSAGE(hub_dim+hid_dim, embed_size*2)
        self.hub_gc2 = GraphSAGE(embed_size*2, embed_size*2)

        self.link_gc1 = GraphSAGE(link_dim+hid_dim, embed_size*2)
        self.link_gc2 = GraphSAGE(embed_size*2, embed_size*2)
        
        self.bigru_hub = torch.nn.GRU(embed_size*2, embed_size*2, bidirectional=True)
        self.bigru_link = torch.nn.GRU(embed_size*2, embed_size*2, bidirectional=True)

        self.coherence_hub_fc = nn.Linear(embed_size*4, embed_size*2)
        self.coherence_link_fc = nn.Linear(embed_size*4, embed_size*2)

        self.path_attn = PathAttention()

        self.vertex_class_fc = nn.Linear(embed_size*2, 4)
        self.vertex_dist_fc = nn.Linear(embed_size*2, 1)
        self.vertex_eta_fc = nn.Linear(embed_size*2, 1)
        
        self.output_fc = nn.Linear(head*embed_size*4+126, 4)
        self.distance_fc = nn.Linear(head*embed_size*4, 1)
        self.duration_fc = nn.Linear(head*embed_size*4, 1)
        self.mode_fc = nn.Linear(head*embed_size*4, 4)
#         self.mode_softmax = nn.Softmax()
#         self.output_fc = nn.Linear(embed_size*2, 1)
        
        self.output_act = nn.Sigmoid()
        
    def forward(self, batch_data, hub_adj_mat, link_adj_mat, hub_static, link_static, hub_traffic, link_traffic):

        b_cxtfeat, b_hubfeat, b_linkfeat, b_linkdistance, b_linketa, b_hubcoherence, b_linkcoherence, \
        b_hubmask, b_linkmask, b_label, b_distance, b_duration, b_mode = batch_data

        N_hub = hub_traffic.shape[0]
        N_link = link_traffic.shape[0]

        # temporal autocorrelation modeling
        h_hub = torch.zeros(N_hub, hid_dim).cuda() # to init GRU hidden state
        h_link = torch.zeros(N_link, hid_dim).cuda() # to init GRU hidden state
        for i in range(T):
            hub_input = self.hub_trans(hub_traffic[:,i,:]) # transform hub traffic features to the same feature space 
            link_input = self.link_trans(link_traffic[:,i,:]) # transform link traffic features to the same feature space 
            h_hub = self.GRU_hub(hub_input, h_hub) # (N_hub, hid_dim)
            h_link = self.GRU_link(link_input, h_link) # (N_link, hid_dim)

        # spatial autocorrelation modeling
        hub_input = torch.cat((hub_static, h_hub), 1)
        link_input = torch.cat((link_static, h_link), 1)

        hub_input = F.relu(self.hub_gc1(hub_input, hub_adj_mat)) # graph convolution from hub-centric view
        hub_input = F.relu(self.hub_gc2(hub_input, hub_adj_mat)) # (N_hub, embed_size)

        link_input = F.relu(self.link_gc1(link_input, link_adj_mat)) # graph convolution from link-centric view
        link_input = F.relu(self.link_gc2(link_input, link_adj_mat)) # (N_link, embed_size)

        zero_padding = torch.zeros(1, embed_size*2).cuda()
        hub_input = torch.cat((hub_input, zero_padding), 0)
        x_hub = hub_input[b_hubfeat]

        link_input = torch.cat((link_input, zero_padding), 0)
        x_link = link_input[b_linkfeat]

        x_hub_coherence = hub_input[b_hubcoherence]
        x_link_coherence = link_input[b_linkcoherence]

        # bi-directional RNN based route coherence modeling
        batch_size = x_hub.shape[0]
        h_bihub = torch.zeros(2, batch_size, embed_size*2).cuda() # to init BiGRU hidden state
        h_bilink = torch.zeros(2, batch_size, embed_size*2).cuda() # to init BiGRU hidden state
        # x_hub:(batch_size, seq_len, embed_size)
        hub_out, hidden = self.bigru_hub(torch.transpose(x_hub, 0, 1), h_bihub) # (seq_len, batch_size, num_directions*hid_size)
        link_out, hidden = self.bigru_link(torch.transpose(x_link, 0, 1), h_bilink)
        hub_out = torch.transpose(hub_out, 0, 1)
        link_out = torch.transpose(link_out, 0, 1)
        hub_out = self.coherence_hub_fc(hub_out) # (batch_size, seq_len, embed_size)
        link_out = self.coherence_hub_fc(link_out)
        
        hub_real = torch.square(hub_out-x_hub)
        hub_fake = torch.square(hub_out-x_hub_coherence)
        link_real = torch.square(link_out-x_link)
        link_fake = torch.square(link_out-x_link_coherence)
        hub_real = torch.sum(hub_real, 2)
        hub_fake = torch.sum(hub_fake, 2)
        link_real = torch.sum(link_real, 2)
        link_fake = torch.sum(link_fake, 2)
        loss_h = F.relu(hub_real-hub_fake+gamma)
        loss_l = F.relu(link_real-link_fake+gamma)
        coherent_hubmask = (1-b_hubmask)*(b_label.repeat(1, 800))
        coherent_linkmask = (1-b_linkmask)*(b_label.repeat(1, 800))
        loss_h = loss_h*coherent_hubmask
        loss_l = loss_l*coherent_linkmask
        loss_h = torch.sum(loss_h)/(torch.sum(coherent_hubmask)+1)
        loss_l = torch.sum(loss_l)/(torch.sum(coherent_linkmask)+1)
        
        # vertex-level tasks
        # vertex_hub_class = self.vertex_class_fc(x_hub)
        # vertex_link_class = self.vertex_class_fc(x_link)
        vertex_dist = self.vertex_dist_fc(x_link)
        vertex_eta = self.vertex_eta_fc(x_link)
        vertex_dist = vertex_dist.squeeze(2)
        vertex_eta = vertex_eta.squeeze(2)
        
        loss_linkdist = torch.square((1-b_linkmask)*vertex_dist-(1-b_linkmask)*b_linkdistance)
        loss_linkdist = torch.sum(loss_linkdist)/(torch.sum((1-b_linkmask))+1)
        loss_linketa = torch.square((1-b_linkmask)*vertex_eta-(1-b_linkmask)*b_linketa)
        loss_linketa = torch.sum(loss_linketa)/(torch.sum((1-b_linkmask))+1)

        # self-attentive route representation learning
        x_hub = self.path_attn(x_hub, b_hubmask) # path attention
        x_link = self.path_attn(x_link, b_linkmask) # path attention

        x_hub = x_hub.view(-1, head*embed_size*2)
        x_link = x_link.view(-1, head*embed_size*2)
        
        x = torch.cat((x_hub, x_link), 1)

        # route-level tasks
        x_distance = self.distance_fc(x)
        x_duration = self.duration_fc(x)
        x_mode = self.mode_fc(x)
        
        # concatenate the context features, such as user profile, weather condition
        x = torch.cat((x, b_cxtfeat), 1)

        # multimodal rank output
        x = self.output_fc(x)
        x = F.sigmoid(x)
        b_mode = b_mode.reshape((-1,))
        x = x[range(x.shape[0]), b_mode].reshape((-1, 1))

        # hierarchical multi-task learning
        vertex_loss = loss_l+loss_h+loss_linkdist+loss_linketa
        route_loss = 0.1*loss_regression(x_distance, b_distance)+\
                     0.5*loss_regression(x_duration, b_duration)+\
                     loss_classification(x_mode, b_mode.squeeze())
        
        return x, vertex_loss, route_loss
    