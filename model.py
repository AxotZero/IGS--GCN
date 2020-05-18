# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 16:47:34 2020

@author: axot
"""

import torch
from torch import nn
from torch.utils import data
from torch.nn.parameter import Parameter
import math

class DataLoader(data.Dataset):

    def __init__(self, datas, targets):
        self.datas = datas
        self.targets = targets
    
    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        
        return self.datas[index], self.targets[index]
    
class MyGCN(nn.Module):
    
    def __init__(self, h=20, w=7, output_shape=2, debug=False):
        super(MyGCN, self).__init__()
        self.debug = debug
        
#       spatial level channel
        c1 = 1
        c2 = 16
        c3 = 16
        c4 = 32
        c5 = 64
        
#       time level
        c7 = 16
        c8 = 32
        
#       spatial level
        self.get_adj = compute_adj(in_channel=c1, hidden_channel=c2, bias=True)
        self.gcn1 = gcn_spa(in_channel=c1, out_channel=c3, bias=True)
        self.gcn2 = gcn_spa(in_channel=c3, out_channel=c4, bias=True)
        self.gcn3 = gcn_spa(in_channel=c4, out_channel=c5, bias=True)
        self.adding1 = cnn1x1(in_channel=c1, out_channel=c3)
        self.adding2 = cnn1x1(in_channel=c1, out_channel=c4)
        self.adding3 = cnn1x1(in_channel=c1, out_channel=c5)
        
#       time level
        self.cnn = time_cnn(c5, h, c7, c8, bias=True)
        self.maxpool = nn.AdaptiveMaxPool2d((1,1))
        
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(c8, output_shape), nn.Softmax())
        
    def forward(self, data):
        
#       Spatial Level

#       get feature adding
        adding_feature1 = self.adding1(data).permute(0, 2, 3, 1).contiguous()
        adding_feature2 = self.adding2(data).permute(0, 2, 3, 1).contiguous()
        adding_feature3 = self.adding3(data).permute(0, 2, 3, 1).contiguous()
        
#       get adjancency matrix
        adj = self.get_adj(data)

    
        input_feature = data.permute(0, 2, 3, 1).contiguous()
#       run gcn
        input_feature = self.gcn1(input_feature, adj)
        input_feature += adding_feature1
        input_feature = self.gcn2(input_feature, adj)
        input_feature += adding_feature2
        input_feature = self.gcn3(input_feature, adj)
        input_feature += adding_feature3
        
        
#       Time Level
        output_feature = self.cnn(input_feature)
        output_feature = self.maxpool(output_feature)
        output = self.classifier(output_feature)
        
        return output


class cnn1x1(nn.Module):
    def __init__(self, in_channel, out_channel, bias = True):
        super(cnn1x1, self).__init__()
        self.cnn = nn.Conv2d(in_channel, out_channel, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.cnn(x)
        return x
    
    
class compute_adj(nn.Module):
    def __init__(self, in_channel=64*3, hidden_channel=64*3, bias = False):
        super(compute_adj, self).__init__()
        self.g1 = cnn1x1(in_channel, hidden_channel, bias=bias)
        self.g2 = cnn1x1(in_channel, hidden_channel, bias=bias)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x1):
        g1 = self.g1(x1).permute(0, 2, 3, 1).contiguous()
        g2 = self.g2(x1).permute(0, 2, 1, 3).contiguous()
        g3 = g1.matmul(g2)
        g = self.softmax(g3)
        return g
    
    
    
class gcn_spa(nn.Module):
    def __init__(self, in_channel, out_channel, bias = False):
        super(gcn_spa, self).__init__()
        self.relu = nn.ReLU()
        self.weight = Parameter(torch.FloatTensor(in_channel, out_channel))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        x = adj.matmul(x)
        x = x.matmul(self.weight)
        x = self.relu(x)
        
        return x
    
    
class time_cnn(nn.Module):
    def __init__(self, w=7, h=20, c7=3, c8=3, bias = False):
        super(time_cnn, self).__init__()
        self.maxpool = nn.AdaptiveMaxPool2d((1, w))
        self.cnn1 = nn.Conv2d(1, c7, kernel_size=(3, 1), padding=(1, 0), bias=bias)
        self.bn1 = nn.BatchNorm2d(h)
        self.relu = nn.ReLU()
        self.cnn2 = nn.Conv2d(h, c8, kernel_size=1, bias=bias)
        self.bn2 = nn.BatchNorm2d(c8)
        self.dropout = nn.Dropout2d(0.2)

    def forward(self, x1):
        x1 = self.maxpool(x1).permute(0, 2, 1, 3).contiguous()
        x = self.cnn1(x1).permute(0, 2, 1, 3).contiguous()
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.cnn2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x

    
if __name__ == '__main__':
    import numpy as np
    batch_size = 10
    h = 20
    w = 6
    test_data = torch.FloatTensor(np.random.randn(batch_size, 1, h, w))
    test_data[0][0][0][0] = 5
    
    gcn = MyGCN(feature_length=h, feature_nums=6, channel_hidden=7, output_shape=2, debug=True)
    
    print(gcn(test_data))
    
    
    
        
        
        
        
    
    