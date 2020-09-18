import math
import torch.nn as nn
import torch
import numpy as np
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F

DEBUG = False

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
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

    def forward(self, input, adj):
        support = torch.mm(adj, input)
        # print(torch.min(support))
        # print(torch.max(support))
        output = torch.mm(support, self.weight)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, nfeat, nhid2, nhid1, nhid0, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid2)
        self.bn1=nn.BatchNorm1d(nfeat)
        self.bn2=nn.BatchNorm1d(nhid2)
        self.bn3=nn.BatchNorm1d(nhid1)

        self.gc2 = GraphConvolution(nhid2, nclass)
        # self.gc3 = GraphConvolution(nhid1, nclass)
        # self.gc4 = GraphConvolution(nhid0, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        if DEBUG:
            print('GC1',x)
        # x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        # if DEBUG:
        #     print('GC2',x)
        # x = self.gc3(self.bn3(x), adj)
        # if DEBUG:
        #     print('GC3',x)
        # x = F.dropout(x, self.dropout, training=self.training)
        # if DEBUG:
        #     print('dropout',x)
        # x = F.relu(self.gc4(x, adj))
        # if DEBUG:
        #     print('GC4',x)
        return F.log_softmax(x, dim=1)


def pairwise_distances(x, y=None):
    """
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    """
    x_norm = (x ** 2).sum(1).view(-1, 1)
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y ** 2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)

    return torch.clamp(dist, 0.0, np.inf)


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


# TODO: Normalizing the features