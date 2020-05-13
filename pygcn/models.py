"""
https://github.com/tkipf/pygcn
"""
import torch.nn as nn
import torch.nn.functional as F
from pygcn.layers import GraphConvolution


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nlayers, nclass, dropout):
        super(GCN, self).__init__()
        assert nlayers >= 2
        self.params = {
            'nfeat': nfeat,
            'nhid': nhid,
            'nlayers': nlayers,
            'nclass': nclass,
            'dropout': dropout
        }
        self.nfeat = nfeat
        self.nhid = nhid
        self.nclass = nclass 
        self.nlayers = nlayers
        self.layers = []
        self.layers.append(GraphConvolution(nfeat, nhid))
        for _ in range(self.nlayers - 2):
            self.layers.append(GraphConvolution(nhid, nhid))
        self.layers.append(GraphConvolution(nhid, nclass))
        self.dropout = dropout
        self.layers = nn.ModuleList(self.layers)

    def forward(self, x, adj):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x, adj))
            x = F.dropout(x, self.dropout, training=self.training)
        x = self.layers[-1](x, adj)
        return x 
