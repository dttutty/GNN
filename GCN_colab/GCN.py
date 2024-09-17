import torch
import torch.nn as nn
from typing import List
from torch_geometric import nn
class GCN(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers, dropout, return_embeds=False):
        super(GCN, self).__init__()
        
        self.convs = None
        self.bns = None
        self.softmax = None
        
        self.convs = torch.nn.ModuleList()
        self.convs.append(nn.GCNConv(in_dim, hidden_dim))
        for i in range(num_layers-2):
                self.convs.append(nn.GCNConv(hidden_dim, hidden_dim))
        self.convs.append(nn.GCNConv(hidden_dim, out_dim))
        
        self.bns = torch.nn.ModuleList()
        for i in range(num_layers-1):
            self.bns.append(nn.BatchNorm(hidden_dim))
        
        self.dropout = dropout
        self.return_embeds = return_embeds
    
    
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
            
    def forward(self, x, adj_t):
        out = None
        
        for i in range(len(self.convs)-1):
            x = self.convs[i](x, adj_t)
            x = self.bns[i](x)
            x = torch.nn.functional.relu(x)
            x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)
        x=self.convs[-1](x, adj_t)
        out = torch.nn.functional.log_softmax(x, dim=1)
        
        return out
        