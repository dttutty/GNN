
from typing import Any, Dict, List
from typing import Optional, Union

from torch import Tensor
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.aggr import Aggregation
from torch_geometric import nn
import torch_scatter
from torch.nn import functional


class MySage(MessagePassing):
    #propagate会自动调用message(), aggregate(), update()
    #这里面的update()是原生的，目的是为了返回embedding
    def __init__(self, in_channel, out_channel, normalize = True, bias = False, **kwargs):
        super(MySage, self).__init__(**kwargs)
        
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.normalize = normalize
        self.bias = bias
        
        self.linear_neighbor = nn.Linear(in_channel, out_channel)
        self.linear_center = nn.Linear(in_channel, out_channel)
        
        self.reset_parameters()#初始化权重和bias
    
    def reset_parameters(self):
        self.linear_center.reset_parameters()
        self.linear_center.reset_parameters()
    
    def forward(self, x, edge_index) -> Any:
        middle = self.propagate(edge_index, x=(x,x), size=None)
        x = self.linear_center(x) + middle
        
        if self.normalize:
            x = functional.normalize(x)
        return x
    
    def message(self, x_j):
        out = self.linear_neighbor(x_j)
        return out
    
    def aggregate(self, inputs, index):
        dim = self.node_dim
        out = torch_scatter.scatter(inputs, index, dim = dim, reduce='mean')
        return out
    

    
    