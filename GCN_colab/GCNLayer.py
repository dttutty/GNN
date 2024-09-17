import torch
import torch.nn as nn
from typing import List


class GCNLayer(nn.Module):
    def __init__(self, 
                 graph_L: torch.Tensor, 
                 in_features: int, 
                 out_features: int, 
                 max_deg: int = 1
        ):
        """
        :param graph_L: the normalized graph laplacian. It is all the information we need to know about the graph
        :param in_features: the number of input features for each node
        :param out_features: the number of output features for each node
        :param max_deg: how many power of the laplacian to consider, i.e. the q in the spacial formula
        """
        super().__init__()
        
        # Each FC is like the alpha_k matrix, with the last one including bias
        self.fc_layers = nn.ModuleList()
        for i in range(max_deg - 1):
            self.fc_layers.append(nn.Linear(in_features, out_features, bias=False))     # q - 1 layers without bias
        self.fc_layers.append(nn.Linear(in_features, out_features, bias=True))          # last one with bias
        
        # Pre-calculate beta_k(L) for every key
        self.laplacians = self.calc_laplacian_functions(graph_L, max_deg)
        
    def calc_laplacian_functions(self, 
                                 L: torch.Tensor, 
                                 max_deg: int
        ) -> List[torch.Tensor]:
        """
        Compute all the powers of L from 1 to max_deg

        :param L: a square matrix
        :param max_deg: number of powers to compute

        :returns: a list of tensors, where the element i is L^{i+1} (i.e. start counting from 1)
        """
        res = [L]
        for _ in range(max_deg-1):
            res.append(torch.mm(res[-1], L))
        return res
        
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Perform one forward step of graph convolution

        :params X: input features maps [vertices, in_features]
        :returns: output features maps [vertices, out_features]
        """
        Z = torch.tensor(0.)
        for k, fc in enumerate(self.fc_layers):
            L = self.laplacians[k]
            LX = torch.mm(L, X)
            Z = fc(LX) + Z
        
        return torch.relu(Z)