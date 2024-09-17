import torch
import os
import torch_geometric
print("PyTorch has version {}".format(torch.__version__))
print("torch_geometric has version {}".format(torch_geometric.__version__))

from torch_geometric.datasets import TUDataset
import fsspec
print(fsspec.__version__)#torch_geometric-2.5.2 + fsspec-2023.5.0 works

root = 'Datasets/enzymes'
name = 'ENZYMES'
# The ENZYMES dataset
pyg_dataset= TUDataset(root, name)
# You will find that there are 600 graphs in this dataset
print(pyg_dataset)

def get_num_classes(pyg_dataset):
    num_classes = pyg_dataset.num_classes

    return num_classes

def get_num_features(pyg_dataset):
    num_features = pyg_dataset.num_features

    return num_features


num_classes = get_num_classes(pyg_dataset)
num_features = get_num_features(pyg_dataset)
print("{} dataset has {} classes".format(name, num_classes))
print("{} dataset has {} features".format(name, num_features))

def get_graph_class(pyg_dataset, idx):

    label = pyg_dataset[idx].y

    return label


graph_0 = pyg_dataset[0]
print(graph_0)
idx = 100
label = get_graph_class(pyg_dataset, idx)
print('Graph with index {} has label {}'.format(idx, label))


def get_graph_num_edges(pyg_dataset, idx):
    num_edges = pyg_dataset[idx].edge_index.size(1)
    return num_edges

idx = 200
num_edges = get_graph_num_edges(pyg_dataset, idx)
print('Graph with index {} has {} edges'.format(idx, num_edges))



import torch_geometric.transforms as T
from ogb.nodeproppred import PygNodePropPredDataset

dataset_name = 'ogbn-arxiv'
# Load the dataset and transform it to sparse tensor
dataset = PygNodePropPredDataset(name=dataset_name, root='Datasets/ogbn-arxiv', transform=T.ToSparseTensor())
print('The {} dataset has {} graph'.format(dataset_name, len(dataset)))

# Extract the graph
data = dataset[0]
print(data)

def graph_num_features(data):
    num_features = 0
    num_features = data.num_features

    return num_features


num_features = graph_num_features(data)
print('The graph has {} features'.format(num_features))


print('end')

import torch
import pandas as pd
import torch.nn.functional as F
print(torch.__version__)

# The PyG built-in GCNConv
from torch_geometric.nn import GCNConv

import torch_geometric.transforms as T
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator


