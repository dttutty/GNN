


import os
import pandas as pd
import torch
import torch.nn.functional as F
print(torch.__version__)
# The PyG built-in GCNConv
from torch_geometric.nn import GCNConv

import torch_geometric.transforms as T
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
import matplotlib.pyplot as plt
from train_n_test import train, test

# Please do not change the args



dataset = PygNodePropPredDataset(name='ogbn-arxiv', root='Datasets/ogbn-arxiv')

# Extract the graph
data = dataset[0]

# 获取节点数量
num_nodes = data.num_nodes

# 获取边索引，edge_index 是 (2, num_edges) 的形状，表示每一条边的两个端点
edge_index = data.edge_index

adj_t = torch.sparse_coo_tensor(edge_index, torch.ones(edge_index.shape[1]), (num_nodes, num_nodes))

# data.adj_t = data.adj_t.to_symmetric()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


data = data.to(device)
split_idx = dataset.get_idx_split()
train_idx = split_idx['train'].to(device)

from GCN import GCN

args = {
    'device': device,
    'num_layers': 3,
    'hidden_dim': 256,
    'dropout': 0.5,
    'lr': 0.01,
    'epochs': 1000,
}


model = GCN(data.num_features, args['hidden_dim'],
            dataset.num_classes, args['num_layers'],
            args['dropout']).to(device)
evaluator = Evaluator(name='ogbn-arxiv')

# Please do not change these args
# Training should take <10min using GPU runtime
import copy
# reset the parameters to initial random value
model.reset_parameters()

optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])
loss_fn = F.nll_loss

best_model = None
best_valid_acc = 0
losses = []
test_acces = []
for epoch in range(1, 1 + args["epochs"]):
    loss = train(model, data, train_idx, optimizer, loss_fn)
    result = test(model, data, split_idx, evaluator)
    train_acc, valid_acc, test_acc = result
    losses.append(loss)
    test_acces.append(test_acc)
    if valid_acc > best_valid_acc:
        best_valid_acc = valid_acc
        best_model = copy.deepcopy(model)
    print(f'Epoch: {epoch:02d}, '
        f'Loss: {loss:.4f}, '
        f'Train: {100 * train_acc:.2f}%, '
        f'Valid: {100 * valid_acc:.2f}% '
        f'Test: {100 * test_acc:.2f}%')
    
plt.figure()
plt.plot(range(1, 1 + args["epochs"]), losses, label="Train Loss")
plt.plot(range(1, 1 + args["epochs"]), test_acces, label="Test Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss Curve")
plt.legend()
plt.show()



best_result = test(best_model, data, split_idx, evaluator, save_model_result=True)
train_acc, valid_acc, test_acc = best_result
print(f'Best model: '
      f'Train: {100 * train_acc:.2f}%, '
      f'Valid: {100 * valid_acc:.2f}% '
      f'Test: {100 * test_acc:.2f}%')