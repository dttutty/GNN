from __future__ import print_function, division

import igraph
from typing import Mapping, Union, Optional
from pathlib import Path

import networkx as nx
import numpy as np
import argparse
import torch
import torch.nn as nn
from torch.nn import Module

import torch.nn.functional as F
import torch.optim as optim
import plotly.graph_objects as go
import plotly.express as px
import matplotlib
import matplotlib.pyplot as plt
import torchvision
from torchvision import datasets, models, transforms

import os
import pickle
from tqdm.notebook import tqdm
from tqdm import trange
from GCN_colab.useless.GCNLayer import GCNLayer

matplotlib.use('Qt5Agg')  # 或 'TkAgg'
plt.ion()  # 启用交互模式


# The karate dataset is built-in in networkx
import networkx as nx

G = nx.karate_club_graph()
#Ignore Weigh

# Known ids of the instructor, admin and members
ID_INSTR = 0    
ID_ADMIN = 33
ID_MEMBERS = set(G.nodes()) - {ID_ADMIN, ID_INSTR}

print(f'{G.name}: {len(G.nodes)} vertices, {len(G.edges)} edges')
# Input featuers (no information on nodes):
X = torch.eye(G.number_of_nodes())

    
# Create ground-truth labels
# - Assign the label "0" to the "Mr. Hi" community
# - Assign the label "1" to the "Officer" community
labels = [int(not d['club']=='Mr. Hi') for _, d in G.nodes().data()]
labels = torch.tensor(labels, dtype=torch.long)
# Let's check the nodes metadata
for (node_id, node_data), label_id in zip(G.nodes().data(), labels):
    print(f'Node id: {node_id},\tClub: {node_data["club"]},\t\tLabel: {label_id.item()}')
    
    # Adjacency matrix, binary
# A = nx.adj_matrix(G, weight=None)
# convert G to adjacency matrix
G_adj = nx.adjacency_matrix(G)
A_sparse = nx.adjacency_matrix(G, weight=None)
A = A_sparse.toarray()
# A = np.array(A.todense())
# Degree matrix
# count  non-zero elements in each row
non_zero = np.count_nonzero(A, axis=1)
D = np.diag(non_zero)
# dii = np.sum(A, axis=1, keepdims=False)  # sum the columns of theadj
# D = np.diag(dii)
# Laplacian
L = D - A


def test_graph():
    # Symmetric
    (L.transpose() == L).all()

    # Sum of degrees
    np.trace(L) == 2 * G.number_of_edges()

    # Sum of colums/rows is zero
    print(np.sum(L, axis=1))
    print(np.sum(L, axis=0))

    # Compute the eigevanlues and eigenvector
    w, Phi = np.linalg.eigh(L)

    plt.plot(w)
    plt.xlabel(r'$\lambda$')
    plt.title('Spectrum')


    #@title visualization 

    # Plot Fourier basis傅里叶基，还是不懂
    pos = nx.spring_layout(G)
    fig, ax = plt.subplots(4, 4, figsize=(8,6), dpi=150)
    ax = ax.reshape(-1)
    vmin, vmax = np.min(Phi), np.max(Phi)
    for i in range(len(ax)):
        nc = Phi[:,i]
        nx.draw_networkx(G, pos, node_color=nc, with_labels=False, node_size=15, ax=ax[i], width=0.25, cmap=plt.cm.magma, vmin=vmin, vmax=vmax)
        ax[i].axis('off')
        ax[i].set_title(rf'$\lambda_{{{i}}}={w[i]:.2f}$',fontdict=dict(fontsize=8))

    # Adjacency matrix

    I = np.eye(A.shape[0])
    A = A + I

    # Degree matrix (only the diagonal)
    dii = np.sum(A, axis=1, keepdims=False)
    #D = np.diag(dii)

    # Normalized Laplacian
    D_inv_h = np.diag(dii**(-0.5)) #size: n*n
    L =  D_inv_h @ A @ D_inv_h #size: n*n


import torch.nn as nn
from typing import List


    
in_features, out_features = X.shape[1], 2
graph_L = torch.tensor(L, dtype=torch.float)
max_deg = 2
hidden_dim = 5

from torch_geometric.nn import GCNConv

# Stack two GCN layers as our model
gcn2 = nn.Sequential(
    GCNConv(in_features, hidden_dim),
    GCNConv(hidden_dim, out_features),
    
    # GCNLayer(graph_L, in_features, hidden_dim, max_deg),
    # GCNLayer(graph_L, hidden_dim, out_features, max_deg),
    nn.LogSoftmax(dim=1)
)
gcn2

import torch.nn.functional as F
import torch.optim

def train_node_classifier(model, optimizer, X, y, epochs=300):
    y_pred_epochs = []
    for epoch in range(epochs+1):
        y_pred = model(X, )  # Compute on all the graph
        y_pred_epochs.append(y_pred.detach())

        # Semi-supervised: only use labels of the Instructor and Admin nodes
        labelled_idx = [ID_ADMIN, ID_INSTR]
        loss = F.nll_loss(y_pred[labelled_idx], y[labelled_idx])  # loss on only two nodes

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f'Epoch {epoch:2d}, loss={loss.item():.5f}')
    return y_pred_epochs

optimizer = torch.optim.Adam(gcn2.parameters(), lr=0.01)

y_pred_epochs = train_node_classifier(gcn2, optimizer, X, labels)


from sklearn.metrics import classification_report

y_pred = torch.argmax(gcn2(X), dim=1).detach().numpy()
y = labels.numpy()
print(classification_report(y, y_pred, target_names=['I','A']))
print('byebye')