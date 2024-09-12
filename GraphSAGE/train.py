import os
import time

import networkx as nx
import numpy as np
import torch
import torch.optim as optim
from tqdm import trange
import pandas as pd
import copy

from torch_geometric.datasets import TUDataset
from torch_geometric.datasets import Planetoid
from torch_geometric.data import DataLoader

import torch_geometric.nn as pyg_nn

import matplotlib.pyplot as plt
from optimizer_builder import build_optimizer
from GNNStack import GNNStack
from GraphSage import GraphSage
from datetime import datetime

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train(dataset, args):
    torch.set_num_threads(4)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    device = torch.device("cpu")

    print("Node task. test set size:", np.sum(dataset[0]['test_mask'].numpy()))
    test_loader = loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # build model
    model = GNNStack(dataset.num_node_features, args.hidden_dim, dataset.num_classes, args).to(device)
    scheduler, opt = build_optimizer(args, model.parameters())

    # train
    losses = []
    test_accs = []
    best_acc = 0
    best_model = None
    for epoch in trange(args.epochs, desc="Training", unit="Epochs"):
        total_loss = 0
        model.train()
        for batch in loader:
            batch = batch.to(device)
            opt.zero_grad()
            pred = model(batch)
            label = batch.y
            pred = pred[batch.train_mask]
            label = label[batch.train_mask]
            loss = model.loss(pred, label)
            loss.backward()
            opt.step()
            total_loss += loss.item() * batch.num_graphs
        total_loss = total_loss / len(loader.dataset)
        losses.append(total_loss)

        if epoch % 10 == 0:
          test_acc = test(test_loader, model)
          test_accs.append(test_acc)
          if test_acc > best_acc:
            best_acc = test_acc
            best_model = copy.deepcopy(model)
        else:
          test_accs.append(test_accs[-1])

    return test_accs, losses, best_model, best_acc, test_loader

def test(loader, test_model, is_validation=False, save_model_preds=False, model_type=None):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    test_model.eval()

    correct = 0
    # Note that Cora is only one graph!
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            # max(dim=1) returns values, indices tuple; only need indices
            pred = test_model(data).max(dim=1)[1]
            label = data.y

        mask = data.val_mask if is_validation else data.test_mask
        # node classification: only evaluate on nodes in test set
        pred = pred[mask]
        label = label[mask]

        if save_model_preds:
          print ("Saving Model Predictions for Model Type", model_type)

          data = {}
          data['pred'] = pred.view(-1).cpu().detach().numpy()
          data['label'] = label.view(-1).cpu().detach().numpy()

          df = pd.DataFrame(data=data)
          # Save locally as csv
          df.to_csv('CORA-Node-' + model_type + '.csv', sep=',', index=False)

        correct = correct + pred.eq(label).sum().item()

    total = 0
    for data in loader.dataset:
        total += torch.sum(data.val_mask if is_validation else data.test_mask).item()

    return correct / total

class objectview(object):
    def __init__(self, d):
        self.__dict__ = d
        

def save_model_and_result(model, dataset_name, model_type, current_time, losses, test_accs):
    title_text = (
        f"{dataset_name}\n"  # 第一行
        f"max accuracy: {max(test_accs):.4f}\n"  # 第二行，max accuracy
        f"min loss: {min(losses):.4f}"  # 第三行，min loss
    )
    plt.title(title_text)

    plt.plot(losses, label="training loss" + " - " + model_type)
    plt.plot(test_accs, label="test accuracy" + " - " + model_type)
    plt.legend()

    directory = f'{dataset_name}'
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(os.path.join(directory, f'{current_time}'))
    plt.clf()
    
    torch.save(model.state_dict(), os.path.join(directory, f'{current_time}_best_model.pth'))

    
    

if 'IS_GRADESCOPE_ENV' not in os.environ:
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for args in [
        {'model_type': 'GraphSage', 'dataset': 'PubMed', 'num_layers': 2, 'heads': 1, 'batch_size': 32, 'hidden_dim': 32, 'dropout': 0.5, 'epochs': 500, 'opt': 'adam', 'opt_scheduler': 'none', 'opt_restart': 0, 'weight_decay': 5e-3, 'lr': 0.01},
    ]:
        args = objectview(args)
            # Match the dimension.

        if args.dataset == 'Cora':
            dataset = Planetoid(root='/tmp/cora', name='Cora')
        elif args.dataset == 'CiteSeer':
            dataset = Planetoid(root='/tmp/citeseer', name='CiteSeer')
        elif args.dataset == 'PubMed':
            dataset = Planetoid(root='/tmp/pubMed', name='PubMed')
        else:
            raise NotImplementedError("Unknown dataset")
        test_accs, losses, best_model, best_acc, test_loader = train(dataset, args)



        print("Maximum test set accuracy: {0}".format(max(test_accs)))
        print("Minimum loss: {0}".format(min(losses)))
        
        # Run test for our best model to save the predictions!
        # test(test_loader, best_model, is_validation=False, save_model_preds=True, model_type='GraphSage')

        save_model_and_result(best_model, dataset.name, args.model_type, current_time, losses, test_accs)    

        print("Model and result saved!")

