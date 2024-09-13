#git config user.name "Lei Zhao"

import argparse
import os
import time
import torch_geometric
import torch
from torch_geometric import datasets
from GCNConv import GCNConv



def train(args):
    #convert the arguments to txt file
    arg_str = ' '.join([f'--{k} {v}' for k, v in vars(args).items()])
    
    #load the dataset
    if args.dataset == 'Cora':
        dataset = datasets.Planetoid(root='~/Projects/GNN/Datasets/Cora', name='Cora')
    elif args.dataset == 'CiteSeer':
        dataset = datasets.Planetoid(root='~/Projects/GNN/Datasets/CiteSeer', name='CiteSeer')
    elif args.dataset == 'PubMed':
        dataset = datasets.Planetoid(root='~/Projects/GNN/Datasets/PubMed', name='PubMed')
    else:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))
    
    
    result_dir = 'results/{}'.format(args.dataset)
    #result file name is the current time
    result_file = 'results/{}.txt'.format(time.strftime('%Y%m%d-%H%M%S'))
    #add the arg_str to the result file
    os.makedirs(result_dir, exist_ok=True)
    with open(result_file, 'w') as f:
        f.write(arg_str + '\n')
            

        
    #load the dataset
    train_loader = torch_geometric.data.DataLoader(dataset, batch_size=1, shuffle=True)
    val_loader = torch_geometric.data.DataLoader(dataset, batch_size=1, shuffle=False)
    
    #initialize the model
    #number_of_features equals to the node word bag size of Cora dataset
    num_features = dataset.num_node_features
    model = GCN(dataset.num_features, dataset.num_classes, args.num_layers, args.hidden, args.dropout).to(args.device)
    #begin the training
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    for epoch in range(args.epochs):
        total_loss = 0
        for data in train_loader:
            data = data.to(args.device)
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)
            loss = torch.nn.functional.cross_entropy(out, data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        total_loss /= len(train_loader)
        print('Epoch {}, Loss: {}'.format(epoch, total_loss))
        with open(result_file, 'a') as f:
            f.write('Epoch {}, Loss: {}\n'.format(epoch, total_loss))
        
        if epoch % 10 == 0:
            val_acc = test(model, val_loader, args.device)
            print('Val Acc: {}'.format(val_acc))
            with open(result_file, 'a') as f:
                f.write('Val Acc: {}\n'.format(val_acc))
                


def test(model, loader, device):
    model.eval()
    correct = 0
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index)
        pred = out.max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset)
    
    


class GCN(torch.nn.Module):
    def __init__(self, num_features, num_classes, num_layers, hidden, dropout):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden)
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden, hidden))
        self.conv2 = GCNConv(hidden, num_classes)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)
        for conv in self.convs:
            x = conv(x, edge_index)
            x = torch.relu(x)
            x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x

if __name__ == '__main__':
    #receive the arguments from the command line and set the default values
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Cora')
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--hidden', type=int, default=16)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    train(args)