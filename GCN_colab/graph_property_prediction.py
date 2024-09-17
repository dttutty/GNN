from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from torch_geometric.data import DataLoader
from tqdm.notebook import tqdm
import torch
import os
from GCN_Graph import GCN_Graph
from train_n_test_for_graph import train, eval

import matplotlib.pyplot as plt

# Load the dataset
dataset = PygGraphPropPredDataset(name='ogbg-molhiv', root='Datasets')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Device: {}'.format(device))

split_idx = dataset.get_idx_split()

# Check task type
print('Task type: {}'.format(dataset.task_type))


# Load the dataset splits into corresponding dataloaders
# We will train the graph classification task on a batch of 32 graphs
# Shuffle the order of graphs for training set

train_loader = DataLoader(dataset[split_idx["train"]], batch_size=32, shuffle=True, num_workers=1)
valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=32, shuffle=False, num_workers=1)
test_loader = DataLoader(dataset[split_idx["test"]], batch_size=32, shuffle=False, num_workers=1)
  
args = {'device': device,'num_layers': 5,'hidden_dim': 256,'dropout': 0.5,'lr': 0.001,'epochs': 30,}



model = GCN_Graph(args['hidden_dim'],
            dataset.num_tasks, args['num_layers'],
            args['dropout']).to(device)
evaluator = Evaluator(name='ogbg-molhiv')

# Please do not change these args
# Training should take <10min using GPU runtime
import copy

model.reset_parameters()

optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])
loss_fn = torch.nn.BCEWithLogitsLoss()

best_model = None
best_valid_acc = 0
losses = []
acces = []
for epoch in range(1, 1 + args["epochs"]):
    print('Training...')
    loss = train(model, device, train_loader, optimizer, loss_fn)

    print('Evaluating...')
    train_result = eval(model, device, train_loader, evaluator)
    val_result = eval(model, device, valid_loader, evaluator)
    test_result = eval(model, device, test_loader, evaluator)

    train_acc, valid_acc, test_acc = train_result[dataset.eval_metric], val_result[dataset.eval_metric], test_result[dataset.eval_metric]
    losses.append(loss)
    acces.append(test_acc)
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
plt.plot(range(1, 1 + args["epochs"]), acces, label="Test Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss Curve")
plt.legend()
plt.show()

    
train_auroc = eval(best_model, device, train_loader, evaluator)[dataset.eval_metric]
valid_auroc = eval(best_model, device, valid_loader, evaluator, save_model_results=True, save_file="valid")[dataset.eval_metric]
test_auroc  = eval(best_model, device, test_loader, evaluator, save_model_results=True, save_file="test")[dataset.eval_metric]

print(f'Best model: '
    f'Train: {100 * train_auroc:.2f}%, '
    f'Valid: {100 * valid_auroc:.2f}% '
    f'Test: {100 * test_auroc:.2f}%')