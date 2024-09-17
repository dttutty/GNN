

def train(model, data, train_idx, optimizer, loss_fn):
    model.train()
    loss = 0
    
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    #train_idx is an array of indices of the training data
    pred = out[train_idx]
    
    loss = loss_fn(pred, data.y)
    loss.backward()
    optimizer.step()
    return loss.item()