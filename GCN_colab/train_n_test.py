
def train(model, data, train_idx, optimizer, loss_fn):
    model.train()
    loss = 0
    
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    #train_idx is an array of indices of the training data
    pred = out[train_idx]
    label = data.y[train_idx].squeeze()
    
    loss = loss_fn(pred, label)
    loss.backward()
    optimizer.step()
    return loss.item()

def test(model, data, split_idx, evaluator, save_model_result=False):
    
    model.eval()
    out = None
    
    out = model(data.x, data.edge_index)
    
    
    y_pred = out.argmax(dim=-1, keepdim=True)
    train_acc = evaluator.eval({
        'y_true': data.y[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': data.y[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': data.y[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']
    
    if save_model_result:
        print("Saving model result")
        
        data = {}
        data['y_true'] = y_pred.view(-1).detach().cpu().numpy()
        
        
        df = pd.DataFrame(data)
        df.to_csv('model_result.csv', sep=',', index=False)
        
    return train_acc, valid_acc, test_acc