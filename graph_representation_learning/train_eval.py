import sys
import time

import torch
import torch.nn.functional as F
from torch import tensor
from torch.optim import Adam
from sklearn.model_selection import StratifiedKFold
from torch_geometric.data import DataLoader, DenseDataLoader as DenseLoader
from sklearn.model_selection import train_test_split
from utils import print_weights
import math
from tqdm import tqdm
import pickle

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def single_train_test(train_dataset,
                      test_dataset,
                      model_func,
                      epochs,
                      batch_size,
                      lr,
                      lr_decay_factor,
                      lr_decay_step_size,
                      weight_decay,
                      epoch_select,
                      with_eval_mode=True):
    assert epoch_select in ['test_last', 'test_max'], epoch_select

    model = model_func(train_dataset).to(device)
    print_weights(model)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)

    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)
    train_accs, test_accs = [], []
    t_start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        train_loss, train_acc = train(
            model, optimizer, train_loader, device)
        train_accs.append(train_acc)
        test_accs.append(eval_acc(model, test_loader, device, with_eval_mode))

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        print('Epoch: {:03d}, Train Acc: {:.4f}, Test Acc: {:.4f}'.format(
            epoch, train_accs[-1], test_accs[-1]))
        sys.stdout.flush()

        if epoch % lr_decay_step_size == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_decay_factor * param_group['lr']

    t_end = time.perf_counter()
    duration = t_end - t_start
    
    
    if epoch_select == 'test_max':
        train_acc = max(train_accs)
        test_acc = max(test_accs)
    else:
        train_acc = train_accs[-1]
        test_acc = test_accs[-1]

    return train_acc, test_acc, duration


def cross_validation_with_val_set(dataset,
                                  model_func,
                                  folds,
                                  epochs,
                                  batch_size,
                                  lr,
                                  lr_decay_factor,
                                  lr_decay_step_size,
                                  weight_decay,
                                  epoch_select,
                                  with_eval_mode=True,
                                  logger=None):
    assert epoch_select in ['val_max', 'test_max'], epoch_select
    print('devices: ',device)
    val_losses, train_accs, test_accs, durations = [], [], [], []
    for fold, (train_idx, test_idx, val_idx) in enumerate(
            zip(*k_fold(dataset, folds, epoch_select))):
        
        all_idx = [i for i in range(len(dataset)) ]
        
        all_loader = DataLoader(dataset, batch_size, shuffle=False)

        all_num_size = math.floor(len(all_idx)/batch_size)
        all_batch_graph_id = []
        for i in range(all_num_size):
            down = batch_size * i
            up = batch_size * (i+1)
            all_batch_graph_id.append(all_idx[down:up])
        all_batch_graph_id.append(all_idx[up:])
                
        train_dataset = dataset[train_idx]
        test_dataset = dataset[test_idx]
        val_dataset = dataset[val_idx]

        train_num_size = math.floor(len(train_idx)/batch_size)
        train_batch_graph_id = []
        for i in range(train_num_size):
            down = batch_size * i
            up = batch_size * (i+1)
            train_batch_graph_id.append(train_idx[down:up])
        train_batch_graph_id.append(train_idx[up:])   

        test_num_size = math.floor(len(test_idx)/batch_size)
        test_batch_graph_id = []
        for i in range(test_num_size):
            down = batch_size * i
            up = batch_size * (i+1)
            test_batch_graph_id.append(test_idx[down:up])
        test_batch_graph_id.append(test_idx[up:])

        val_num_size = math.floor(len(val_idx)/batch_size)
        val_batch_graph_id = []
        for i in range(val_num_size):
            down = batch_size * i
            up = batch_size * (i+1)
            val_batch_graph_id.append(val_idx[down:up])
        val_batch_graph_id.append(val_idx[up:]) 
        
        
        train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

        model = model_func(dataset).to(device)
        if fold == 0:
            print_weights(model)
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_start = time.perf_counter()
        epochtestacc = []
        for epoch in range(1, epochs + 1):
            train_loss, train_acc,train_emb = train(
                model, optimizer, train_loader, device)
            train_accs.append(train_acc)

            valacc,val_emb = eval_loss(
                model, val_loader, device, with_eval_mode)

            val_losses.append(valacc)        
        
            testacc,test_emb,pred_list = eval_acc(
                model, test_loader, device, with_eval_mode)

            test_accs.append(testacc)
            epochtestacc.append(testacc)
            eval_info = {
                'fold': fold,
                'epoch': epoch,
                'train_loss': train_loss,
                'train_acc': train_accs[-1],
                'val_loss': val_losses[-1],
                'test_acc': test_accs[-1],
            }

            if logger is not None:
                logger(eval_info)

                
            
            
            if epoch % lr_decay_step_size == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_decay_factor * param_group['lr']

                    
        allacc,all_emb,pred_list = eval_acc(
                model, all_loader, device, with_eval_mode)            
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        t_end = time.perf_counter()
        durations.append(t_end - t_start)

        torch.save(model, './modelsaved/trainmodel_'+str(fold)+'.pkl')
        
        
    print('devices: ',device)    
    duration = tensor(durations)
    train_acc, test_acc = tensor(train_accs), tensor(test_accs)
    val_loss = tensor(val_losses)
    train_acc = train_acc.view(folds, epochs)
    test_acc = test_acc.view(folds, epochs)
    val_loss = val_loss.view(folds, epochs)
    if epoch_select == 'test_max':  # take epoch that yields best test results.
        _, selected_epoch = test_acc.mean(dim=0).max(dim=0)
        selected_epoch = selected_epoch.repeat(folds)
    else:  # take epoch that yields min val loss for each fold individually.
        _, selected_epoch = val_loss.min(dim=1)
    test_acc = test_acc[torch.arange(folds, dtype=torch.long), selected_epoch]
    train_acc_mean = train_acc[:, -1].mean().item()
    test_acc_mean = test_acc.mean().item()
    test_acc_std = test_acc.std().item()
    duration_mean = duration.mean().item()

    print('Train Acc: {:.4f}, Test Acc: {:.3f} ± {:.3f}, Duration: {:.3f}'.
          format(train_acc_mean, test_acc_mean, test_acc_std, duration_mean))
    sys.stdout.flush()

    return train_acc_mean, test_acc_mean, test_acc_std, duration_mean,train_batch_graph_id,test_batch_graph_id,val_batch_graph_id,train_emb,test_emb,val_emb


def k_fold(dataset, folds, epoch_select):
    skf = StratifiedKFold(folds, shuffle=True, random_state=12345)

    test_indices, train_indices = [], []
    for _, idx in skf.split(torch.zeros(len(dataset)), dataset.data.y):
        test_indices.append(torch.from_numpy(idx))

    if epoch_select == 'test_max':
        val_indices = [test_indices[i] for i in range(folds)]
    else:
        val_indices = [test_indices[i - 1] for i in range(folds)]

    for i in range(folds):
        train_mask = torch.ones(len(dataset), dtype=torch.uint8)
        train_mask[test_indices[i].long()] = 0
        train_mask[val_indices[i].long()] = 0
        train_indices.append(train_mask.nonzero().view(-1))

    return train_indices, test_indices, val_indices


def num_graphs(data):
    if data.batch is not None:
        return data.num_graphs
    else:
        return data.x.size(0)


def train(model, optimizer, loader, device):
    model.train()
    
    total_loss = 0
    correct = 0
    readout_list = []
    for data in loader:
        optimizer.zero_grad()
        data = data.to(device)
        out,readout = model(data)
        readout_list.append(readout)
        loss = F.nll_loss(out, data.y.view(-1))
        pred = out.max(1)[1]

        correct += pred.eq(data.y.view(-1)).sum().item()
        loss.backward()
        total_loss += loss.item() * num_graphs(data)
        optimizer.step()
    return total_loss / len(loader.dataset), correct / len(loader.dataset),readout_list 


def eval_acc(model, loader, device, with_eval_mode):
    if with_eval_mode:
        model.eval()
    readout_list = []
    correct = 0
    pred_list = []
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            out,readout = model(data)
            pred = out.max(1)[1]
            pred_list.append(pred.tolist())
        readout_list.append(readout)
        correct += pred.eq(data.y.view(-1)).sum().item()
    return correct / len(loader.dataset),readout_list, pred_list


def eval_loss(model, loader, device, with_eval_mode):
    if with_eval_mode:
        model.eval()
    readout_list  = []
    loss = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            out,readout = model(data)
        readout_list.append(readout)
        loss += F.nll_loss(out, data.y.view(-1), reduction='sum').item()
    return loss / len(loader.dataset),readout_list 
