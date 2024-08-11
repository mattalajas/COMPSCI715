from ast import literal_eval

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import tqdm
from sklearn.metrics import (average_precision_score, roc_auc_score,
                             root_mean_squared_error)
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

from models import MLP, ConvBasic, actionGRU

device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
verbose = False

if torch.cuda.is_available():
    torch.cuda.empty_cache()

earth_path = 'Vrnet/files/earth_data_file.csv'
wild_path = 'Vrnet/files/wild_data_file.csv'

earth_gym = pd.read_csv(earth_path)
wild_quest = pd.read_csv(wild_path, nrows=2500)

earth_fin_gym = pd.DataFrame(data = {'frame': earth_gym['frame'], 'thumbstick': earth_gym['Thumbstick']})
wild_fin_gym = pd.DataFrame(data = {'frame': wild_quest['frame'], 'thumbstick': wild_quest['Thumbstick']})

earth_fin_gym.dropna(how='any', inplace=True)
wild_fin_gym.dropna(how='any', inplace=True)

earth_fin_gym['thumbstick'] = earth_fin_gym['thumbstick'].apply(lambda x : literal_eval(str(x)))
wild_fin_gym['thumbstick'] = wild_fin_gym['thumbstick'].apply(lambda x : literal_eval(str(x)))

earth_fin_gym[['T1', 'T2', 'T3', 'T4']] = pd.DataFrame(earth_fin_gym['thumbstick'].to_list(), index = earth_fin_gym.index)
earth_fin_gym.drop(columns=['thumbstick'], inplace=True)

wild_fin_gym[['T1', 'T2', 'T3', 'T4']] = pd.DataFrame(wild_fin_gym['thumbstick'].to_list(), index = wild_fin_gym.index)
wild_fin_gym.drop(columns=['thumbstick'], inplace=True)

init_conv = ConvBasic().to(device)
init_gru = actionGRU().to(device)
fin_mlp = MLP().to(device)

optimizer = torch.optim.Adam(
    set(init_conv.parameters()) | set(init_gru.parameters())
    | set(fin_mlp.parameters()), lr=0.0001)
criterion = torch.nn.MSELoss()

seq_size = 100
batch_size = 2

earth_fin_gym_t = torch.Tensor(earth_fin_gym.values).to(device)
earth_fin_gym_t = torch.split(earth_fin_gym_t, seq_size)

wild_fin_gym_t = torch.Tensor(wild_fin_gym.values).to(device)
wild_fin_gym_t = torch.split(wild_fin_gym_t, seq_size)

if earth_fin_gym_t[-1].shape[0] != seq_size:
    earth_fin_gym_t = earth_fin_gym_t[:-1]

if wild_fin_gym_t[-1].shape[0] != seq_size:
    wild_fin_gym_t = wild_fin_gym_t[:-1]

train_loader = DataLoader(earth_fin_gym_t, batch_size=batch_size, shuffle=False)
train_ind = earth_fin_gym['frame']

test_loader = DataLoader(wild_fin_gym_t, batch_size=batch_size, shuffle=False)
test_ind = wild_fin_gym['frame']

if verbose: writer = SummaryWriter(f'runs/init_test_{seq_size}')

def train(loader, optimizer, criterion):
    init_gru.train()
    init_conv.train()
    init_gru.train()
    
    total_loss = []
    preds = torch.empty(0)

    prog_bar = tqdm.tqdm(range(len(loader)))
    for batch in loader:
        optimizer.zero_grad()
        h0 = torch.ones((batch.shape[0], 512)).to(device)

        for seq in range(batch.shape[1]-1):
            # (100, 5)
            indices = batch[:, seq, 0]
            
            image_t = [cv2.imread(f'Vrnet/files/Earth_gym/{int(img_ind + 10)}.jpg') for img_ind in indices]
            image_t = np.array(image_t)
            image_t = image_t.transpose(0, 3, 1, 2)
            image_t = torch.Tensor(image_t).to(device)

            image_r = init_conv(image_t)
            h0 = init_gru(image_r, batch[:, seq, 1:], h0)

        fin = fin_mlp(h0)
        y = batch[:, -1, 1:]
        loss = criterion(fin, y)
        loss.backward()
        optimizer.step()

        preds = torch.cat((preds, y.cpu()))
        total_loss.append(loss)

        prog_bar.update(1)
    prog_bar.close()


    u, c =torch.unique(preds, return_counts = True, dim=0)
    u = [str(x) for x in u.tolist()]
    
    if verbose: writer.add_histogram('Training values', values=c, bins=u)
    print(u)
    print(c)
    plt.bar(u, c)
    plt.show()
    
    return sum(total_loss) / len(loader), total_loss

def test(loader, criterion):
    init_gru.eval()
    init_conv.eval()
    init_gru.eval()

    rmses = []
    preds = torch.empty(0)

    prog_bar = tqdm.tqdm(range(len(loader)))

    with torch.no_grad():
        for batch in loader:
            h0 = torch.ones((batch.shape[0], 512)).to(device)

            for seq in range(batch.shape[1]-1):
                indices = batch[:, seq, 0]
                
                image_t = [cv2.imread(f'Vrnet/files/Wild_quest/{int(img_ind + 10)}.jpg') for img_ind in indices]
                image_t = np.array(image_t)
                image_t = image_t.transpose(0, 3, 1, 2)
                image_t = torch.Tensor(image_t).to(device)

                image_r = init_conv(image_t)
                h0 = init_gru(image_r, batch[:, seq, 1:], h0)

            fin = fin_mlp(h0)
            y = batch[:, -1, 1:]

            preds = torch.cat((preds, y.cpu()))
            rmses.append(torch.sqrt(criterion(fin, y)))

            prog_bar.update(1)
    prog_bar.close()

    u, c = torch.unique(preds, return_counts = True, dim=0)
    u = [str(x) for x in u.tolist()]
    if verbose: writer.add_histogram('Testing values', values=c, bins=u)
    print(u)
    print(c)
    
    plt.bar(u, c)
    plt.show()

    return sum(rmses) / len(loader)

for epoch in range(1, 151):
    loss, loss_list = train(train_loader, optimizer, criterion)
    test_rmse = test(test_loader, criterion)

    # val_rmse, val_ap, val_auc = test(val_loader)
    # print(f'Val AP: {val_ap:.4f}, Val AUC: {val_auc:.4f}')

    print(f'Epoch: {epoch:02d}, Train Loss: {loss:.4f}, Test AP: {test_rmse:.4f}')

    if verbose:
        writer.add_scalar('training_loss', loss, epoch)
        # writer.add_scalar('val_rmse', val_rmse, epoch)
        # writer.add_scalar('val_ap', val_ap, epoch)
        # writer.add_scalar('val_auc', val_auc, epoch)
        writer.add_scalar('test_rmse', test_rmse, epoch)
        # writer.add_scalar('test_ap', test_ap, epoch)
        # writer.add_scalar('test_auc', test_auc, epoch)

if verbose: writer.close()