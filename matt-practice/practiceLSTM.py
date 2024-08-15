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

from models import MLP, LeNet, actionLSTM, actionGRU
from utils import create_train_test_split

device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
# For data collection, change to True if want to evaluate output 
verbose = False

if torch.cuda.is_available():
    torch.cuda.empty_cache()

# seq_size = how long is each sequence, start_pred = when to start predicting thumbstick movement
seq_size = 30
batch_size = 1
start_pred = 16
epochs = 40
iter_val = 15
img_size = 64
learning_rate = 0.001

game_name = 'Barbie'
dir = r"../Vrnet"

# Create train test split
path_map, train_loader, test_loader = create_train_test_split(game_name, dir, device, seq_size=seq_size, batch_size=batch_size, iter=iter_val)

# Run tensorboard summary writer
if verbose: writer = SummaryWriter(f'runs/{game_name}_init_test2_seq_size_{seq_size}_seqstart_{start_pred}')

# Initialise models
init_conv = LeNet().to(device)
init_lstm = actionLSTM().to(device)
fin_mlp = MLP().to(device)

# Initialise optimiser and loss function
optimizer = torch.optim.Adam(
    set(init_conv.parameters()) | set(init_lstm.parameters())
    | set(fin_mlp.parameters()), lr=learning_rate)
criterion = torch.nn.MSELoss()

def train(loader, optimizer, criterion):
    init_lstm.train()
    init_conv.train()
    fin_mlp.train()
    
    total_loss = []
    preds = torch.empty(0)

    prog_bar = tqdm.tqdm(range(len(loader)))
    for batch in loader:
        optimizer.zero_grad()
        # Need to initialise the hidden state for LSTM
        h0 = torch.empty((batch.shape[0], 512)).to(device)
        h0 = torch.nn.init.xavier_uniform_(h0)

        c0 = torch.empty((batch.shape[0], 512)).to(device)
        c0 = torch.nn.init.xavier_uniform_(h0)

        losses = torch.empty(0).to(device)

        # Iterate through all frames in sequence
        for seq in range(batch.shape[1]-1):
            # (image_ind, path, T1, T2, T3, T4)
            indices = batch[:, seq, 0]
            path = batch[:, seq, 1]
            path = [path_map[int(i)] for i in path]
            
            # Reads and encodes the image
            image_t = []
            for i, img_ind in enumerate(indices):
                image = cv2.imread(f'{path[i]}/video/{int(img_ind)}.jpg')
                image = cv2.resize(image, (img_size, img_size))
                image_t.append(image)

            image_t = np.array(image_t)
            image_t = image_t.transpose(0, 3, 1, 2)
            image_t = torch.Tensor(image_t).to(device)

            image_r = init_conv(image_t)

            # GRU step per image and its associated thumbstick comman
            h0, c0 = init_lstm(image_r, batch[:, seq, 2:], h0, c0)

            # Final prediction for each frame 
            fin = fin_mlp(h0)

            # Will only start prediction after certain number of frames
            if seq >= start_pred:
                y = batch[:, seq + 1, 2:]

                # Loss calculation and appending to total loss
                loss = criterion(fin, y)
                losses = torch.cat((losses, loss.reshape(1)))

                preds = torch.cat((preds, y.cpu()))

        # Loss calculation, gradient calculation, then backprop
        losses = torch.mean(losses)
        losses.backward()
        optimizer.step()

        total_loss.append(losses)

        prog_bar.update(1)
    prog_bar.close()

    # This is just for visualising the data imbalance (fucky dont use)
    # u, c =torch.unique(preds, return_counts = True, dim=0)
    # u = [str(x) for x in u.tolist()]
    
    # if verbose: writer.add_histogram('Training values', values=c, bins=u)
    # print(u)
    # c = sorted(c, reverse=True)
    # print(c[0], sum(c[1:]))

    # plt.bar(u, c)
    # plt.xticks(rotation=90)
    # plt.show()
    
    # Returns evaluation scores
    return sum(total_loss) / len(loader), total_loss

# Test is very similar to training
# Instead I use RMSE and not MSE
# I also used torch no grad to be space efficient
def test(loader, criterion):
    init_lstm.eval()
    init_conv.eval()
    fin_mlp.eval()

    rmses = torch.empty(0)
    preds = torch.empty(0)

    prog_bar = tqdm.tqdm(range(len(loader)))

    with torch.no_grad():
        for batch in loader:
            h0 = torch.ones((batch.shape[0], 512)).to(device)
            h0 = torch.nn.init.xavier_uniform_(h0)
            
            c0 = torch.empty((batch.shape[0], 512)).to(device)
            c0 = torch.nn.init.xavier_uniform_(h0)

            losses = torch.empty(0).to(device)

            for seq in range(batch.shape[1]-1):
                # (image_ind, path, T1, T2, T3, T4)
                indices = batch[:, seq, 0]
                path = batch[:, seq, 1]
                path = [path_map[int(i)] for i in path]
                
                image_t = []
                for i, img_ind in enumerate(indices):
                    image = cv2.imread(f'{path[i]}/video/{int(img_ind)}.jpg')
                    image = cv2.resize(image, (64, 64))
                    image_t.append(image)

                image_t = np.array(image_t)
                image_t = image_t.transpose(0, 3, 1, 2)
                image_t = torch.Tensor(image_t).to(device)

                image_r = init_conv(image_t)
                h0, c0 = init_lstm(image_r, batch[:, seq, 2:], h0, c0)

                fin = fin_mlp(h0)

                if seq >= start_pred:
                    y = batch[:, seq + 1, 2:]

                    preds = torch.cat((preds, y.cpu()))
                    loss = criterion(fin, y)
                    losses = torch.cat((losses, loss.reshape(1)))
            
            losses = torch.mean(losses)
            rmses = torch.cat((rmses, losses.reshape(1).cpu()))

            prog_bar.update(1)
    prog_bar.close()

    # u, c = torch.unique(preds, return_counts = True, dim=0)
    # u = [str(x) for x in u.tolist()]
    # if verbose: writer.add_histogram('Testing values', values=c, bins=u)
    # print(u)
    # c = sorted(c, reverse=True)
    # print(c[0], sum(c[1:]))
    
    # plt.bar(u, c)
    # plt.xticks(rotation=90)
    # plt.show()

    return sum(rmses) / len(loader), sum(torch.sqrt(rmses)) / len(loader)

# Epoch train + testing
for epoch in range(1, epochs+1):
    loss, loss_list = train(train_loader, optimizer, criterion)
    test_mse, test_rmse = test(test_loader, criterion)

    # Only add this if val data is available
    # val_rmse, val_ap, val_auc = test(val_loader)
    # print(f'Val AP: {val_ap:.4f}, Val AUC: {val_auc:.4f}')

    print(f'Epoch: {epoch:02d}, Train Loss: {loss:.4f}, Test MSE: {test_mse:.4f}, Test RMSE: {test_rmse:.4f}')

    if verbose:
        writer.add_scalar('training_loss', loss, epoch)
        writer.add_scalar('test_mse', test_mse, epoch)
        writer.add_scalar('test_rmse', test_rmse, epoch)

if verbose: writer.close()