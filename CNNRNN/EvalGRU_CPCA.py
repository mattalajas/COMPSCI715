from ast import literal_eval

import cv2
import copy
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

from aux_task import CPCA
from models import MLP, LeNet, actionGRUdeep, actionGRU
from RNNCNNutils import *
from data_utils_copy import *
from string import Template

cuda_num = 5
device = torch.device('mps' if torch.backends.mps.is_available() else f'cuda:{cuda_num}' if torch.cuda.is_available() else 'cpu')
# For data collection, change to True if want to evaluate output 
verbose = False
save_df = True

if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Main task hyperparams
seq_size = 50
batch_size = 10
start_pred = 20
epochs = 150
iter_val = 10
img_size = 64
main_lr= 0.01
regularisation = 0.00001
dropout = 0.2
rnn_emb = 256

num_outputs= 11

# Aux task hyperparams
hid_size = 256
aux_steps = seq_size - start_pred
sub_rate = 0.1
loss_fac = 0.5
aux_reguarisation = 0
aux_lr = 0.01

train_game_names = ['Barbie', 'Kawaii_Fire_Station', 'Kawaii_Playroom', 'Kawaii_Police_Station']
test_game_names = ['Kawaii_House', 'Kawaii_Daycare']
val_game_names = ['Kawaii_House', 'Kawaii_Daycare']
image_path = Template("/data/ysun209/VR.net/videos/${game_session}/video/${imgind}.jpg")

# Create train test split
val_sessions = DataUtils.read_txt("/data/mala711/COMPSCI715/datasets/final_data_splits/val.txt")
test_sessions = DataUtils.read_txt("/data/mala711/COMPSCI715/datasets/final_data_splits/test.txt")

col_pred = ["thumbstick_left_x", "thumbstick_left_y", "thumbstick_right_x", "thumbstick_right_y", "head_pos_x", "head_pos_y", "head_pos_z", "head_dir_a", "head_dir_b", "head_dir_c", "head_dir_d"]

val_set = MultiGameDataset(val_game_names, val_sessions, cols_to_predict=col_pred)
test_set = MultiGameDataset(test_game_names, test_sessions, cols_to_predict=col_pred) 

# Normalisation
thumbsticks_loc = 6
head_pos_loc = 9

val_set.df[val_set.df.columns[2:thumbsticks_loc]] = (val_set.df[val_set.df.columns[2:thumbsticks_loc]] + 1) / 2
test_set.df[test_set.df.columns[2:thumbsticks_loc]] = (test_set.df[test_set.df.columns[2:thumbsticks_loc]] + 1) / 2

val_set.df[val_set.df.columns[thumbsticks_loc:head_pos_loc]] = (val_set.df[val_set.df.columns[thumbsticks_loc:head_pos_loc]] + 2) / 4
test_set.df[test_set.df.columns[thumbsticks_loc:head_pos_loc]] = (test_set.df[test_set.df.columns[thumbsticks_loc:head_pos_loc]] + 2) / 4

val_set.df[val_set.df.columns[head_pos_loc:]] = (val_set.df[val_set.df.columns[head_pos_loc:]] + 1) / 2
test_set.df[test_set.df.columns[head_pos_loc:]] = (test_set.df[test_set.df.columns[head_pos_loc:]] + 1) / 2

test_path_map, test_loader = filter_dataframe(test_sessions, test_set.df, device, seq_size, batch_size, iter=iter_val)

# Run tensorboard summary writer
common_name = f'GRU_CPCA_train_{train_game_names}_test_{test_game_names}_init_test_seq_size_{seq_size}_seqstart_{start_pred}_iter_{iter_val}_reg_{regularisation}_auxreg_{aux_reguarisation}_lr_{main_lr}_auxlr_{aux_lr}_dropout_{dropout}'
if verbose: writer = SummaryWriter(f'/data/mala711/COMPSCI715/CNNRNN/runs/Eval{common_name}')

# Initialise models
init_conv = LeNet(img_size, hid_size, dropout=dropout).to(device)
init_gru = actionGRU(num_outputs, rnn_emb, hid_size, hid_size, dropout).to(device)
thumb_fin_mlp = MLP(hid_size, 4, dropout).to(device)
headpos_fin_mlp = MLP(hid_size, 3, dropout).to(device)
headdir_fin_mlp = MLP(hid_size, 4, dropout).to(device)
cpca = CPCA(num_outputs, hid_size, aux_steps, sub_rate, loss_fac, dropout, device).to(device)

# Initialise optimiser and loss function
optimizer = torch.optim.Adam([
    {'params': init_conv.parameters()},
    {'params': init_gru.parameters()},
    {'params': thumb_fin_mlp.parameters()},
    {'params': headpos_fin_mlp.parameters()},
    {'params': headdir_fin_mlp.parameters()},
    {'params': cpca.parameters(), 'lr': aux_lr}], lr=main_lr)
# optimizer = torch.optim.Adam([
#     {'params': init_conv.parameters()},
#     {'params': init_gru.parameters()},
#     {'params': cpca.parameters(), 'lr': aux_lr}], lr=main_lr)
# optimizer = torch.optim.Adam([
#     {'params': cpca.parameters(), 'lr': aux_lr}], lr=main_lr)
criterion = torch.nn.MSELoss()

save_path = f'/data/mala711/COMPSCI715/CNNRNN/models/{common_name}.pth'

checkpoint = torch.load(save_path, weights_only=True)
init_conv.load_state_dict(checkpoint['init_conv_state_dict'])
init_gru.load_state_dict(checkpoint['init_gru_state_dict'])
thumb_fin_mlp.load_state_dict(checkpoint['thumb_fin_mlp_state_dict'])
headpos_fin_mlp.load_state_dict(checkpoint['headpos_fin_mlp_state_dict'])
headdir_fin_mlp.load_state_dict(checkpoint['headdir_fin_mlp_state_dict'])
cpca.load_state_dict(checkpoint['cpca_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# Test is very similar to training
# Instead I use RMSE and not MSE
# I also used torch no grad to be space efficient
def test(loader, path_map, criterion):
    init_gru.eval()
    init_conv.eval()
    thumb_fin_mlp.eval()
    headpos_fin_mlp.eval()
    headdir_fin_mlp.eval()
    cpca.eval()

    rmses = torch.empty(0)
    preds = torch.empty(0)

    prog_bar = tqdm.tqdm(range(len(loader)))

    all_actions = []
    all_paths = []
    all_indices = []


    with torch.no_grad():
        for batch in loader:
            h0 = torch.ones((batch.shape[0], hid_size)).to(device)
            h0 = torch.nn.init.xavier_uniform_(h0)

            losses = torch.empty(0).to(device)

            for seq in range(batch.shape[1]-1):
                # (path, image_ind, T1, T2, T3, T4)
                indices = batch[:, seq, 1]
                path = batch[:, seq, 0]
                path = [path_map[int(i)] for i in path]

                all_paths += path
                all_indices += indices
                
                image_t = []
                for i, img_ind in enumerate(indices):
                    image = cv2.imread(image_path.substitute(game_session = path[i], imgind = int(img_ind)))
                    image = cv2.resize(image, (64, 64))
                    image_t.append(image)

                image_t = np.array(image_t)
                image_t = image_t.transpose(0, 3, 1, 2)
                image_t = torch.Tensor(image_t).to(device)

                image_r = init_conv(image_t)
                h0 = init_gru(image_r, batch[:, seq, 2:], h0)

                thumb_fin = thumb_fin_mlp(h0)
                headpos_fin = headpos_fin_mlp(h0)
                headdir_fin = headdir_fin_mlp(h0)

                fin = torch.cat((thumb_fin, headpos_fin, headdir_fin), dim = 1)
                all_actions += fin

                h0 = h0.detach()

                if seq >= start_pred:
                    y = batch[:, seq + 1, 2:]

                    preds = torch.cat((preds, y.cpu()))
                    loss = criterion(fin, y)
                    losses = torch.cat((losses, loss.reshape(1)))
            
            losses = torch.mean(losses)
            rmses = torch.cat((rmses, losses.reshape(1).cpu()))

            prog_bar.update(1)
    prog_bar.close()


    all_indices = torch.stack(all_indices).cpu().detach().int()
    all_actions = torch.stack(all_actions).cpu().detach()

    data = {'game_session': all_paths, 'frame': all_indices}
    paths_df = pd.DataFrame(data)
    actiondf = pd.DataFrame(all_actions, columns=col_pred)
    fin_df = paths_df.join(actiondf)

    # u, c = torch.unique(preds, return_counts = True, dim=0)
    # u = [str(x) for x in u.tolist()]
    # if verbose: writer.add_histogram('Testing values', values=c, bins=u)
    # print(u)
    # c = sorted(c, reverse=True)
    # print(c[0], sum(c[1:]))
    
    # plt.bar(u, c)
    # plt.xticks(rotation=90)
    # plt.show()

    return sum(rmses) / len(loader), sum(torch.sqrt(rmses)) / len(loader), fin_df

# Epoch train + testing
test_mse, test_rmse, final_df = test(test_loader, test_path_map, criterion)

# Only add this if val data is available
# val_rmse, val_ap, val_auc = test(val_loader)
# print(f'Val AP: {val_ap:.4f}, Val AUC: {val_auc:.4f}')

print(f'Test MSE: {test_mse:.4f}, Test RMSE: {test_rmse:.4f}')

if verbose:
    writer.add_scalar('test_mse', test_mse, 0)
    writer.add_scalar('test_rmse', test_rmse, 0)

if save_df:
    # final_df = final_df.sort_values(by='frame')
    final_df = final_df.sort_values(by=['game_session', 'frame'])
    final_df.to_csv(f'/data/mala711/COMPSCI715/CNNRNN/csv_files/{common_name}.csv', index=False)

if verbose: writer.close()
