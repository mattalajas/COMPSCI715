import cv2
import copy
import numpy as np
import torch
import torch.utils.data
import tqdm
import os
import sys
from torch.utils.tensorboard import SummaryWriter

# Initialise path
sys.path.insert(0, os.getcwd())

from CNNRNN.models import MLP, LeNet, actionGRU
from utils.data_utils import *
from utils.datasets import *
from string import Template

# Cuda setup
cuda_num = 5
device = torch.device('mps' if torch.backends.mps.is_available() else f'cuda:{cuda_num}' if torch.cuda.is_available() else 'cpu')
# For data collection, change to True if want to evaluate output 
verbose = False
save_file = False

if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Main task hyperparams
seq_size = 50
batch_size = 10
start_pred = 20
epochs = 5
iter_val = 10
img_size = 64
main_lr= 0.01
regularisation = 0.00001
dropout = 0.2
rnn_emb = 256
hid_size = 256
weighted = True

# Change this to match dataset
train_game_names = ['Barbie']
test_game_names = ['Barbie']
val_game_names = ['Kawaii_House', 'Kawaii_Daycare']
image_path = Template("/data/ysun209/VR.net/videos/${game_session}/video/${imgind}.jpg")

# Create train test split
train_sessions = DataUtils.read_txt("/data/mala711/COMPSCI715/datasets/barbie_demo_dataset/train.txt")
val_sessions = DataUtils.read_txt("/data/mala711/COMPSCI715/datasets/final_data_splits/val.txt")
test_sessions = DataUtils.read_txt("/data/mala711/COMPSCI715/datasets/barbie_demo_dataset/test.txt")

# Columns to predict
col_pred = ["thumbstick_left_x", "thumbstick_left_y", "thumbstick_right_x", "thumbstick_right_y", "head_pos_x", "head_pos_y", "head_pos_z", "head_dir_a", "head_dir_b", "head_dir_c", "head_dir_d"]
num_outputs = len(col_pred)

# Get val and test sets
train_set = MultiGameDataset(train_game_names, train_sessions, cols_to_predict=col_pred)
val_set = MultiGameDataset(val_game_names, val_sessions, cols_to_predict=col_pred)
test_set = MultiGameDataset(test_game_names, test_sessions, cols_to_predict=col_pred) 

# Normalisation, feel free to change this 
thumbsticks_loc = 6
head_pos_loc = 9

train_set.df[train_set.df.columns[2:thumbsticks_loc]] = (train_set.df[train_set.df.columns[2:thumbsticks_loc]] + 1) / 2
val_set.df[val_set.df.columns[2:thumbsticks_loc]] = (val_set.df[val_set.df.columns[2:thumbsticks_loc]] + 1) / 2
test_set.df[test_set.df.columns[2:thumbsticks_loc]] = (test_set.df[test_set.df.columns[2:thumbsticks_loc]] + 1) / 2

train_set.df[train_set.df.columns[thumbsticks_loc:head_pos_loc]] = (train_set.df[train_set.df.columns[thumbsticks_loc:head_pos_loc]] + 2) / 4
val_set.df[val_set.df.columns[thumbsticks_loc:head_pos_loc]] = (val_set.df[val_set.df.columns[thumbsticks_loc:head_pos_loc]] + 2) / 4
test_set.df[test_set.df.columns[thumbsticks_loc:head_pos_loc]] = (test_set.df[test_set.df.columns[thumbsticks_loc:head_pos_loc]] + 2) / 4

train_set.df[train_set.df.columns[head_pos_loc:]] = (train_set.df[train_set.df.columns[head_pos_loc:]] + 1) / 2
val_set.df[val_set.df.columns[head_pos_loc:]] = (val_set.df[val_set.df.columns[head_pos_loc:]] + 1) / 2
test_set.df[test_set.df.columns[head_pos_loc:]] = (test_set.df[test_set.df.columns[head_pos_loc:]] + 1) / 2

# Subsample datasets
train_path_map, train_loader = filter_dataframe(train_sessions, train_set.df, device, seq_size, batch_size, iter=iter_val)
test_path_map, test_loader = filter_dataframe(test_sessions, test_set.df, device, seq_size, batch_size, iter=iter_val)

# Run tensorboard summary writer
save_name = f'GRU_train_{train_game_names}_test_{test_game_names}_init_test_seq_size_{seq_size}_seqstart_{start_pred}_iter_{iter_val}_reg_{regularisation}_lr_{main_lr}_dropout_{dropout}_weighting_{weighted}'
if verbose: writer = SummaryWriter(f'/data/mala711/COMPSCI715/CNNRNN/runs/{save_name}')

# Model path
save_path = f'/data/mala711/COMPSCI715/CNNRNN/models/{save_name}.pth'

############################### DO NOT CHANGE ANYTHING AFTER THIS LINE ###########################################

# Initialise models
init_conv = LeNet(img_size, hid_size, dropout=dropout).to(device)
init_gru = actionGRU(num_outputs, rnn_emb, hid_size, hid_size, dropout).to(device)
thumb_fin_mlp = MLP(hid_size, 4, dropout).to(device)
headpos_fin_mlp = MLP(hid_size, 3, dropout).to(device)
headdir_fin_mlp = MLP(hid_size, 4, dropout).to(device)

# Initialise optimiser and loss function
optimizer = torch.optim.Adam([
    {'params': init_conv.parameters()},
    {'params': init_gru.parameters()},
    {'params': thumb_fin_mlp.parameters()},
    {'params': headpos_fin_mlp.parameters()},
    {'params': headdir_fin_mlp.parameters()}], lr=main_lr)

def weighted_mse_loss(input, target, weight):
    return (weight * (input - target) ** 2)

# Get either weighted or non weighted loss
if weighted:
    criterion = weighted_mse_loss
else:
    criterion = torch.nn.MSELoss()

# Train function
def train(loader, path_map, optimizer, criterion):
    init_gru.train()
    init_conv.train()
    thumb_fin_mlp.train()
    headpos_fin_mlp.train()
    headdir_fin_mlp.train()
    
    total_loss = []
    preds = torch.empty(0)

    prog_bar = tqdm.tqdm(range(len(loader)))
    for batch in loader:
        optimizer.zero_grad()

        # Need to initialise the hidden state for GRU
        h0 = torch.empty((batch.shape[0], hid_size)).to(device)
        h0 = torch.nn.init.xavier_uniform_(h0)

        image_emb = []
        beliefs = []

        losses = torch.empty(0).to(device)

        # Iterate through all frames in sequence
        for seq in range(batch.shape[1]-1):
            # (path, image_ind, T1, T2, T3, T4)
            indices = batch[:, seq, 1]
            path = batch[:, seq, 0]
            path = [path_map[int(i)] for i in path]
            
            # Reads and encodes the image
            image_t = []
            for i, img_ind in enumerate(indices):
                cur_path = image_path.substitute(game_session = path[i], imgind = int(img_ind))
                image = cv2.imread(cur_path)
                image = cv2.resize(image, (img_size, img_size))
                image_t.append(image)

            image_t = np.array(image_t)
            image_t = image_t.transpose(0, 3, 1, 2)
            image_t = torch.Tensor(image_t).to(device)

            image_r = init_conv(image_t)
            image_emb.append(copy.deepcopy(image_r.detach()))

            # GRU step per image and its associated thumbstick comman
            h0 = init_gru(image_r, batch[:, seq, 2:], h0)
            beliefs.append(copy.deepcopy(h0.detach()))

            # Final prediction for each frame 
            thumb_fin = thumb_fin_mlp(h0)
            headpos_fin = headpos_fin_mlp(h0)
            headdir_fin = headdir_fin_mlp(h0)

            fin = torch.cat((thumb_fin, headpos_fin, headdir_fin), dim = 1)

            h0 = h0.detach()

            # Will only start prediction after certain number of frames
            if seq >= start_pred:
                y = batch[:, seq + 1, 2:]
                weights = torch.ones_like(y).to(device)

                if weighted:
                    weights = torch.abs(0.5 - y) / 0.5
                    loss = criterion(fin, y, weights)
                else:
                    loss = criterion(fin, y)
                # Loss calculation and appending to total loss
                
                loss = torch.mean(loss)
                losses = torch.cat((losses, loss.reshape(1)))

                preds = torch.cat((preds, y.cpu()))

        # Regularisation
        l1 = sum(p.abs().sum() for p in init_gru.parameters())
        l1 += sum(p.abs().sum() for p in init_conv.parameters())
        l1 += sum(p.abs().sum() for p in thumb_fin_mlp.parameters())
        l1 += sum(p.abs().sum() for p in headdir_fin_mlp.parameters())
        l1 += sum(p.abs().sum() for p in headpos_fin_mlp.parameters())
        
        # Loss calculation, gradient calculation, then backprop
        losses = torch.mean(losses)

        losses += regularisation*l1
        losses.backward()
        optimizer.step()

        total_loss.append(losses)

        prog_bar.update(1)
    prog_bar.close()
    
    # Returns evaluation scores
    return sum(total_loss) / len(loader), total_loss #, sum(total_aux) / len(loader)

# Test is very similar to training
# Instead I use RMSE and not MSE
# I also used torch no grad to be space efficient
def test(loader, path_map, criterion):
    init_gru.eval()
    init_conv.eval()
    thumb_fin_mlp.eval()
    headpos_fin_mlp.eval()
    headdir_fin_mlp.eval()

    rmses = torch.empty(0)

    prog_bar = tqdm.tqdm(range(len(loader)))

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

                h0 = h0.detach()

                if seq >= start_pred:
                    y = batch[:, seq + 1, 2:]

                    weights = torch.ones_like(y).to(device)

                    if weighted:
                        weights = torch.abs(0.5 - y) / 0.5
                        loss = criterion(fin, y, weights)
                    else:
                        loss = criterion(fin, y)
                        
                    # Loss calculation and appending to total loss
                    loss = torch.mean(loss)
                    losses = torch.cat((losses, loss.reshape(1)))
            
            losses = torch.mean(losses)
            rmses = torch.cat((rmses, losses.reshape(1).cpu()))

            prog_bar.update(1)
    prog_bar.close()

    return sum(rmses) / len(loader), sum(torch.sqrt(rmses)) / len(loader)

# Epoch train + testing
for epoch in range(1, epochs+1):
    loss, loss_list = train(train_loader, train_path_map, optimizer, criterion)
    test_mse, test_rmse = test(test_loader, test_path_map, criterion)

    print(f'Epoch: {epoch:02d}, Train Loss: {loss:.4f}, Test MSE: {test_mse:.4f}, Test RMSE: {test_rmse:.4f}')

    if verbose:
        writer.add_scalar('training_loss', loss, epoch)
        writer.add_scalar('test_mse', test_mse, epoch)
        writer.add_scalar('test_rmse', test_rmse, epoch)

if save_file:
    torch.save({
            'init_conv_state_dict': init_conv.state_dict(),
            'init_gru_state_dict': init_gru.state_dict(),
            'thumb_fin_mlp_state_dict': thumb_fin_mlp.state_dict(),
            'headpos_fin_mlp_state_dict': headpos_fin_mlp.state_dict(),
            'headdir_fin_mlp_state_dict': headdir_fin_mlp.state_dict(),
            'optimizer_state_dict' : optimizer.state_dict(),
            }, f"{save_path}")

if verbose: writer.close()
