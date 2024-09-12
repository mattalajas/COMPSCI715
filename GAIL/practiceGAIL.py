from ast import literal_eval

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import tqdm
from string import Template
from torch.utils.tensorboard import SummaryWriter
from CNNRNN.models import *
from utils import *

from models import *
from utils.data_utils import *

device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
# For data collection, change to True if want to evaluate output 
verbose = False

if torch.cuda.is_available():
    torch.cuda.empty_cache()

num_inputs  = 256

#Hyper params:
a2c_hidden_size      = 256
lr                   = 3e-3
num_steps            = 20
mini_batch_size      = 5 # TODO: Check this one out
ppo_epochs           = 4
threshold_reward     = -200

# Main task hyperparams
seq_size = num_steps = 50
batch_size = 10
start_pred = 20
epochs = 150
iter_val = 10
img_size = 64
lr = 3e-3
regularisation = 0.00001
dropout = 0.2
rnn_emb = 256
hid_size = 256
rnn_type = 'gru'
discrim_hidden_size  = 128

num_outputs = 4

train_game_names = ['Wild_Quest', 'Barbie', 'Circle_Kawaii']
test_game_names = ['Barbie']
val_game_names = ['Barbie']
image_path = Template("/data/ysun209/VR.net/videos/${game_session}/video/${imgind}.jpg")

# Create train test split
train_sessions = DataUtils.read_txt("/data/mala711/COMPSCI715/datasets/barbie_demo_dataset/train.txt")
val_sessions = DataUtils.read_txt("/data/mala711/COMPSCI715/datasets/barbie_demo_dataset/val.txt")
test_sessions = DataUtils.read_txt("/data/mala711/COMPSCI715/datasets/barbie_demo_dataset/test.txt")

train_set = MultiGameDataset(train_game_names, train_sessions)
val_set = MultiGameDataset(val_game_names, val_sessions)
test_set = MultiGameDataset(test_game_names, test_sessions) 

train_path_map, train_loader = filter_dataframe(train_sessions, train_set.df, device, seq_size, batch_size, iter=iter_val)
test_path_map, test_loader = filter_dataframe(test_sessions, test_set.df, device, seq_size, batch_size, iter=iter_val)

# Run tensorboard summary writer
if verbose: writer = SummaryWriter(f'/data/mala711/COMPSCI715/GAIL/runs/GAIL_train_{train_game_names}_test_{test_game_names}_init_test_seq_size_{seq_size}_seqstart_{start_pred}_iter_{iter_val}_reg_{regularisation}_dropout_{dropout}')

img_encoder   = models.LeNet(img_size, hid_size, dropout=dropout)
model         = ActorCritic(num_outputs, img_size, hid_size, rnn_type, rnn_emb, hid_size, dropout=dropout).to(device)
discriminator = Discriminator(num_inputs + num_outputs, discrim_hidden_size).to(device)

d_criterion = nn.BCELoss()

optimizer_img_encoder = torch.optim.Adam(img_encoder.parameters(), lr=lr)
optimizer  = torch.optim.Adam(model.parameters(), lr=lr)
optimizer_discrim = torch.optim.Adam(discriminator.parameters(), lr=lr)

test_rewards = []

# Change to epochs
tot_epochs = 500
epoch = 0

# Change this to actual actions + states
expert_traj = 0
early_stop = False

optimizer_img_encoder.zero_grad()

def train(loader, path_map, img_encoder, model, discriminator, d_criterion, optimizer_img_encoder, \
          optimizer, optimizer_discrim):
    img_encoder.train()
    model.train()
    discriminator.train()
    
    log_probs = []
    values    = []
    states    = []
    actions   = []
    rewards   = []
    entropy = 0

    ppo_losses = []
    discrim_losses = []

    # Num steps is how many steps in the gameplay 
    # States are the images 
    # For now were keeping it the same
    prog_bar = tqdm.tqdm(range(len(loader)))
    for batch in loader:
        optimizer_img_encoder.zero_grad()
        
        # Initialise RNN hidden states
        h0_a = torch.empty((batch.shape[0], hid_size)).to(device)
        h0_a = torch.nn.init.xavier_uniform_(h0_a)

        h0_c = torch.empty((batch.shape[0], hid_size)).to(device)
        h0_c = torch.nn.init.xavier_uniform_(h0_c)

        c0_a = torch.zeros((batch.shape[0], hid_size)).to(device)
        c0_a = torch.nn.init.xavier_uniform_(c0_a)

        c0_c = torch.zeros((batch.shape[0], hid_size)).to(device)
        c0_c = torch.nn.init.xavier_uniform_(c0_c)

        path = batch[:, 0, 0]
        path = [path_map[int(i)] for i in path]
        state = read_image(path, batch[:, 0, 1], image_path, img_size, device)

        action = torch.Tensor(batch[:, 0, 2:]).to(device)

        for seq in range(1, num_steps-1):
            # Get actor critic values
            dist, value, h0_c, h0_a, c0_c, c0_a = model(state, action, h0_c, h0_a, c0_c, c0_a)

            action = dist.sample()

            # Read next state and calculate reward
            n_indices = batch[:, seq, 1]
            n_path = batch[:, seq, 0]
            n_path = [path_map[int(i)] for i in path]

            next_state = read_image(n_path, n_indices, image_path, img_size, device)
            # next_state, _, done, _ = envs.step(action.cpu().numpy())
            reward = expert_reward(state, action.cpu().numpy(), device, discriminator)
            
            log_prob = dist.log_prob(action)
            entropy += dist.entropy().mean()
            
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(torch.FloatTensor(reward).to(device))
            
            states.append(state)
            actions.append(action)
            
            state = next_state

            # Change the reward gain
            # if epoch % 1000 == 0:
            #     test_reward = np.mean([test_env() for _ in range(10)])
            #     test_rewards.append(test_reward)
            #     # plot(epoch, test_rewards)
            #     if test_reward > threshold_reward: early_stop = True
                
        _, next_value, _, _, _, _ = model(next_state, action, h0_c, h0_a, c0_c, c0_a)
        returns = compute_gae(next_value, rewards, values)

        returns   = torch.cat(returns).detach()
        log_probs = torch.cat(log_probs).detach()
        values    = torch.cat(values).detach()
        states    = torch.cat(states)
        actions   = torch.cat(actions)
        advantage = returns - values
        
        # if i_update % 3 == 0:
        ppo_loss = ppo_update(model, optimizer, 4, mini_batch_size, states, actions, log_probs, returns, advantage)
        ppo_losses.append(ppo_loss)

        # Change expert traj to the actions of expert
        # expert_state_action = expert_traj[np.random.randint(0, expert_traj.shape[0], 2 * num_steps * num_envs), :]
        expert_state_action = torch.cat([states, batch[:, :, 2:]], 1).to(device)
        state_action        = torch.cat([states, actions], 1).to(device)
        fake = discriminator(state_action)
        real = discriminator(expert_state_action)
        optimizer_discrim.zero_grad()
        discrim_loss = d_criterion(fake, torch.ones((states.shape[0], 1)).to(device)) + \
                d_criterion(real, torch.zeros((expert_state_action.size(0), 1)).to(device))
        discrim_losses.append(discrim_loss)
        discrim_loss.backward()

        optimizer_discrim.step()
        prog_bar.update(1)
    prog_bar.close()

    return sum(ppo_losses) / len(ppo_losses), sum(discrim_losses) / len(discrim_losses), sum(rewards) / len(rewards)

def test(loader, path_map, img_encoder, model, discriminator, d_criterion, i_update):
    i_update += 1
    img_encoder.eval()
    model.eval()
    discriminator.eval()
    
    log_probs = []
    values    = []
    states    = []
    actions   = []
    rewards   = []
    entropy = 0

    ppo_losses = []
    discrim_losses = []

    # Num steps is how many steps in the gameplay 
    # States are the images 
    # For now were keeping it the same
    prog_bar = tqdm.tqdm(range(len(loader)))

    with torch.no_grad():
        for batch in loader:
            # Initialise RNN hidden states
            h0_a = torch.empty((batch.shape[0], hid_size)).to(device)
            h0_a = torch.nn.init.xavier_uniform_(h0_a)

            h0_c = torch.empty((batch.shape[0], hid_size)).to(device)
            h0_c = torch.nn.init.xavier_uniform_(h0_c)

            c0_a = torch.zeros((batch.shape[0], hid_size)).to(device)
            c0_a = torch.nn.init.xavier_uniform_(c0_a)

            c0_c = torch.zeros((batch.shape[0], hid_size)).to(device)
            c0_c = torch.nn.init.xavier_uniform_(c0_c)

            path = batch[:, 0, 0]
            path = [path_map[int(i)] for i in path]
            state = read_image(path, batch[:, 0, 1], image_path, img_size, device)

            action = torch.Tensor(batch[:, 0, 2:]).to(device)

            for seq in range(1, num_steps):
                # Get actor critic values
                dist, value, h0_c, h0_a, c0_c, c0_a = model(state, action, h0_c, h0_a, c0_c, c0_a)

                action = dist.sample()

                # Read next state and calculate reward
                n_indices = batch[:, seq, 1]
                n_path = batch[:, seq, 0]
                n_path = [path_map[int(i)] for i in path]

                next_state = read_image(n_path, n_indices, image_path, img_size, device)
                # next_state, _, done, _ = envs.step(action.cpu().numpy())
                reward = expert_reward(state, action.cpu().numpy(), device, discriminator)
                
                log_prob = dist.log_prob(action)
                entropy += dist.entropy().mean()
                
                log_probs.append(log_prob)
                values.append(value)
                rewards.append(torch.FloatTensor(reward).to(device))
                
                states.append(state)
                actions.append(action)
                
                state = next_state

                # Change the reward gain
                # if epoch % 1000 == 0:
                #     test_reward = np.mean([test_env() for _ in range(10)])
                #     test_rewards.append(test_reward)
                #     # plot(epoch, test_rewards)
                #     if test_reward > threshold_reward: early_stop = True
                    
            _, next_value, _, _, _, _ = model(next_state, action, h0_c, h0_a, c0_c, c0_a)
            returns = compute_gae(next_value, rewards, values)

            returns   = torch.cat(returns).detach()
            log_probs = torch.cat(log_probs).detach()
            values    = torch.cat(values).detach()
            states    = torch.cat(states)
            actions   = torch.cat(actions)

            # Change expert traj to the actions of expert
            # expert_state_action = expert_traj[np.random.randint(0, expert_traj.shape[0], 2 * num_steps * num_envs), :]
            expert_state_action = torch.cat([states, batch[:, :, 2:]], 1).to(device)
            state_action        = torch.cat([states, actions], 1).to(device)
            fake = discriminator(state_action)
            real = discriminator(expert_state_action)
            discrim_loss = d_criterion(fake, torch.ones((states.shape[0], 1)).to(device)) + \
                    d_criterion(real, torch.zeros((expert_state_action.size(0), 1)).to(device))
            discrim_losses.append(discrim_loss)

            prog_bar.update(1)
        prog_bar.close()

    return sum(ppo_losses) / len(ppo_losses), sum(discrim_losses) / len(discrim_losses), sum(rewards) / len(rewards)

for epoch in range(1, epochs+1):
    train_ppo_loss, train_discrim_loss, train_rewards = train(train_loader, train_path_map, img_encoder, \
                                      model, discriminator, d_criterion, optimizer_img_encoder, \
                                      optimizer, optimizer_discrim)
    test_ppo_loss, test_discrim_loss, test_rewards = test(test_loader, test_path_map, img_encoder, model, discriminator, d_criterion)

    # Only add this if val data is available
    # val_rmse, val_ap, val_auc = test(val_loader)
    # print(f'Val AP: {val_ap:.4f}, Val AUC: {val_auc:.4f}')

    print(f'Epoch: {epoch:02d}, Train PPO Loss: {train_ppo_loss:.4f}, Train Discrim MSE: {train_discrim_loss:.4f}, Train Reward: {train_rewards:.4f}')
    print(f'Test PPO Loss: {test_ppo_loss:.4f}, Train Discrim MSE: {test_discrim_loss:.4f}, Train Reward: {test_rewards:.4f}')

    if verbose:
        writer.add_scalar('train_ppo_loss', train_ppo_loss, epoch)
        writer.add_scalar('train_discrim_loss', train_discrim_loss, epoch)
        writer.add_scalar('train_rewards', train_rewards, epoch)
        writer.add_scalar('test_ppo_loss', test_ppo_loss, epoch)
        writer.add_scalar('test_discrim_loss', test_discrim_loss, epoch)
        writer.add_scalar('test_rewards', test_rewards, epoch)

if verbose: writer.close()