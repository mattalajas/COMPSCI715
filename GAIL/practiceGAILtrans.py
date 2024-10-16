import torch
import torch.nn as nn
import torch.utils.data
import tqdm
from copy import deepcopy
from functools import partial
from string import Template
import sys
import os
from torch.utils.tensorboard import SummaryWriter

# Initialise path
sys.path.insert(0, os.getcwd())

from utils.data_utils import *
from utils.datasets import *
from CNNRNN.models import *
from GAIL.utils import *
from GAIL.models import *

# Cuda settings
cuda_num = 6
device = torch.device('mps' if torch.backends.mps.is_available() else f'cuda:{cuda_num}' if torch.cuda.is_available() else 'cpu')
# For data collection, change to True if want to evaluate output 
verbose = True
save_file = True

# Empty cuda cache
if torch.cuda.is_available():
    torch.cuda.empty_cache()

#Hyper params:
ppo_epochs           = 10
mini_batch_size      = 10

# Main task hyperparams
seq_size = num_steps = 50
batch_size = 10
epochs = 150
iter_val = 10
img_size = 64
lr = 3e-2
disc_lr = 3e-3
dropout = 0.4
hid_size = 256
weight_decay = 5e-3
early_stop = 40

# Change this to match dataset
train_game_names = ['Barbie']
test_game_names = ['Barbie']
val_game_names = ['Kawaii_House', 'Kawaii_Daycare']
image_path = Template("/data/ysun209/VR.net/videos/${game_session}/video/${imgind}.jpg")

# Create train test split
train_sessions = DataUtils.read_txt("/data/mala711/COMPSCI715/datasets/barbie_demo_dataset/train.txt")
val_sessions = DataUtils.read_txt("/data/mala711/COMPSCI715/datasets/barbie_demo_dataset/val.txt")
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
save_name = f'GAIL_transfer_{train_game_names}_test_{test_game_names}_ppoepochs_{ppo_epochs}_mini_batch_{mini_batch_size}_lr_{lr}_dlr_{disc_lr}_init_test_seq_size_{seq_size}_iter_{iter_val}_dropout_{dropout}_weightd_{weight_decay}'
if verbose: writer = SummaryWriter(f'/data/mala711/COMPSCI715/GAIL/runs/{save_name}')

# Model path
save_path = f'/data/mala711/COMPSCI715/GAIL/models/{save_name}.pth'

############################### DO NOT CHANGE ANYTHING AFTER THIS LINE ###########################################

# Get models
model_img_encoder = TrainedResNet(hid_size, False).to(device)
disc_img_encoder = TrainedResNet(hid_size, False).to(device)
model = ActorCriticSingle(hid_size, num_outputs, hid_size).to(device)
discriminator = DiscriminatorSingle(hid_size + num_outputs, hid_size).to(device)

# Read image function
read_img = partial(read_images, image_path = image_path, img_size = img_size, device = device)

# Loss and optimiser functions
d_criterion = nn.BCELoss()

optimizer  = torch.optim.Adam([{'params': model_img_encoder.parameters()},
                               {'params': model.parameters()}], lr=lr, weight_decay=weight_decay)
optimizer_discrim = torch.optim.Adam([{'params': discriminator.parameters()},
                                    {'params': disc_img_encoder.parameters()}], lr=disc_lr, weight_decay=weight_decay)

# Train function
def train(loader, path_map, model_img_encoder, disc_img_encoder,\
          model, discriminator, d_criterion, \
          optimizer, optimizer_discrim):
    model_img_encoder.train()
    disc_img_encoder.train()
    model.train()
    discriminator.train()

    ppo_losses = []
    discrim_losses = []
    all_rewards = []

    # Num steps is how many steps in the gameplay 
    # States are the images 
    # For now were keeping it the same
    prog_bar = tqdm.tqdm(range(len(loader)))
    # with torch.autograd.set_detect_anomaly(True):
    for batch in loader:
        log_probs = []
        values    = []
        states    = []
        d_states  = []
        actions   = []
        rewards   = []
        entropy = 0

        path = batch[:, 0, 0]
        path = [path_map[int(i)] for i in path]
        images = read_images(path, batch[:, 0, 1], image_path, img_size, device)

        for seq in range(1, num_steps):
            # Get actor critic values
            state = model_img_encoder(images)
            dist, value = model(state)

            action = dist.sample()

            # Read next state and calculate reward
            n_indices = batch[:, seq, 1]
            n_path = batch[:, seq, 0]
            n_path = [path_map[int(i)] for i in n_path]
            next_images = read_images(n_path, n_indices, image_path, img_size, device)

            disc_state = disc_img_encoder(images) 
            reward = expert_reward_single(disc_state, action, device, discriminator)
            
            log_prob = dist.log_prob(action)
            entropy += dist.entropy().mean()
            
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)
            
            states.append(state.detach())
            d_states.append(disc_state)

            actions.append(action)

            images = next_images
            # Change the reward gain
            # if epoch % 1000 == 0:
            #     test_reward = np.mean([test_env() for _ in range(10)])
            #     test_rewards.append(test_reward)
            #     # plot(epoch, test_rewards)
            #     if test_reward > threshold_reward: early_stop = True
        
        all_rewards.append(sum(rewards) / len(rewards))
        state = model_img_encoder(images)
        _, next_value = model(state)
        returns = compute_gae(next_value, rewards, values)

        returns   = torch.stack(returns).detach()
        log_probs = torch.stack(log_probs).detach()
        values    = torch.stack(values).detach()
        states    = torch.stack(states)
        actions   = torch.stack(actions)
        d_states  = torch.stack(d_states)
        advantage = returns - values
        
        # if i_update % 3 == 0:
        ppo_loss = ppo_update_single(model, optimizer, ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantage)
        ppo_losses.append(ppo_loss)

        # Change expert traj to the actions of expert
        # expert_state_action = expert_traj[np.random.randint(0, expert_traj.shape[0], 2 * num_steps * num_envs), :]
        expert_action = batch[:, 1:, 2:]
        expert_action = expert_action.permute(1, 0, 2)
        expert_state_action = torch.cat([d_states, expert_action], 2).to(device)
        state_action        = torch.cat([states, actions], 2).to(device)

        fake = discriminator(state_action)
        real = discriminator(expert_state_action)

        optimizer_discrim.zero_grad()
        discrim_loss = d_criterion(fake, torch.ones(fake.shape).to(device)) + \
                d_criterion(real, torch.zeros(real.shape).to(device))
        discrim_losses.append(discrim_loss)
        discrim_loss.backward()

        optimizer_discrim.step()
        prog_bar.update(1)
    prog_bar.close()

    mean_ppo_loss = sum(ppo_losses) / len(ppo_losses)
    mean_ppo_loss = mean_ppo_loss.detach().cpu().item()

    mean_discrim_loss = sum(discrim_losses) / len(discrim_losses)
    mean_discrim_loss = mean_discrim_loss.detach().cpu().item()

    all_rewards = torch.stack(all_rewards)
    mean_rewards = torch.mean(all_rewards, 0)
    mean_rewards = torch.mean(mean_rewards, 0)
    mean_rewards = mean_rewards.detach().cpu().item()

    return mean_ppo_loss, mean_discrim_loss, mean_rewards

# Test function
def test(loader, path_map, model_img_encoder, disc_img_encoder, model, discriminator, d_criterion):
    model_img_encoder.eval()
    disc_img_encoder.eval()
    model.eval()
    discriminator.eval()

    discrim_losses = []
    all_rewards = []

    # Num steps is how many steps in the gameplay 
    # States are the images 
    # For now were keeping it the same
    prog_bar = tqdm.tqdm(range(len(loader)))

    with torch.no_grad():
        for batch in loader:
            log_probs = []
            values    = []
            states    = []
            d_states  = []
            actions   = []
            rewards   = []
            entropy = 0

            path = batch[:, 0, 0]
            path = [path_map[int(i)] for i in path]
            images = read_images(path, batch[:, 0, 1], image_path, img_size, device)

            for seq in range(1, num_steps):
                # Get actor critic values
                state = model_img_encoder(images)
                dist, value = model(state)

                action = dist.sample()

                # Read next state and calculate reward
                n_indices = batch[:, seq, 1]
                n_path = batch[:, seq, 0]
                n_path = [path_map[int(i)] for i in n_path]
                next_images = read_images(n_path, n_indices, image_path, img_size, device)
                
                # next_state, _, done, _ = envs.step(action.cpu().numpy())
                disc_state = disc_img_encoder(images)
                reward = expert_reward_single(disc_state, action, device, discriminator)
                
                log_prob = dist.log_prob(action)
                entropy += dist.entropy().mean()
                
                log_probs.append(log_prob)
                values.append(value)
                rewards.append(reward)
                
                states.append(state)
                d_states.append(disc_state)

                actions.append(action)
                
                images = next_images
                # Change the reward gain
                # if epoch % 1000 == 0:
                #     test_reward = np.mean([test_env() for _ in range(10)])
                #     test_rewards.append(test_reward)
                #     # plot(epoch, test_rewards)
                #     if test_reward > threshold_reward: early_stop = True
            
            all_rewards.append(sum(rewards) / len(rewards))
            state = model_img_encoder(images)
            _, next_value = model(state)
            returns = compute_gae(next_value, rewards, values)

            returns   = torch.stack(returns)
            log_probs = torch.stack(log_probs)
            values    = torch.stack(values)
            states    = torch.stack(states)
            d_states  = torch.stack(d_states)
            actions   = torch.stack(actions)

            # Change expert traj to the actions of expert
            # expert_state_action = expert_traj[np.random.randint(0, expert_traj.shape[0], 2 * num_steps * num_envs), :]
            expert_action = batch[:, 1:, 2:]
            expert_action = expert_action.permute(1, 0, 2)
            expert_state_action = torch.cat([d_states, expert_action], 2).to(device)
            state_action        = torch.cat([states, actions], 2).to(device)

            fake = discriminator(state_action)
            real = discriminator(expert_state_action)

            discrim_loss = d_criterion(fake, torch.ones(fake.shape).to(device)) + \
                d_criterion(real, torch.zeros(real.shape).to(device))
            discrim_losses.append(discrim_loss)

            prog_bar.update(1)
        prog_bar.close()

    mean_discrim_loss = sum(discrim_losses) / len(discrim_losses)
    mean_discrim_loss = mean_discrim_loss.cpu().item()

    all_rewards = torch.stack(all_rewards)
    mean_rewards = torch.mean(all_rewards, 0)
    mean_rewards = torch.mean(mean_rewards, 0)
    mean_rewards = mean_rewards.cpu().item()

    return mean_discrim_loss, mean_rewards

# Run epochs
for epoch in range(1, epochs+1):
    train_ppo_loss, train_discrim_loss, train_rewards = train(train_loader, train_path_map, \
                                                              model_img_encoder, disc_img_encoder, model, \
                                                              discriminator, d_criterion, optimizer, optimizer_discrim)
    test_discrim_loss, test_rewards = test(test_loader, test_path_map, model_img_encoder, \
                                           disc_img_encoder, model, discriminator, d_criterion)

    # Only add this if val data is available
    # val_rmse, val_ap, val_auc = test(val_loader)
    # print(f'Val AP: {val_ap:.4f}, Val AUC: {val_auc:.4f}')

    if epoch == early_stop:
        break

    print(f'Epoch: {epoch:02d}, Train PPO Loss: {train_ppo_loss:.4f}, Train Discrim MSE: {train_discrim_loss:.4f}, Train Reward: {train_rewards:.4f}')
    print(f'Test Discrim MSE: {test_discrim_loss:.4f}, Test Reward: {test_rewards:.4f}')

    if verbose:
        writer.add_scalar('train_ppo_loss', train_ppo_loss, epoch)
        writer.add_scalar('train_discrim_loss', train_discrim_loss, epoch)
        writer.add_scalar('train_rewards', train_rewards, epoch)
        writer.add_scalar('test_discrim_loss', test_discrim_loss, epoch)
        writer.add_scalar('test_rewards', test_rewards, epoch)

# Save model
if save_file:
    torch.save({
            'model_img_encoder_state_dict': model_img_encoder.state_dict(),
            'disc_img_encoder_state_dict': disc_img_encoder.state_dict(),
            'model_state_dict': model.state_dict(),
            'discriminator_state_dict': discriminator.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'optimizer_discrim_state_dict': optimizer_discrim.state_dict(),
            }, f"{save_path}")

if verbose: writer.close()