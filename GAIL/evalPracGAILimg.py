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
cuda_num = 7
device = torch.device('mps' if torch.backends.mps.is_available() else f'cuda:{cuda_num}' if torch.cuda.is_available() else 'cpu')
# For data collection, change to True if want to evaluate output 
verbose = False
save_df = True

# Empty cuda cache
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Hyperparams:
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
val_set = MultiGameDataset(val_game_names, val_sessions, cols_to_predict=col_pred)
test_set = MultiGameDataset(test_game_names, test_sessions, cols_to_predict=col_pred) 

# Normalisation, feel free to change this 
thumbsticks_loc = 6
head_pos_loc = 9

val_set.df[val_set.df.columns[2:thumbsticks_loc]] = (val_set.df[val_set.df.columns[2:thumbsticks_loc]] + 1) / 2
test_set.df[test_set.df.columns[2:thumbsticks_loc]] = (test_set.df[test_set.df.columns[2:thumbsticks_loc]] + 1) / 2

val_set.df[val_set.df.columns[thumbsticks_loc:head_pos_loc]] = (val_set.df[val_set.df.columns[thumbsticks_loc:head_pos_loc]] + 2) / 4
test_set.df[test_set.df.columns[thumbsticks_loc:head_pos_loc]] = (test_set.df[test_set.df.columns[thumbsticks_loc:head_pos_loc]] + 2) / 4

val_set.df[val_set.df.columns[head_pos_loc:]] = (val_set.df[val_set.df.columns[head_pos_loc:]] + 1) / 2
test_set.df[test_set.df.columns[head_pos_loc:]] = (test_set.df[test_set.df.columns[head_pos_loc:]] + 1) / 2

# Subsample dataset
test_path_map, test_loader = filter_dataframe(test_sessions, test_set.df, device, seq_size, batch_size, iter=iter_val)

# Run tensorboard summary writer
save_name = f'GAILimg_train_{train_game_names}_test_{test_game_names}_ppoepochs_{ppo_epochs}_mini_batch_{mini_batch_size}_lr_{lr}_dlr_{disc_lr}_init_test_seq_size_{seq_size}_iter_{iter_val}_dropout_{dropout}_weightd_{weight_decay}'
if verbose: writer = SummaryWriter(f'/data/mala711/COMPSCI715/GAIL/runs/Eval{save_name}')

# Model and csv paths
save_path = f'/data/mala711/COMPSCI715/GAIL/models/{save_name}.pth'
save_csv_path = f'/data/mala711/COMPSCI715/GAIL/csv_files/{save_name}.csv'

############################### DO NOT CHANGE ANYTHING AFTER THIS LINE ###########################################

# Get models
model_img_encoder = models.LeNet(img_size, hid_size, dropout=dropout).to(device)
disc_img_encoder = models.LeNet(img_size, hid_size, dropout=dropout).to(device)
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

# Load states
checkpoint = torch.load(save_path, weights_only=True)
model_img_encoder.load_state_dict(checkpoint['model_img_encoder_state_dict'])
disc_img_encoder.load_state_dict(checkpoint['disc_img_encoder_state_dict'])
model.load_state_dict(checkpoint['model_state_dict'])
discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
optimizer_discrim.load_state_dict(checkpoint['optimizer_discrim_state_dict'])

# Test function
def test(loader, path_map, model_img_encoder, disc_img_encoder, model, discriminator, d_criterion):
    model_img_encoder.eval()
    disc_img_encoder.eval()
    model.eval()
    discriminator.eval()

    discrim_losses = []
    all_rewards = []
    all_actions = []
    all_paths = []
    all_indices = []

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

            n_indices = batch[:, 0, 1]
            n_path = batch[:, 0, 0]
            n_path = [path_map[int(i)] for i in n_path]
            images = read_images(n_path, n_indices, image_path, img_size, device)

            for seq in range(1, num_steps):
                # Save all paths
                all_paths += n_path
                all_indices += n_indices

                # Get actor critic values
                state = model_img_encoder(images)
                dist, value = model(state)

                action = dist.sample()

                # Save all actions
                all_actions += action

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

    all_indices = torch.stack(all_indices).cpu().detach().int()
    all_actions = torch.stack(all_actions).cpu().detach()

    data = {'game_session': all_paths, 'frame': all_indices}
    paths_df = pd.DataFrame(data)
    actiondf = pd.DataFrame(all_actions, columns=col_pred)
    fin_df = paths_df.join(actiondf)

    mean_discrim_loss = sum(discrim_losses) / len(discrim_losses)
    mean_discrim_loss = mean_discrim_loss.cpu().item()

    all_rewards = torch.stack(all_rewards)
    mean_rewards = torch.mean(all_rewards, 0)
    mean_rewards = torch.mean(mean_rewards, 0)
    mean_rewards = mean_rewards.cpu().item()

    return mean_discrim_loss, mean_rewards, fin_df

# Get single run of test data
test_discrim_loss, test_rewards, final_df = test(test_loader, test_path_map, model_img_encoder, \
                                        disc_img_encoder, model, discriminator, d_criterion)

print(f'Test Discrim MSE: {test_discrim_loss:.4f}, Test Reward: {test_rewards:.4f}')

if verbose:
    writer.add_scalar('test_discrim_loss', test_discrim_loss, 0)
    writer.add_scalar('test_rewards', test_rewards, 0)

# Save outputs in csv
if save_df:
    final_df = final_df.sort_values(by=['game_session', 'frame'])
    final_df.to_csv(f'{save_csv_path}', index=False)

if verbose: writer.close()