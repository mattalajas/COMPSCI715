# Code retrieved from: https://github.com/higgsfield-ai/higgsfield
import cv2
import numpy as np
import torch
import torch.utils.data

def expert_reward(state, h0_c, h0_a, c0_c, c0_a, action, device, discriminator):
    state_action = torch.cat([state, action], 1).to(device)
    state_action = torch.unsqueeze(state_action, 0)
    return -torch.log(discriminator(state_action, h0_c, h0_a, c0_c, c0_a))

def compute_gae(next_value, rewards, values, gamma=0.99, tau=0.95):
    values = values + [next_value]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] - values[step]
        gae = delta + gamma * tau * gae
        returns.insert(0, gae + values[step])
    return returns

def ppo_iter(mini_batch_size, states, actions, h0_as, h0_cs, c0_as, c0_cs, log_probs, returns, advantage):
    batch_size = states.size(1)
    seq_len = states.size(0)
    for _ in range(batch_size // mini_batch_size):
        rand_ids = np.random.randint(0, batch_size, mini_batch_size)
        seq_inds = np.random.randint(0, seq_len, mini_batch_size)

        yield states[seq_inds, rand_ids, :], actions[seq_inds, rand_ids, :], h0_as[seq_inds, rand_ids, :], \
            h0_cs[seq_inds, rand_ids, :], c0_as[seq_inds, rand_ids, :], c0_cs[seq_inds, rand_ids, :], \
            log_probs[seq_inds, rand_ids, :], returns[seq_inds, rand_ids, :], advantage[seq_inds, rand_ids, :]
        

def ppo_update(model, optimizer, ppo_epochs, mini_batch_size, states, actions, \
               h0_as, h0_cs, c0_as, c0_cs, log_probs, returns, advantages, clip_param=0.2):
    for _ in range(ppo_epochs):
        for state, action, h0_a, h0_c, c0_a, c0_c, old_log_probs, return_, advantage \
            in ppo_iter(mini_batch_size, states, actions, h0_as, h0_cs, c0_as, c0_cs, log_probs, returns, advantages):
        
            dist, value, _, _, _, _ = model(state, action, h0_a, h0_c, c0_a, c0_c)
            entropy = dist.entropy().mean()
            new_log_probs = dist.log_prob(action)

            ratio = (new_log_probs - old_log_probs).exp()
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage

            actor_loss  = - torch.min(surr1, surr2).mean()
            critic_loss = (return_ - value).pow(2).mean()

            loss = 0.5 * critic_loss + actor_loss - 0.001 * entropy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    return loss

# Single utils

def expert_reward_single(state, action, device, discriminator):
    state_action = torch.cat([state, action], 1).to(device)
    return -torch.log(discriminator(state_action))

def ppo_iter_single(mini_batch_size, states, actions, log_probs, returns, advantage):
    batch_size = states.size(0)
    for _ in range(batch_size // mini_batch_size):
        rand_ids = np.random.randint(0, batch_size, mini_batch_size)
        yield states[rand_ids, :], actions[rand_ids, :], log_probs[rand_ids, :], returns[rand_ids, :], advantage[rand_ids, :]
        
def ppo_update_single(model, optimizer, ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantages, clip_param=0.2):
    for _ in range(ppo_epochs):
        for state, action, old_log_probs, return_, advantage in ppo_iter_single(mini_batch_size, states, actions, log_probs, returns, advantages):
            dist, value = model(state)
            entropy = dist.entropy().mean()
            new_log_probs = dist.log_prob(action)

            ratio = (new_log_probs - old_log_probs).exp()
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage

            actor_loss  = - torch.min(surr1, surr2).mean()
            critic_loss = (return_ - value).pow(2).mean()

            loss = 0.5 * critic_loss + actor_loss - 0.001 * entropy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    return loss

# Util code to read image
def read_images(path, indices, image_path, img_size, device):
    image_t = []
    for i, img_ind in enumerate(indices):
        cur_path = image_path.substitute(game_session = path[i], imgind = int(img_ind))
        image = cv2.imread(cur_path)
        image = cv2.resize(image, (img_size, img_size))
        image_t.append(image)

    image_t = np.array(image_t)
    image_t = image_t.transpose(0, 3, 1, 2)
    image_t = torch.Tensor(image_t).to(device)

    return image_t