# Code retrieved from: https://github.com/higgsfield-ai/higgsfield
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
from torch.utils.data import DataLoader, TensorDataset
from torch.distributions import Normal
from COMPSCI715.CNNRNN import models

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0., std=0.1)
        nn.init.constant_(m.bias, 0.1)

class Actor(nn.Module):
    def __init__(self, conv_out, rnn_type, rnn_emb, act_dim, final_out, dropout = 0):
        super(Actor, self).__init__()
        
        # self.conv = models.LeNet(img_size, conv_out, padding, kernel, stride, dropout)
        assert rnn_type.lower() == 'gru' or rnn_type.lower() == 'lstm' 
        
        self.rnn_type = rnn_type
        if self.rnn_type.lower() == 'gru':
            self.rnn = models.actionGRU(rnn_emb, act_dim, conv_out, dropout)
        else:
            self.rnn = models.actionLSTM(rnn_emb, act_dim, conv_out, dropout)
        
        self.mlp = models.MLP(conv_out, final_out, dropout)
        
    def forward(self, img, actions, h0, c0=0):
        # image_r = self.conv(img)

        # GRU step per image and its associated thumbstick comman
        if self.rnn_type.lower() == 'gru':
            h0 = self.rnn(img, actions, h0)
        else:
            h0 = self.rnn(img, actions, h0, c0)
        
        h0 = F.relu(h0)

        # Final prediction for each frame 
        fin = self.mlp(h0)
        return fin, h0, c0
    
class Critic(nn.Module):
    def __init__(self, conv_out, rnn_type, rnn_emb, act_dim, dropout = 0):
        super(Critic, self).__init__()
        
        # self.conv = models.LeNet(img_size, conv_out, padding, kernel, stride, dropout)
        assert rnn_type.lower() == 'gru' or rnn_type.lower() == 'lstm' 
        
        self.rnn_type = rnn_type
        if self.rnn_type.lower() == 'gru':
            self.rnn = models.actionGRU(rnn_emb, act_dim, conv_out, dropout)
        else:
            self.rnn = models.actionLSTM(rnn_emb, act_dim, conv_out, dropout)
        
        self.mlp = models.MLP(conv_out, 1, dropout)
        
    def forward(self, img, actions, h0, c0=0):
        # image_r = self.conv(img)

        # GRU step per image and its associated thumbstick comman
        if self.rnn_type.lower() == 'gru':
            h0 = self.rnn(img, actions, h0)
        else:
            h0 = self.rnn(img, actions, h0, c0)

        h0 = F.relu(h0)

        # Final prediction for each frame 
        fin = self.mlp(h0)
        return fin, h0, c0

class ActorCritic(nn.Module):
    def __init__(self, num_outputs, conv_out, rnn_type, rnn_emb, \
                 act_dim, final_out, dropout = 0, std=0.0):
        super(ActorCritic, self).__init__()
        
        self.critic = Critic(conv_out, rnn_type, rnn_emb, act_dim, dropout)
        
        # Make actor into CNN with RNN maybe
        self.actor = Actor(conv_out, rnn_type, rnn_emb, act_dim, final_out, dropout)

        self.log_std = nn.Parameter(torch.ones(1, num_outputs) * std)
        
        self.apply(init_weights)
        
    def forward(self, img, actions, h0_c, h0_a, c0_c=0, c0_a=0):
        value, h1_c, c1_c = self.critic(img, actions, h0_c, c0_c)
        mu, h1_a, c1_a = self.actor(img, actions, h0_a, c0_a)

        value = F.relu(value)
        mu = F.relu(mu)

        std = self.log_std.exp().expand_as(mu)
        dist = Normal(mu, std)
        return dist, value, h1_c, h1_a, c1_c, c1_a
    
class Discriminator(nn.Module):
    def __init__(self, num_inputs, hidden_size, rnn_type, dropout = 0):
        super(Discriminator, self).__init__()

        self.hlinear = nn.Linear(hidden_size + hidden_size, hidden_size)
        self.clinear = nn.Linear(hidden_size + hidden_size, hidden_size)

        assert rnn_type.lower() == 'gru' or rnn_type.lower() == 'lstm' 
        
        self.rnn_type = rnn_type
        if self.rnn_type.lower() == 'gru':
            self.rnn = nn.GRU(input_size=num_inputs, hidden_size=hidden_size)
        else:
            self.rnn = nn.LSTM(input_size=num_inputs, hidden_size=hidden_size)

        self.drop1 = nn.Dropout(dropout)
        self.linear1 = nn.Linear(hidden_size, hidden_size//2)
        self.linear2 = nn.Linear(hidden_size//2, 1)
        self.linear2.weight.data.mul_(0.1)
        self.linear2.bias.data.mul_(0.0)
    
    def forward(self, x, h0_c, h0_a, c0_c = 0, c0_a = 0):
        if self.rnn_type.lower() == 'gru':
            h0 = self.hlinear(torch.cat([h0_c, h0_a], 1))
            h0 = torch.unsqueeze(h0, 0)

            x, _ = self.rnn(x, h0)
        else:
            h0 = self.hlinear(torch.cat([h0_c, h0_a], 1))
            c0 = self.clinear(torch.cat([c0_c, c0_a], 1))

            h0 = torch.unsqueeze(h0, 0)
            c0 = torch.unsqueeze(c0, 0)

            x, _= self.rnn(x, h0, c0)

        x = self.drop1(x)
        x = F.tanh(self.linear1(x))
        x = F.sigmoid(self.linear2(x))

        out = torch.sigmoid(x)
        num_dims = len(out.shape)
        reduction_dims = tuple(range(1, num_dims))
        out = torch.mean(out, dim=reduction_dims)

        return out
