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

# Model for Image classification
class ConvBasic(nn.Module):
    def __init__(self):
        super(ConvBasic, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(6, 18, kernel_size=3, stride=2)
        self.hidden1 = nn.Linear(4050, 512)
        self.hidden2 = nn.Linear(512, 256)
        self.hidden3 = nn.Linear(256, 16)

        self.initialise_weights()

    def initialise_weights(self):
        torch.nn.init.xavier_uniform_(self.conv1.weight)
        torch.nn.init.zeros_(self.conv1.bias)

        torch.nn.init.xavier_uniform_(self.conv2.weight)
        torch.nn.init.zeros_(self.conv2.bias)

        torch.nn.init.xavier_uniform_(self.hidden1.weight)
        torch.nn.init.zeros_(self.hidden1.bias)

        torch.nn.init.xavier_uniform_(self.hidden2.weight)
        torch.nn.init.zeros_(self.hidden2.bias)

        torch.nn.init.xavier_uniform_(self.hidden3.weight)
        torch.nn.init.zeros_(self.hidden3.bias)

    def forward(self, x):
        # Two convolutional layers for image encoding
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        flat_out = out.reshape((x.shape[0], -1))
        
        # Final encoding to transform 2 dim conv output to a single vector
        flat_out = F.relu(self.hidden1(flat_out))
        flat_out = F.relu(self.hidden2(flat_out))
        flat_out = self.hidden3(flat_out)
        return flat_out
    
class LeNet(nn.Module):
    def __init__(self, size, padding=0, kernel=5, stride=1):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=kernel, stride=stride, padding=padding)
        size1 = int((size + 2*padding - kernel)/stride)+1
        self.batch1 = nn.BatchNorm2d(6, size1, size1, track_running_stats=False)

        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=kernel, stride=stride, padding=padding)
        size2 = int((size1-2)/2)+1
        size2 = int((size2 + 2*padding - kernel)/stride)+1
        self.batch2 = nn.BatchNorm2d(16, size2, size2, track_running_stats=False)

        size3 = int((size2-2)/2)+1
        self.fc1 = nn.Linear(16*size3*size3, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 16)

        self.initialise_weights()
    
    def initialise_weights(self):
        torch.nn.init.xavier_uniform_(self.conv1.weight)
        torch.nn.init.zeros_(self.conv1.bias)

        torch.nn.init.xavier_uniform_(self.conv2.weight)
        torch.nn.init.zeros_(self.conv2.bias)

        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.zeros_(self.fc1.bias)

        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.zeros_(self.fc2.bias)

        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.zeros_(self.fc2.bias)
        

    def forward(self, x):
        out = F.relu(self.batch1(self.conv1(x)))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.batch2(self.conv2(out)))
        out = F.max_pool2d(out, 2)
        out = out.reshape(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

# This model isnt being used
# class GridObservationMLP(nn.Module):
#     def __init__(self):
#         super(GridObservationMLP, self).__init__()
#         self.hidden1 = nn.Linear(25, 256)
#         self.relu1 = nn.ReLU(inplace=False)
#         self.hidden2 = nn.Linear(256, 512)
#         self.relu2 = nn.ReLU(inplace=False)

#     def forward(self, x):
#         # Standard two layer mlp 
#         out = self.relu1(self.hidden1(x.reshape(-1, 25)))
#         out = self.relu2(self.hidden2(out))
#         return out


# TODO: Possible remove belief and action GRU classes and use nn.GRU directly
# TODO: Check GRU num_layers

# class beliefGRU(nn.Module):
#     def __init__(self):
#         super(beliefGRU, self).__init__()
#         # Check input size
#         self.gru1 = nn.GRU(516, 512, batch_first=True)

#     def forward(self, x):
#         out = self.gru1(x)
#         return out

# Model for memory module GRU
class actionGRU(nn.Module):
    def __init__(self):
        super(actionGRU, self).__init__()
        # Check input size
        self.hid1 = nn.Linear(4, 16)
        self.hid2 = nn.Linear(32, 128)
        self.batch1 = nn.BatchNorm1d(128, track_running_stats=False)
        self.gru1 = nn.GRUCell(128, 512)
    
        self.initialise_weights()
    
    def initialise_weights(self):
        torch.nn.init.xavier_uniform_(self.hid1.weight)
        torch.nn.init.zeros_(self.hid1.bias)

        torch.nn.init.xavier_uniform_(self.hid2.weight)
        torch.nn.init.zeros_(self.hid2.bias)

        torch.nn.init.xavier_uniform_(self.gru1.weight_hh)
        torch.nn.init.zeros_(self.gru1.bias_hh)

        torch.nn.init.xavier_uniform_(self.gru1.weight_ih)
        torch.nn.init.zeros_(self.gru1.bias_ih)

    def forward(self, image, action, h0):
        # Encodes thumbstick output using MLP
        act_emb = F.relu(self.hid1(action))

        # Concatenates thumbstick encoding and image encoding
        x = torch.cat((image, act_emb), dim=1)
        x = x.reshape((h0.shape[0], -1))
        x = F.relu(self.batch1(self.hid2(x)))

        # Feeds concatenated vector to GRU alongside hidden layer output
        out = self.gru1(x, h0)
        return out

# Model for memory module (LSTM)
class actionLSTM(nn.Module):
    def __init__(self):
        super(actionLSTM, self).__init__()
        # Check input size
        self.hid1 = nn.Linear(4, 16)
        self.hid2 = nn.Linear(32, 128)
        self.batch1 = nn.BatchNorm1d(128, track_running_stats=False)
        self.lstm1 = nn.LSTMCell(128, 512)
    
        self.initialise_weights()
    
    def initialise_weights(self):
        torch.nn.init.xavier_uniform_(self.hid1.weight)
        torch.nn.init.zeros_(self.hid1.bias)

        torch.nn.init.xavier_uniform_(self.hid2.weight)
        torch.nn.init.zeros_(self.hid2.bias)

        torch.nn.init.xavier_uniform_(self.lstm1.weight_hh)
        torch.nn.init.zeros_(self.lstm1.bias_hh)

        torch.nn.init.xavier_uniform_(self.lstm1.weight_ih)
        torch.nn.init.zeros_(self.lstm1.bias_ih)

    def forward(self, image, action, h0, c0):
        # Encodes thumbstick output using MLP
        act_emb = F.relu(self.hid1(action))
        # Concatenates thumbstick encoding and image encoding
        x = torch.cat((image, act_emb), dim=1)
        x = x.reshape((h0.shape[0], -1))
        x = F.relu(self.batch1(self.hid2(x)))

        # Feeds concatenated vector to LSTM alongside hidden layer and cell state
        hx, cx = self.lstm1(x, (h0, c0))
        return hx, cx 

class actionGRUdeep(nn.Module):
    def __init__(self):
        super(actionGRUdeep, self).__init__()
        # Check input size
        self.hid1 = nn.Linear(4, 12)
        self.rel1 = nn.ReLU(inplace=False)
        
        self.hid2 = nn.Linear(12, 16)
        self.rel2 = nn.ReLU(inplace=False)

        self.hid3 = nn.Linear(32, 128)
        self.rel3 = nn.ReLU(inplace=False)
        
        self.gru1 = nn.GRUCell(128, 512)
        self.rel4 = nn.ReLU(inplace=False)

    def forward(self, image, action, h0):
        # Encodes thumbstick output using MLP
        act_emb = self.rel1(self.hid1(action))
        act_emb = self.rel2(self.hid2(act_emb))

        # Concatenates thumbstick encoding and image encoding
        x = torch.cat((image, act_emb), dim=1)
        x = x.reshape(h0.shape[0], -1)

        # Feeds concatenated vector to GRU alongside hidden layer output
        x = self.rel3(self.hid3(x))

        out = self.rel4(self.gru1(x, h0))
        return out

# Standard MLP for fina prediction
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        # Check input size
        self.hidden1 = nn.Linear(512, 256)
        self.hidden2 = nn.Linear(256, 64)
        self.hidden3 = nn.Linear(64, 4)
    
        self.initialise_weights()

    def initialise_weights(self):
        torch.nn.init.xavier_uniform_(self.hidden1.weight)
        torch.nn.init.zeros_(self.hidden1.bias)

        torch.nn.init.xavier_uniform_(self.hidden2.weight)
        torch.nn.init.zeros_(self.hidden2.bias)

        torch.nn.init.xavier_uniform_(self.hidden3.weight)
        torch.nn.init.zeros_(self.hidden3.bias)

    def forward(self, x):
        out = F.relu(self.hidden1(x))
        out = F.relu(self.hidden2(out))
        out = self.hidden3(out)
        return out


# # Model isnt being used
# class evalMLP(nn.Module):
#     def __init__(self, grid_dims):
#         super(evalMLP, self).__init__()
#         self.x_size, self.y_size = grid_dims
#         # TODO: init input size - b_t + grid_size[0]*grid_size[1]
#         # TODO: change size to accomodate orientation
#         self.hidden1 = nn.Linear(512,  300)
#         self.relu1 = nn.ReLU()
#         self.hidden3 = nn.Linear(300, 4)
#         self.hidden4 = nn.Linear(300, self.x_size * self.y_size)

#     def forward(self, x):
#         out = self.relu1(self.hidden1(x))
#         out_1 = self.hidden3(out)
#         out_2 = self.hidden4(out)
#         return out_2, out_1