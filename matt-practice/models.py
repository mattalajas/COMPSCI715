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


class ConvBasic(nn.Module):
    def __init__(self):
        super(ConvBasic, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(6, 18, kernel_size=5, stride = 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.hidden1 = nn.Linear(271674, 16)
        self.relu4 = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu1(self.conv1(x))
        out = self.relu2(self.conv2(out))
        flat_out = out.reshape((x.shape[0], -1))
        
        flat_out = self.relu4(self.hidden1(flat_out))
        return flat_out #16

class GridObservationMLP(nn.Module):
    def __init__(self):
        super(GridObservationMLP, self).__init__()
        self.hidden1 = nn.Linear(25, 256)
        self.relu1 = nn.ReLU(inplace=False)
        self.hidden2 = nn.Linear(256, 512)
        self.relu2 = nn.ReLU(inplace=False)

    def forward(self, x):
        out = self.relu1(self.hidden1(x.reshape(-1, 25)))
        out = self.relu2(self.hidden2(out))
        return out


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


class actionGRU(nn.Module):
    def __init__(self):
        super(actionGRU, self).__init__()
        # Check input size
        self.hid = nn.Linear(4, 16)
        self.rel = nn.ReLU(inplace=False)
        self.gru1 = nn.GRUCell(32, 512)

    def forward(self, image, action, h0):
        act_emb = self.rel(self.hid(action))
        x = torch.cat((image, act_emb), dim=1)
        x = x.reshape((h0.shape[0], -1))
        out = self.gru1(x, h0)
        return out


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        # Check input size
        self.hidden1 = nn.Linear(512, 256)
        self.relu1 = nn.ReLU()
        self.hidden2 = nn.Linear(256, 64)
        self.relu2 = nn.ReLU()
        self.hidden3 = nn.Linear(64, 4)

    def forward(self, x):
        out = self.relu1(self.hidden1(x))
        out = self.relu2(self.hidden2(out))
        out = self.hidden3(out)
        return out



class evalMLP(nn.Module):
    def __init__(self, grid_dims):
        super(evalMLP, self).__init__()
        self.x_size, self.y_size = grid_dims
        # TODO: init input size - b_t + grid_size[0]*grid_size[1]
        # TODO: change size to accomodate orientation
        self.hidden1 = nn.Linear(512,  300)
        self.relu1 = nn.ReLU()
        self.hidden3 = nn.Linear(300, 4)
        self.hidden4 = nn.Linear(300, self.x_size * self.y_size)

    def forward(self, x):
        out = self.relu1(self.hidden1(x))
        out_1 = self.hidden3(out)
        out_2 = self.hidden4(out)
        return out_2, out_1