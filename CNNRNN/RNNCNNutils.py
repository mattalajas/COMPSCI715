import glob
import os
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

game_name = 'Barbie'
dir = r"..\Vrnet"
interval = 5

def create_csvs(game_name, dir, interval):
    for filename in glob.glob(f"{dir}\{game_name}\*\data_file.csv"):
        print(filename)
        cur_csv = pd.read_csv(filename)

        final_csv = pd.DataFrame(data = {'frame': cur_csv['frame'], 'thumbstick': cur_csv['Thumbstick']})
        final_csv.dropna(how='any', inplace=True)

        final_csv['thumbstick'] = final_csv['thumbstick'].apply(lambda x : literal_eval(str(x)))
        final_csv['frame'] = final_csv['frame'].apply(lambda x : int(x) + interval)

        final_csv[['T1', 'T2', 'T3', 'T4']] = pd.DataFrame(final_csv['thumbstick'].to_list(), index = final_csv.index)
        final_csv.drop(columns=['thumbstick'], inplace=True)

        paths = filename.split("\\")
        final_csv['path'] = f'{paths[1]}\{paths[2]}\{paths[3]}'
        final_csv.to_csv(f'{dir}/{game_name}/{paths[3]}/fin_data.csv', index=False)

def create_train_test_split(game_name, dir, device, seq_size = 150, batch_size = 3, split_ratio = 0.8, shuffle = False, iter = 1):
    path_map = {}
    counter = 0
    seqs = ()

    for filename in glob.glob(f"{dir}/{game_name}/*/fin_data.csv"):
        cur_csv = pd.read_csv(filename)
        cur_csv = cur_csv[::iter]
        cur_csv = cur_csv[:-(len(cur_csv)%seq_size)]

        path_map[counter] = filename[:-13]
        cur_csv['path'] = counter

        # Split gameplay into sequences
        cur_csv_t = torch.Tensor(cur_csv.values).to(device)
        cur_csv_t = torch.split(cur_csv_t, seq_size)

        counter += 1
        seqs = seqs + cur_csv_t

    train_seq = seqs[:int(len(seqs)*split_ratio)]
    test_seq = seqs[int(len(seqs)*split_ratio):]

    # Create batches for training and testing
    train_loader = DataLoader(train_seq, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_seq, batch_size=batch_size, shuffle=shuffle)

    return path_map, train_loader, test_loader

def filter_dataframe(game_sessions, data_frame, device, seq_size = 150, batch_size = 3, shuffle = False, iter = 1):
    path_map = {}
    counter = 0
    seqs = ()

    df_groups = data_frame.groupby('game_session')

    for game_session in game_sessions:
        cur_df = df_groups.get_group(game_session)
        cur_df = cur_df[::iter]
        cur_df = cur_df[:-(len(cur_df)%seq_size)]

        path_map[counter] = game_session # f"/data/ysun209/VR.net/videos/{game_session}" #/video/{frame}.jpg"
        cur_df['game_session'] = counter

        # Split gameplay into sequences
        cur_csv_t = torch.Tensor(cur_df.values).to(device)
        cur_csv_t = torch.split(cur_csv_t, seq_size)

        counter += 1
        seqs = seqs + cur_csv_t

    # Create batches for training and testing
    loader = DataLoader(seqs, batch_size=batch_size, shuffle=shuffle, drop_last=True)

    return path_map, loader

def image_dir_to_csv(img_path, height, width, save_path):
    df = pd.DataFrame(columns=['frame', 'R', 'G', 'B'])

    for ind, filename in enumerate(os.listdir(img_path)):
        img = cv2.imread(f'{img_path}/{filename}')
        img = cv2.resize(img, (height, width))
        img = img.reshape(-1, 3)
        img_row = {'frame': int(filename[:-4]),
                    'R': tuple(img[:, 0]),
                    'G': tuple(img[:, 1]),
                    'B': tuple(img[:, 2])}
        df.loc[ind] = img_row

    df = df.sort_values(by = ['frame'], ignore_index=True)
    df.to_csv(f'{save_path}/image.csv', index = False)
