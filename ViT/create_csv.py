import os
import sys
import random
import csv

import numpy as np
import torch
import torch.utils
import torch.utils.data
import torch.utils.data.dataloader
import torchvision
import torchvision.transforms.v2 as v2
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from vit_pytorch.vit_pytorch.vit import ViT
from vit_pytorch.vit_pytorch.vivit import ViT as VideoViT


#add path to import data utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.data_utils import DataUtils as d_u
from utils.datasets import SingleGameDataset, MultiGameDataset

img_size = 512
frames = 10

reshape_for_vivit = lambda x: x.transpose(0, 1)

x_test_transform = v2.Compose([
    v2.Resize((img_size, img_size)),
    reshape_for_vivit
])

train_game_names = ['Barbie', 'Kawaii_Fire_Station', 'Kawaii_Playroom', 'Kawaii_Police_Station']
test_game_names = ['Kawaii_House', 'Kawaii_Daycare']
test_sessions = d_u.read_txt("/data/kraw084/COMPSCI715/datasets/final_data_splits/test.txt")

col_pred = ["thumbstick_left_x", "thumbstick_left_y", "thumbstick_right_x", "thumbstick_right_y", "head_pos_x", "head_pos_y", "head_pos_z", "head_dir_a", "head_dir_b", "head_dir_c", "head_dir_d"]

test_set = MultiGameDataset(test_game_names, test_sessions, cols_to_predict=col_pred, frame_count=frames, transform=x_test_transform)

# Normalisation
thumbstick_start = 2 + frames - 1
thumbsticks_loc = thumbstick_start + 4
head_pos_loc = thumbsticks_loc + 3

test_set.df[test_set.df.columns[thumbstick_start:thumbsticks_loc]] = (test_set.df[test_set.df.columns[thumbstick_start:thumbsticks_loc]] + 1) / 2
test_set.df[test_set.df.columns[thumbsticks_loc:head_pos_loc]] = (test_set.df[test_set.df.columns[thumbsticks_loc:head_pos_loc]] + 2) / 4
test_set.df[test_set.df.columns[head_pos_loc:]] = (test_set.df[test_set.df.columns[head_pos_loc:]] + 1) / 2

gpu_num = 0
device = torch.device(f'cuda:{gpu_num}')
print(f"Using device {gpu_num}: {torch.cuda.get_device_properties(gpu_num).name}")

#create ViT model
model = VideoViT(image_size = img_size,
                image_patch_size = 64,
                frame_patch_size = 2,
                num_classes = len(test_set.cols_to_predict),
                dim = 256,
                spatial_depth = 4,
                temporal_depth = 4,
                heads = 10,
                mlp_dim = 512,
                dropout = 0.3,
                emb_dropout = 0.1,
                frames = frames,
                variant = "factorized_encoder").to(device)

model.load_state_dict(torch.load(f"models/vivit_multigame_full_controls/Epoch15.pt", weights_only=True))
model.eval()

csv_file = open("COMPSCI715/ViT/csvs/vivit_preds.csv", "w")
headers = ["game_session", "frame","thumbstick_left_x","thumbstick_left_y","thumbstick_right_x","thumbstick_right_y","head_pos_x","head_pos_y","head_pos_z","head_dir_a","head_dir_b","head_dir_c","head_dir_d"]
writer = csv.DictWriter(csv_file, fieldnames=headers)
writer.writeheader()

for i in tqdm(range(len(test_set))):
    df_row = test_set.df.iloc[i]
    model_input, _ = test_set[i]
    model_pred = model(model_input.unsqueeze(0).to(device))[0]
    
    row1 = {"game_session":df_row["game_session"], "frame":df_row["frame_0"]}
    row2 = {headers[i + 2]:model_pred[i].item() for i in range(len(model_pred))}
    
    for key, value in row2.items():
        row1[key] = value
    
    writer.writerow(row1)
    
