import os
import sys

import torch
import torchvision
from vit_pytorch.vit_pytorch.vit import ViT
from vit_pytorch.vit_pytorch.vivit import ViT as VideoViT

#add path to import data utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import utils.datasets as d_u
from train_vit import evaluate_model, norm_dataset

#set image size and number of frames (for ViViT)
img_size = 512
frames = 10

#function to reshape image batch tensors for ViVit models
reshape_for_vivit = lambda x: x.transpose(0, 1)

#setup image validation augmentation (just formats image to right shape and size)
x_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((img_size, img_size)),
    reshape_for_vivit
])

#control items the model is trained to predict
col_pred = ["thumbstick_left_x", "thumbstick_left_y", "thumbstick_right_x", "thumbstick_right_y", "head_pos_x", "head_pos_y", "head_pos_z", "head_dir_a", "head_dir_b", "head_dir_c", "head_dir_d"]

#setup validation and testsets
test_game_names = ['Kawaii_House', 'Kawaii_Daycare']
val_sessions = d_u.read_txt("/data/kraw084/COMPSCI715/datasets/final_data_splits/val.txt")
test_sessions = d_u.read_txt("/data/kraw084/COMPSCI715/datasets/final_data_splits/test.txt")

col_pred = ["thumbstick_left_x", "thumbstick_left_y", "thumbstick_right_x", "thumbstick_right_y", "head_pos_x", "head_pos_y", "head_pos_z", "head_dir_a", "head_dir_b", "head_dir_c", "head_dir_d"]

val_set = d_u.MultiGameDataset(test_game_names, val_sessions, cols_to_predict=col_pred, frame_count=frames, transform=x_transform)
test_set = d_u.MultiGameDataset(test_game_names, test_sessions, cols_to_predict=col_pred, frame_count=frames, transform=x_transform)

# Normalisation
thumbstick_start = 2 + frames - 1
thumbsticks_loc = thumbstick_start + 4
head_pos_loc = thumbsticks_loc + 3

norm_dataset(val_set, thumbstick_start, thumbsticks_loc, head_pos_loc)
norm_dataset(test_set, thumbstick_start, thumbsticks_loc, head_pos_loc)


#select device
gpu_num = 4
device = torch.device(f'cuda:{gpu_num}')
print(f"Using device {gpu_num}: {torch.cuda.get_device_properties(gpu_num).name}")

#create ViViT model object
model = VideoViT(image_size = img_size,
                    image_patch_size = 64,
                    frame_patch_size = 2,
                    num_classes = len(val_set.cols_to_predict),
                    dim = 256,
                    spatial_depth = 4,
                    temporal_depth = 4,
                    heads = 10,
                    mlp_dim = 512,
                    dropout = 0.2,
                    emb_dropout = 0.1,
                    frames = frames,
                    variant = "factorized_encoder").to(device)

#put model in eval model and load weights
model.eval()
model_path = "models/vivit_multigame_full_controls/Epoch15.pt"
model.load_state_dict(torch.load(model_path, weights_only=True))

#compute validation loss
loss = evaluate_model(model, val_set, 128, torch.nn.MSELoss(), device)
print(f"Model val loss: {loss}")

#compute test loss
loss = evaluate_model(model, test_set, 128, torch.nn.MSELoss(), device)
print(f"\nModel test loss: {loss}")