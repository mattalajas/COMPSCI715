import os
import sys
sys.path.insert(0, os.getcwd())
import utils.data_utils as data_utils
import utils.datasets as data_utils_datasets
import torch
import torch.nn as nn
import timm
from torch.utils.data import DataLoader
import torchvision
from torch.optim import Adam
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import numpy as np

img_size = 224

transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((img_size, img_size))
])

#setup train and validation sets
cols_to_predict_value = ["thumbstick_left_x", "thumbstick_left_y", "thumbstick_right_x", "thumbstick_right_y", "head_pos_x", "head_pos_y", "head_pos_z", "head_dir_a", "head_dir_b", "head_dir_c", "head_dir_d"]

train_game_names = ['Barbie', 'Kawaii_Fire_Station', 'Kawaii_Playroom', 'Kawaii_Police_Station']
test_game_names = ['Kawaii_House', 'Kawaii_Daycare']
val_game_names = ['Kawaii_House', 'Kawaii_Daycare']

train_sessions = data_utils.DataUtils.read_txt("./datasets/final_data_splits/train.txt")
train_set = data_utils_datasets.MultiGameDataset(train_game_names, train_sessions, transform=transform, cols_to_predict=cols_to_predict_value)

val_sessions = data_utils.DataUtils.read_txt("./datasets/final_data_splits/val.txt")
val_set = data_utils_datasets.MultiGameDataset(val_game_names, val_sessions, transform=transform, cols_to_predict=cols_to_predict_value)

test_sessions = data_utils.DataUtils.read_txt("./datasets/final_data_splits/test.txt")
test_set = data_utils_datasets.MultiGameDataset(test_game_names, test_sessions, transform=transform, cols_to_predict=cols_to_predict_value)


train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_set, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_set, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)

# Set device (GPU if available, else CPU)
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

# Load a pre-trained model from timm
model = timm.create_model('resnet50', pretrained=True)

# Modify the final layer for regression according to cols_to_predict
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(cols_to_predict_value))

model.load_state_dict(torch.load('models/ResNet50-MultipleGame.pth', map_location=device))
model = model.to(device)
model.eval()

criterion = nn.MSELoss()  # Use MSE for regression


all_predictions = []
all_targets = []

test_loss = 0.0

with torch.no_grad():  # Disable gradient computation
    index = 0
    for images, targets in tqdm(test_loader, desc="Testing"):
        images, targets = images.to(device).float(), targets.to(device).float()

        outputs = model(images)
        loss = criterion(outputs, targets)
        test_loss += loss.item()

        outputs = outputs.cpu().detach().numpy()

        new_columns = np.array([(test_set.df.iloc[index]['game_session'], test_set.df.iloc[index]['frame'])],dtype=object)

        combined_array = np.hstack((new_columns, outputs))

        # Collect predictions and ground truth
        all_predictions.append(combined_array)  # Move to CPU for easier handling
        index = index + 1

avg_test_loss = test_loss / len(test_loader)

print(f"Testing Loss: {avg_test_loss}")


column_names = [
    "game_session", "frame", "thumbstick_left_x", "thumbstick_left_y",
    "thumbstick_right_x", "thumbstick_right_y", "head_pos_x", 
    "head_pos_y", "head_pos_z", "head_dir_a", "head_dir_b", 
    "head_dir_c", "head_dir_d"
]

df_list = []

# Convert each object array to a DataFrame and append to the list
for data in all_predictions:
    df = pd.DataFrame(data, columns=column_names)
    df_list.append(df)

combined_df = pd.concat(df_list, ignore_index=True)

# Export combined DataFrame to CSV
csv_filename = "output_data.csv"
combined_df.to_csv(csv_filename, index=False)  # index=False to avoid adding row numbers
