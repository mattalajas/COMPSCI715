from flask import Flask, request, jsonify
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
from PIL import Image
import numpy as np
from torchvision import transforms
from torchvision.io import read_image


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
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load a pre-trained model from timm
model = timm.create_model('resnet50', pretrained=True)

# Modify the final layer for regression according to cols_to_predict
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(cols_to_predict_value))




model.load_state_dict(torch.load('/data/ysun209/app/0_python/0_git/transfer-learning/ResNet/models/ResNet50-MultipleGame_20241004.pth', map_location=device))
model = model.to(device)
model.eval()


transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the image
    transforms.ToTensor(),    # Convert the image to a tensor
])



app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part', 400
    
    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400

    file.save(os.path.join(UPLOAD_FOLDER, file.filename))
    

    image_path = './uploads/screenshot.png'  # Replace with your image path
    image = Image.open(image_path)
    image_tensor = transform(image)

    image_tensor = image_tensor.unsqueeze(0)  # Now image_tensor is (1, C, H, W)


    outputs = model(image_tensor.to(device).float())
    print(outputs)
    outputs_np = outputs.detach().cpu().numpy()
    outputs_list = outputs_np.tolist()
    print(outputs_list[0])
    column_names = [
        "thumbstick_left_x", "thumbstick_left_y",
        "thumbstick_right_x", "thumbstick_right_y",
        "head_pos_x", "head_pos_y", "head_pos_z",
        "head_dir_a", "head_dir_b", "head_dir_c", "head_dir_d"
    ]
    outputs_dict = {col: outputs_list[0][i] for i, col in enumerate(column_names)}

    return jsonify(outputs=outputs_dict)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)