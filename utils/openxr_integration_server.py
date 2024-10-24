import json
import os
import sys

sys.path.insert(0, os.getcwd())
import torch
import torch.nn as nn
import timm
import torchvision
import numpy as np
from torchvision import transforms

import zmq
import cv2
import numpy as np
from image_pb2 import Image

column_names = [
    "left_thumbstick_x",
    "left_thumbstick_y",
    "right_thumbstick_x",
    "right_thumbstick_y",
    "head_pos_x",
    "head_pos_y",
    "head_pos_z",
    "head_dir_w", # not sure if it is wxyz or xyzw
    "head_dir_x",
    "head_dir_y",
    "head_dir_z",
]

# Set device (GPU if available, else CPU)
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load a pre-trained model from timm
model = timm.create_model("resnet50", pretrained=True)

# Modify the final layer for regression according to cols_to_predict
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(column_names))

# model.load_state_dict(
#     torch.load(
#         "/data/ysun209/app/0_python/0_git/transfer-learning/ResNet/models/ResNet50-MultipleGame_20241004.pth",
#         map_location=device,
#     )
# )
model = model.to(device)
model.eval()

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5555")

while True:
    print("Waiting for request...")
    message = socket.recv()

    image = Image()
    image.ParseFromString(message)

    img_array = np.frombuffer(image.data, dtype=np.uint8)
    img = img_array.reshape((image.height, image.width, image.channels))

    # Convert cv2 image to tensor
    processed_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    image_tensor = transforms.ToTensor()(processed_image)

    image_tensor = image_tensor.unsqueeze(0)  # Now image_tensor is (1, C, H, W)

    outputs = model(image_tensor.to(device).float())
    print(outputs)
    outputs_np = outputs.detach().cpu().numpy()
    outputs_list = outputs_np.tolist()
    print(outputs_list[0])
    outputs_dict = {col: outputs_list[0][i] for i, col in enumerate(column_names)}

    response = json.dumps(outputs_dict)
    print(response)
    socket.send_string(response)
