import os
import sys

import torch
import torchvision
from vit_pytorch.vit_pytorch.vit import ViT
from vit_pytorch.vit_pytorch.vivit import ViT as VideoViT

#add path to import data utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import utils.data_utils as d_u
from train_vit import evaluate_model


img_size = 512
frames = 10

reshape_for_vivit = lambda x: x.transpose(0, 1)
    
x_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((img_size, img_size)),
    reshape_for_vivit
])

val_sessions = d_u.DataUtils.read_txt("COMPSCI715/datasets/barbie_demo_dataset/val.txt")
val_set = d_u.SingleGameDataset("Barbie", val_sessions, transform=x_transform, frame_count=frames)

test_sessions = d_u.DataUtils.read_txt("COMPSCI715/datasets/barbie_demo_dataset/test.txt")
test_set = d_u.SingleGameDataset("Barbie", test_sessions, transform=x_transform, frame_count=frames)


gpu_num = 4
device = torch.device(f'cuda:{gpu_num}')
print(f"Using device {gpu_num}: {torch.cuda.get_device_properties(gpu_num).name}")


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


model_path = "models/vivit_v1/Epoch19.pt"

model.load_state_dict(torch.load(model_path, weights_only=True))

loss = evaluate_model(model, val_set, 128, torch.nn.MSELoss(), device)
print(f"Model val loss: {loss}")

loss = evaluate_model(model, test_set, 128, torch.nn.MSELoss(), device)
print(f"\nModel test loss: {loss}")