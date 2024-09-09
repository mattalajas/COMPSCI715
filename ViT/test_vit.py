import os
import sys

import torch
import torchvision
from vit_pytorch.vit_pytorch.vit import ViT

#add path to import data utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import utils.data_utils as d_u
from train_vit import evaluate_model


img_size = 256
x_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((img_size, img_size))
])

val_sessions = d_u.DataUtils.read_txt("COMPSCI715/datasets/barbie_demo_dataset/val.txt")
val_set = d_u.SingleGameDataset("Barbie", val_sessions, transform=x_transform)

test_sessions = d_u.DataUtils.read_txt("COMPSCI715/datasets/barbie_demo_dataset/test.txt")
test_set = d_u.SingleGameDataset("Barbie", test_sessions, transform=x_transform)


gpu_num = 4
device = torch.device(f'cuda:{gpu_num}')
print(f"Using device {gpu_num}: {torch.cuda.get_device_properties(gpu_num).name}")


model = ViT(image_size = img_size,
            patch_size = 32,
            num_classes = len(val_set.cols_to_predict),
            dim = 1024,
            depth = 6,
            heads = 16,
            mlp_dim = 2048,
            dropout = 0.1,
            emb_dropout = 0.1).to(device)


model_path = "models/vit_v2/Epoch6.pt"

model.load_state_dict(torch.load(model_path, weights_only=True))

loss = evaluate_model(model, val_set, 128, torch.nn.MSELoss(), device)
print(f"Model val loss: {loss}")

loss = evaluate_model(model, test_set, 128, torch.nn.MSELoss(), device)
print(f"\nModel test loss: {loss}")