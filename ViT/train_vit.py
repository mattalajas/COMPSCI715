import os
import sys
import random

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
import utils.data_utils as d_u


def process_batch(batch_X, batch_Y, model, loss_func, opt):
    prediction = model(batch_X)
    loss = loss_func(prediction, batch_Y)
    
    loss.backward()
    opt.step()
    opt.zero_grad()
    
    return loss.item()
  

def train_model(model, train_dataset, val_dataset, epochs, batch_size, loss_func, opt, device, save_path, resume=None):
    dataloader = torch.utils.data.dataloader.DataLoader(train_dataset, batch_size=batch_size, num_workers=8)
    
    tboard = SummaryWriter(save_path)
    
    start_epoch = 0
    if resume:
        start_epoch = resume + 1
        model.load_state_dict(torch.load(save_path + f"/Epoch{resume}.pt", weights_only=True))
    
    for i in range(start_epoch, epochs):
        model.train()
        epoch_loss = 0
        batch_loss = 0
        progress_bar = tqdm(dataloader, desc = f"Epoch {i}/{epochs}", bar_format = "{l_bar}{bar:20}{r_bar}")
 
        for X, Y in progress_bar:
            X, Y = X.to(device), Y.to(device)
   
            batch_loss = process_batch(X, Y, model, loss_func, opt)
            epoch_loss += batch_loss
            
            progress_bar.set_postfix({"loss": batch_loss})

            
        tboard.add_scalar("Loss/train", epoch_loss/len(dataloader), i)
        print(f"Epoch {i} finished - Avg loss: {epoch_loss/len(dataloader)}\n")
        
        if i % 1 == 0: # and i != 0:
            eval_loss = evaluate_model(model, val_dataset, batch_size, loss_func, device)
            print(f"Validation loss (MSE) after epoch {i}: {eval_loss}\n")
            
            tboard.add_scalar("Loss/val", eval_loss, i)
            
        torch.save(model.state_dict(), save_path + f"/Epoch{i}.pt")
        
    print("\nTraining finished")


def evaluate_model(model, dataset, batch_size, loss_func, device):
    model.eval()
    dataloader = torch.utils.data.dataloader.DataLoader(dataset, batch_size=batch_size, num_workers=8)
    total_loss = 0
    with torch.no_grad():
        for X, Y in tqdm(dataloader, desc = "Evaluating model", bar_format = "{l_bar}{bar:20}{r_bar}"):
            X, Y = X.to(device), Y.to(device)
            prediction = model(X)
            total_loss += loss_func(prediction, Y)
            
    return total_loss/len(dataloader)



if __name__ == "__main__":
    img_size = 512
    frames = 12
    
    reshape_for_vivit = lambda x: x.transpose(0, 1)
    
    x_train_transform = v2.Compose([
        v2.Resize((img_size, img_size)),
        reshape_for_vivit
        #v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        #v2.RandomAffine(degrees = 5, translate=(0.1, 0.1)),
        #v2.GaussianNoise()
    ])
    
    
    x_test_transform = v2.Compose([
        v2.Resize((img_size, img_size)),
        reshape_for_vivit
    ])
    
    #setup train and validation sets
    train_sessions = d_u.DataUtils.read_txt("COMPSCI715/datasets/barbie_demo_dataset/train.txt")
    train_set = d_u.SingleGameDataset("Barbie", train_sessions, transform=x_train_transform, frame_count=frames)
    
    val_sessions = d_u.DataUtils.read_txt("COMPSCI715/datasets/barbie_demo_dataset/val.txt")
    val_set = d_u.SingleGameDataset("Barbie", val_sessions, transform=x_test_transform, frame_count=frames)

    #select device
    gpu_num = 5
    device = torch.device(f'cuda:{gpu_num}')
    print(f"Using device {gpu_num}: {torch.cuda.get_device_properties(gpu_num).name}")
    
    #create ViT model
    model = VideoViT(image_size = img_size,
                    image_patch_size = 64,
                    frame_patch_size = 2,
                    num_classes = len(train_set.cols_to_predict),
                    dim = 256,
                    spatial_depth = 4,
                    temporal_depth = 4,
                    heads = 10,
                    mlp_dim = 512,
                    dropout = 0.3,
                    emb_dropout = 0.1,
                    frames = frames,
                    variant = "factorized_encoder").to(device)
    

    lr = 1e-4
    weight_decay = 1e-5
    loss_func = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    save_dir = "models/vivit_v1"
    
    #train the model
    train_model(model = model,
                train_dataset = train_set,
                val_dataset = val_set,
                epochs = 100,
                batch_size = 128,
                loss_func = loss_func,
                opt = optimizer,
                device = device,
                save_path = save_dir,
                resume = None)


#venv/bin/python COMPSCI715/ViT/train_vit.py

    
