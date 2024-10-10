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
from utils.data_utils import DataUtils as d_u
from utils.datasets import SingleGameDataset, MultiGameDataset

#number of workers for dataloaders
workers = 16

def process_batch(batch_X, batch_Y, model, loss_func, opt):
    """
    Foward and backwards pass for a single batch
    Args:
        batch_X: image tensor (for model input)
        batch_Y: control vector tensors
        model: ViT model
        loss_func: torch loss function
        opt: torch optimizer
    Returns:
        loss: loss for the batch
    """
    #foward pass and compute loss
    prediction = model(batch_X)
    loss = loss_func(prediction, batch_Y)
    
    #backpropagate and update model
    loss.backward()
    opt.step()
    opt.zero_grad()
    
    return loss.item()
  

def train_model(model, train_dataset, val_dataset, epochs, batch_size, loss_func, opt, device, save_path, resume=None, sched=None):
    """
    Trains the vit model with the given datasets and hyperparmeters
    Args:
        model: ViT model
        train_dataset: pytorch training dataset
        val_dataset: pytorch validation dataset
        epochs: number of epochs to train
        batch_size: batch size
        loss_func: torch loss function
        opt: torch optimizer
        device: device to train on (cuda or cpu)
        save_path: path to save the model and tensorboard logs
        resume: epoch to resume training from (none to start from scratch)
        sched: learning rate scheduler
    """
    #create training dataloder and tensorboard
    dataloader = torch.utils.data.dataloader.DataLoader(train_dataset, batch_size=batch_size, num_workers=workers)
    tboard = SummaryWriter(save_path)
    
    #load model if resuming from checkpoint
    start_epoch = 0
    if resume:
        start_epoch = resume + 1
        model.load_state_dict(torch.load(save_path + f"/Epoch{resume}.pt", weights_only=True))
    
    #main epoch loop
    for i in range(start_epoch, epochs):
        model.train()
        epoch_loss = 0
        batch_loss = 0
        
        #wrap dataloader in progress bar
        progress_bar = tqdm(dataloader, desc = f"Epoch {i}/{epochs}", bar_format = "{l_bar}{bar:20}{r_bar}")
 
        #loop over batches in dataloader
        for X, Y in progress_bar:
            X, Y = X.to(device), Y.to(device)
   
            #forward and backward pass
            batch_loss = process_batch(X, Y, model, loss_func, opt)
            epoch_loss += batch_loss
            
            #append batch loss to progress bar
            progress_bar.set_postfix({"loss": batch_loss})

        #add epoch loss to tensorboard and print
        tboard.add_scalar("Loss/train", epoch_loss/len(dataloader), i)
        print(f"Epoch {i} finished - Avg loss: {epoch_loss/len(dataloader)}\n")
        
        #frequency to evaluate on validation set
        save_freq = 1
        if i % save_freq == 0:
            eval_loss = evaluate_model(model, val_dataset, batch_size, loss_func, device)
            print(f"Validation loss (MSE) after epoch {i}: {eval_loss}\n")
            
            tboard.add_scalar("Loss/val", eval_loss, i)
        
        #save model checkpoint       
        torch.save(model.state_dict(), save_path + f"/Epoch{i}.pt")
        
        #if using a lr schedular, update it
        if sched:
            sched.step()
        
    print("\nTraining finished")


def evaluate_model(model, dataset, batch_size, loss_func, device):
    """
    Evaluate the model on the given (validation or test) dataset
    Args:
        model: ViT model
        dataset: pytorch dataset
        batch_size: batch size
        loss_func: torch loss function
        device: device to evaluate on (cuda or cpu)
    Returns:
        loss: average loss over the dataset
    """
    #put model in eval mode and create dataloader
    model.eval()
    dataloader = torch.utils.data.dataloader.DataLoader(dataset, batch_size=batch_size, num_workers=workers)
    total_loss = 0
    
    #loop over batches and compute the loss
    with torch.no_grad():
        for X, Y in tqdm(dataloader, desc = "Evaluating model", bar_format = "{l_bar}{bar:20}{r_bar}"):
            X, Y = X.to(device), Y.to(device)
            prediction = model(X)
            total_loss += loss_func(prediction, Y)
            
    return total_loss/len(dataloader)



if __name__ == "__main__":
    #set image size and number of frames (frames should be 1 is using VIT)
    img_size = 512
    frames = 10
    
    #function to reshape image batch tensors for ViVit models
    reshape_for_vivit = lambda x: x.transpose(0, 1)
    
    #setup image training augmentation
    x_train_transform = v2.Compose([
        v2.Resize((img_size, img_size)),
        reshape_for_vivit
        #v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        #v2.RandomAffine(degrees = 5, translate=(0.1, 0.1)),
        #v2.GaussianNoise()
    ])
    
    #setup image validation augmentation (just formats image to right shape and size)
    x_test_transform = v2.Compose([
        v2.Resize((img_size, img_size)),
        reshape_for_vivit
    ])
    
    #control items for the models to predict
    col_pred = ["thumbstick_left_x", "thumbstick_left_y", "thumbstick_right_x", "thumbstick_right_y", "head_pos_x", "head_pos_y", "head_pos_z", "head_dir_a", "head_dir_b", "head_dir_c", "head_dir_d"]
    
    
    #change game datasets by commenting out the code between the lines below
    
    #Barbie dataset -----------------------------------------------------------------------------
    #train_sessions = d_u.read_txt("COMPSCI715/datasets/barbie_demo_dataset/train.txt")
    #val_sessions = d_u.read_txt("COMPSCI715/datasets/barbie_demo_dataset/val.txt")
    
    #train_set = SingleGameDataset("Barbie", train_sessions, transform=x_train_transform, frame_count=frames, cols_to_predict=col_pred)
    #val_set = SingleGameDataset("Barbie", val_sessions, transform=x_test_transform, frame_count=frames, cols_to_predict=col_pred)
    #--------------------------------------------------------------------------------------------
    
    #Multi game dataset ---------------------------------------------------------------------------
    train_game_names = ['Barbie', 'Kawaii_Fire_Station', 'Kawaii_Playroom', 'Kawaii_Police_Station']
    val_game_names = ['Kawaii_House', 'Kawaii_Daycare']
    train_sessions =d_u.read_txt("/data/kraw084/COMPSCI715/datasets/final_data_splits/train.txt")
    val_sessions = d_u.read_txt("/data/kraw084/COMPSCI715/datasets/final_data_splits/val.txt")
    
    train_set = MultiGameDataset(train_game_names, train_sessions, cols_to_predict=col_pred, frame_count=frames, transform=x_train_transform)
    val_set = MultiGameDataset(val_game_names, val_sessions, cols_to_predict=col_pred, frame_count=frames, transform=x_test_transform)
    #----------------------------------------------------------------------------------------------


    # Normalisation
    thumbstick_start = 2 + frames - 1
    thumbsticks_loc = thumbstick_start + 4
    head_pos_loc = thumbsticks_loc + 3

    train_set.df[train_set.df.columns[thumbstick_start:thumbsticks_loc]] = (train_set.df[train_set.df.columns[thumbstick_start:thumbsticks_loc]] + 1) / 2
    val_set.df[val_set.df.columns[thumbstick_start:thumbsticks_loc]] = (val_set.df[val_set.df.columns[thumbstick_start:thumbsticks_loc]] + 1) / 2

    train_set.df[train_set.df.columns[thumbsticks_loc:head_pos_loc]] = (train_set.df[train_set.df.columns[thumbsticks_loc:head_pos_loc]] + 2) / 4
    val_set.df[val_set.df.columns[thumbsticks_loc:head_pos_loc]] = (val_set.df[val_set.df.columns[thumbsticks_loc:head_pos_loc]] + 2) / 4
    
    train_set.df[train_set.df.columns[head_pos_loc:]] = (train_set.df[train_set.df.columns[head_pos_loc:]] + 1) / 2
    val_set.df[val_set.df.columns[head_pos_loc:]] = (val_set.df[val_set.df.columns[head_pos_loc:]] + 1) / 2
    
    
    #select device
    gpu_num = 0
    device = torch.device(f'cuda:{gpu_num}')
    print(f"Using device {gpu_num}: {torch.cuda.get_device_properties(gpu_num).name}")
    
    #create ViViT object (change to ViT and set parameters accordingly to train a single image transformer)
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
    
    #create optimizer and lr decay scheduler
    lr = 1e-4
    weight_decay = 1e-5
    loss_func = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    schedular = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.99)
    
    #set model save dir
    save_dir = "models/vivit_multigame_full_controls"
    
    #train the model
    train_model(model = model,
                train_dataset = train_set,
                val_dataset = val_set,
                epochs = 25,
                batch_size = 128,
                loss_func = loss_func,
                opt = optimizer,
                device = device,
                save_path = save_dir,
                resume = None,
                sched = schedular)

