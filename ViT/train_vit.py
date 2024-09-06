import os
import sys

import torch
import torch.utils
import torch.utils.data
import torch.utils.data.dataloader
import torchvision
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from vit_pytorch.vit_pytorch.vit import ViT

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
    dataloader = torch.utils.data.dataloader.DataLoader(train_dataset, batch_size=batch_size)
    
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
        
        if i % 10 == 0 and i != 0:
            eval_loss = evaluate_model(model, val_dataset, batch_size, loss_func, device)
            print(f"Validation loss (MSE) after epoch {i}: {eval_loss}\n")
            
            tboard.add_scalar("Loss/val", eval_loss, i)
            
        torch.save(model.state_dict(), save_path + f"/Epoch{i}.pt")
        
    print("\nTraining finished")


def evaluate_model(model, dataset, batch_size, loss_func, device):
    model.eval()
    dataloader = torch.utils.data.dataloader.DataLoader(dataset, batch_size=batch_size)
    
    total_loss = 0
    with torch.no_grad():
        for X, Y in tqdm(dataloader, desc = "Evaluating model", bar_format = "{l_bar}{bar:20}{r_bar}"):
            X, Y = X.to(device), Y.to(device)
            prediction = model(X)
            total_loss += loss_func(prediction, Y)
            
    return total_loss/len(dataloader)



if __name__ == "__main__":
    img_size = 256
    x_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((img_size, img_size))
    ])
    
    #setup train and validation sets
    train_sessions = d_u.DataUtils.read_txt("COMPSCI715/datasets/barbie_demo_dataset/train.txt")
    train_set = d_u.SingleGameDataset("Barbie", train_sessions, transform=x_transform)
    
    val_sessions = d_u.DataUtils.read_txt("COMPSCI715/datasets/barbie_demo_dataset/val.txt")
    val_set = d_u.SingleGameDataset("Barbie", val_sessions, transform=x_transform)

    #select device
    gpu_num = 5
    device = torch.device(f'cuda:{gpu_num}')
    print(f"Using device {gpu_num}: {torch.cuda.get_device_properties(gpu_num).name}")
    
    #create ViT model
    model = ViT(image_size = img_size,
                patch_size = 32,
                num_classes = len(train_set.cols_to_predict),
                dim = 1024,
                depth = 6,
                heads = 16,
                mlp_dim = 2048,
                dropout = 0.1,
                emb_dropout = 0.1).to(device)

    lr = 1e-4
    loss_func = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    save_dir = "models/vit_v1"
    
    #train the model
    train_model(model = model,
                train_dataset = train_set,
                val_dataset = val_set,
                epochs = 50,
                batch_size = 128,
                loss_func = loss_func,
                opt = optimizer,
                device = device,
                save_path = save_dir,
                resume = 18)



    
