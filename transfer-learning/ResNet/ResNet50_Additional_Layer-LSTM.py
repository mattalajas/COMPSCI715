import os
import sys
sys.path.insert(0, '/data/ysun209/app/0_python/COMPSCI715')
import utils.data_utils as data_utils
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from datetime import datetime

# Set image size
img_size = 224

# Define transformations
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((img_size, img_size))
])

# Setup train and validation sets
cols_to_predict_value = ["thumbstick_left_x", "thumbstick_left_y", "thumbstick_right_x", "thumbstick_right_y", "head_dir_a", "head_dir_b", "head_dir_c", "head_dir_d", "head_pos_x", "head_pos_y", "head_pos_z"]

train_sessions = data_utils.DataUtils.read_txt("/data/ysun209/app/0_git/COMPSCI715/datasets/barbie_demo_dataset/train.txt")
train_set = data_utils.SingleGameDataset("Barbie", train_sessions, transform=transform, cols_to_predict=cols_to_predict_value)

val_sessions = data_utils.DataUtils.read_txt("/data/ysun209/app/0_git/COMPSCI715/datasets/barbie_demo_dataset/val.txt")
val_set = data_utils.SingleGameDataset("Barbie", val_sessions, transform=transform, cols_to_predict=cols_to_predict_value)

# Setup data loaders
train_loader = DataLoader(train_set, batch_size=256, shuffle=True, num_workers=16, pin_memory=True)
val_loader = DataLoader(val_set, batch_size=256, shuffle=True, num_workers=16, pin_memory=True)

# Set device (GPU if available, else CPU)
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")


# Define the custom LSTM class as per the request
class actionLSTM(nn.Module):
    def __init__(self, fin_emb, act_dim, img_dim, dropout=0):
        super(actionLSTM, self).__init__()
        
        # Thumbstick (action) encoder: encodes 4-dimensional action input
        self.hid1 = nn.Linear(4, act_dim)
        
        # Concatenated vector of action embedding and image embedding
        self.hid2 = nn.Linear(act_dim + img_dim, 128)
        self.batch1 = nn.BatchNorm1d(128, track_running_stats=False)
        
        # LSTM Cell
        self.lstm1 = nn.LSTMCell(128, fin_emb)
        
        # Linear layer to map LSTM output to 4 dimensions
        self.fc_out = nn.Linear(fin_emb, 11)

        # Dropout layers
        self.drop1 = nn.Dropout(p=dropout)
        self.drop2 = nn.Dropout(p=dropout)
        self.drop3 = nn.Dropout(p=dropout)

        # Initialize weights
        self.initialise_weights()
    
    def initialise_weights(self):
        torch.nn.init.xavier_uniform_(self.hid1.weight)
        torch.nn.init.zeros_(self.hid1.bias)

        torch.nn.init.xavier_uniform_(self.hid2.weight)
        torch.nn.init.zeros_(self.hid2.bias)

        torch.nn.init.xavier_uniform_(self.lstm1.weight_hh)
        torch.nn.init.zeros_(self.lstm1.bias_hh)

        torch.nn.init.xavier_uniform_(self.lstm1.weight_ih)
        torch.nn.init.zeros_(self.lstm1.bias_ih)

        torch.nn.init.xavier_uniform_(self.fc_out.weight)
        torch.nn.init.zeros_(self.fc_out.bias)

    def forward(self, image, action, h0, c0):
        # Encodes thumbstick (action) output using MLP
        act_emb = F.relu(self.hid1(action))
        act_emb = self.drop1(act_emb)

        # Concatenates thumbstick encoding and image encoding
        x = torch.cat((image, act_emb), dim=1)
        x = self.drop2(x)

        x = F.relu(self.batch1(self.hid2(x)))
        x = self.drop3(x)

        # Feeds concatenated vector to LSTM alongside hidden layer and cell state
        hx, cx = self.lstm1(x, (h0, c0))
        
        # Map LSTM output to the desired 4-dimensional output
        output = self.fc_out(hx)
        
        return output, cx

# Modify ResNet output to provide the image embeddings for LSTM
resnet_model = timm.create_model('resnet50', pretrained=True)

# Freeze ResNet weights (optional)
for param in resnet_model.parameters():
    param.requires_grad = False

resnet_model.fc = nn.Linear(resnet_model.fc.in_features, 128)  # Adjust output dimensions


class VideoFrameModelWithCustomLSTM(nn.Module):
    def __init__(self, fin_emb, act_dim, img_dim, dropout=0):
        super(VideoFrameModelWithCustomLSTM, self).__init__()
        self.resnet = resnet_model
        self.action_lstm = actionLSTM(fin_emb, act_dim, img_dim, dropout)
    
    def forward(self, images, actions, h0, c0):
        # ResNet outputs the image embedding
        img_emb = self.resnet(images)
        
        # Pass through the actionLSTM
        hx, cx = self.action_lstm(img_emb, actions, h0, c0)
        
        return hx, cx


# Example initialization and usage
fin_emb = 64  # output embedding size from LSTM
act_dim = 64  # size of the action embedding (64 dimensions)
img_dim = 128  # image embedding dimension from ResNet
dropout = 0.5  # dropout rate

# Initialize model
model = VideoFrameModelWithCustomLSTM(fin_emb, act_dim, img_dim, dropout)

# Move model to device
model = model.to(device)


# Define regularization strengths
l2_lambda = 1e-4  # L2 regularization strength (also known as weight decay)

# Define loss function and optimizer
criterion = nn.MSELoss()  # Use MSE for regression
optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=l2_lambda)

# Define a learning rate scheduler
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

# Initialize TensorBoard writer
writer = SummaryWriter(log_dir="./runs/TL_ResNet50_CustomLSTM-L2-regularization-LSTM")

# Training loop
num_epochs = 100

# Training loop with validation
for epoch in range(num_epochs):
    
    # model.train() when training starts after first epoch
    if epoch == 0:
        model.eval()  # Model is in eval mode for the first epoch
    else:
        model.train()  # Resume training mode after first epoch

    running_loss = 0.0

    for images, targets in tqdm(train_loader, desc="Training", leave=False):
        images, targets = images.to(device), targets.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Initial hidden and cell states
        batch_size = images.size(0)
        h0 = torch.zeros(batch_size, fin_emb).to(device)
        c0 = torch.zeros(batch_size, fin_emb).to(device)

        # Forward pass with actions and image inputs
        actions = torch.randn(batch_size, 4).to(device)  # Random actions (replace with actual data)
        output, cx = model(images, actions, h0, c0)
        
        # Compute loss (compare output with targets)
        loss = criterion(output, targets)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {running_loss/len(train_loader)}")
    writer.add_scalar('Training Loss', running_loss / len(train_loader), epoch)

    # Validation phase
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc="Validation", leave=False):
            images, targets = images.to(device), targets.to(device)
            
            # Initial hidden and cell states for validation
            batch_size = images.size(0)
            h0 = torch.zeros(batch_size, fin_emb).to(device)
            c0 = torch.zeros(batch_size, fin_emb).to(device)
            
            actions = torch.randn(batch_size, 4).to(device)  # Random actions (replace with actual data)
            hx, cx = model(images, actions, h0, c0)
            
            loss = criterion(hx, targets)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_val_loss}")
    writer.add_scalar('Validation Loss', avg_val_loss, epoch)

    # Step the scheduler
    scheduler.step(avg_val_loss)

    # Get the current learning rate
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Current Learning Rate: {current_lr:.6f}")
    writer.add_scalar('Learning Rate', current_lr, epoch)

# Save model after training
current_file_name = os.path.splitext(os.path.basename(__file__))[0]
torch.save(model.state_dict(), f"models/{current_file_name}.pth")
