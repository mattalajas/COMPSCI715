import os
import sys
sys.path.insert(0, '/data/ysun209/app/0_python/COMPSCI715')
import utils.data_utils as data_utils
import torch
import torch.nn as nn
import timm
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms, datasets
from torch.utils.tensorboard import SummaryWriter
# from torch.optim import Adam
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from datetime import datetime

img_size = 224

transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((img_size, img_size))
])

# Setup train and validation sets
train_sessions = data_utils.DataUtils.read_txt("/data/ysun209/app/0_git/COMPSCI715/datasets/barbie_demo_dataset/train.txt")
train_set = data_utils.SingleGameDataset("Barbie", train_sessions, transform=transform)

val_sessions = data_utils.DataUtils.read_txt("/data/ysun209/app/0_git/COMPSCI715/datasets/barbie_demo_dataset/val.txt")
val_set = data_utils.SingleGameDataset("Barbie", val_sessions, transform=transform)


train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_set, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)

# Set device (GPU if available, else CPU)
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")


# Load a pre-trained ResNet-50 model from timm
resnet_model = timm.create_model('resnet50', pretrained=True)

# Freeze ResNet weights (optional)
for param in resnet_model.parameters():
    param.requires_grad = False # True or False

# Modify ResNet output
resnet_model.fc = nn.Linear(resnet_model.fc.in_features, 128)  # Adjust output dimensions

# # Define additional layers for controller data generation
# controller_layers = nn.Sequential(
#     nn.ReLU(),
#     nn.Linear(128, 64),
#     nn.ReLU(),
#     nn.Linear(64, 4)  # Adjust output dimensions
# )

# Define additional layers for controller data generation
controller_layers = nn.Sequential(
    nn.Linear(128, 128),
    nn.ReLU(),
    nn.BatchNorm1d(128),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.BatchNorm1d(64),
    nn.Dropout(0.5),  # Dropout layer for regularization
    nn.Linear(64, 4)  # Adjust output dimensions
)


# Combine ResNet with additional layers
class VideoFrameModel(nn.Module):
    def __init__(self):
        super(VideoFrameModel, self).__init__()
        self.resnet = resnet_model
        self.controller_layers = controller_layers

    def forward(self, x):
        x = self.resnet(x)
        x = self.controller_layers(x)
        return x

model = VideoFrameModel()

# Move model to the device (GPU)
model = model.to(device)


# Define loss function and optimizer
criterion = nn.MSELoss()  # Use MSE for regression
# optimizer = Adam(model.parameters(), lr=1e-4)
optimizer = AdamW(model.parameters(), lr=1e-4)

# Define the L1 regularization strength (hyperparameter)
l1_lambda = 1e-5  # You can adjust this value

# Define a learning rate scheduler
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

# Initialize TensorBoard writer
writer = SummaryWriter(log_dir="./runs/TL_ResNet50_Additional_Layer-L1-regularzation")

# Training loop
num_epochs = 100

# Training loop with validation
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, targets in train_loader:
        images, targets = images.to(device), targets.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, targets)

        # L1 regularization
        l1_norm = sum(p.abs().sum() for p in model.parameters())
        loss += l1_lambda * l1_norm

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
        for images, targets in val_loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)
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

    
