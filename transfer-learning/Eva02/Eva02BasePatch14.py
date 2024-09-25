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
from torch.optim import Adam
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# Set image size for the model (224x224 is the default for EVA02 and ResNet models)
img_size = 224

# Image preprocessing pipeline
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((img_size, img_size))
])

# Setup train and validation sets
cols_to_predict_value = ["thumbstick_left_x", "thumbstick_left_y", "thumbstick_right_x", "thumbstick_right_y"]

train_sessions = data_utils.DataUtils.read_txt("/data/ysun209/app/0_git/COMPSCI715/datasets/barbie_demo_dataset/train.txt")
train_set = data_utils.SingleGameDataset("Barbie", train_sessions, transform=transform, cols_to_predict=cols_to_predict_value)

val_sessions = data_utils.DataUtils.read_txt("/data/ysun209/app/0_git/COMPSCI715/datasets/barbie_demo_dataset/val.txt")
val_set = data_utils.SingleGameDataset("Barbie", val_sessions, transform=transform, cols_to_predict=cols_to_predict_value)

# Create data loaders
train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_set, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)

# Set device (GPU if available, else CPU)
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

# Load the EVA02 model pre-trained on ImageNet
model = timm.create_model('eva02_base_patch14_224', pretrained=True)

# Modify the final layer for regression according to cols_to_predict
if isinstance(model.head, nn.Identity):
    # If the model's head is an identity layer, add a new head for regression
    num_features = model.embed_dim  # Use embed_dim for transformer models
else:
    num_features = model.head.in_features  # Standard case

# Replace the head with a new fully connected layer for regression
model.head = nn.Linear(num_features, len(cols_to_predict_value))  # Adjust head for regression

# Move the model to the device (GPU)
model = model.to(device)

# Set all parameters to be trainable (fine-tuning)
for param in model.parameters():
    param.requires_grad = True

# Define the loss function and optimizer
criterion = nn.MSELoss()  # MSE Loss for regression
optimizer = Adam(model.parameters(), lr=1e-4)  # Adam optimizer with learning rate 1e-4

# Initialize TensorBoard writer for logging
current_file_name = os.path.splitext(os.path.basename(__file__))[0]
writer = SummaryWriter(log_dir=f"./runs/{current_file_name}")

# Training settings
num_epochs = 100

# Training loop with validation
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    
    model.train()
    running_loss = 0.0

    for images, targets in tqdm(train_loader, desc="Training", leave=False):
        images, targets = images.to(device), targets.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, targets)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {running_loss/len(train_loader)}")
    writer.add_scalar('Training Loss', running_loss / len(train_loader), epoch)

    # Validation phase
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc="Validation", leave=False):
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)

    print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_val_loss}")
    writer.add_scalar('Validation Loss', avg_val_loss, epoch)

# Save model after training
torch.save(model.state_dict(), f"models/{current_file_name}.pth")
