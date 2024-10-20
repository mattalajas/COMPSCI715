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
test_loader = DataLoader(test_set, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)

# Set device (GPU if available, else CPU)
device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")

# Load a pre-trained model from timm
model = timm.create_model('mobilenetv4_conv_small', pretrained=True)

# Modify the final layer for regression according to cols_to_predict
num_features = model.classifier.in_features
model.classifier = nn.Linear(num_features, len(cols_to_predict_value))

# Move model to the device (GPU)
model = model.to(device)

# Set all parameters to be trainable
for param in model.parameters():
    param.requires_grad = True  # Fine-tune all layers

# Define loss function and optimizer
criterion = nn.MSELoss()  # Use MSE for regression
optimizer = Adam(model.parameters(), lr=1e-4)

# Initialize TensorBoard writer
current_file_name = os.path.splitext(os.path.basename(__file__))[0]
writer = SummaryWriter(log_dir=f"./runs/{current_file_name}")

# Training loop
num_epochs = 100

# Training loop with validation
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    
    model.train()
    running_loss = 0.0

    for images, targets in tqdm(train_loader, desc="Training", leave=False):
        images, targets = images.to(device).float(), targets.to(device).float()

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, targets)

        # Backward pass and optimize
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
            images, targets = images.to(device).float(), targets.to(device).float()
            outputs = model(images)
            loss = criterion(outputs, targets)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)

    print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_val_loss}")
    writer.add_scalar('Validation Loss', avg_val_loss, epoch)

    if epoch % 10 == 0:
        # Save model after training
        torch.save(model.state_dict(), f"models/{current_file_name}.pth")

