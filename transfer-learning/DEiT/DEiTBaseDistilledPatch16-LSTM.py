import os
import sys
sys.path.insert(0, '/data/ysun209/app/0_python/COMPSCI715')
import utils.data_utils as data_utils
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from torch.optim import Adam
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# Define image size
img_size = 224

transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((img_size, img_size))
])

# Setup train and validation sets
cols_to_predict_value = ["thumbstick_left_x", "thumbstick_left_y", "thumbstick_right_x", "thumbstick_right_y", "head_dir_a", "head_dir_b", "head_dir_c", "head_dir_d", "head_pos_x", "head_pos_y", "head_pos_z"]

train_sessions = data_utils.DataUtils.read_txt("/data/ysun209/app/0_git/COMPSCI715/datasets/barbie_demo_dataset/train.txt")
train_set = data_utils.SingleGameDataset("Barbie", train_sessions, transform=transform, cols_to_predict=cols_to_predict_value)

val_sessions = data_utils.DataUtils.read_txt("/data/ysun209/app/0_git/COMPSCI715/datasets/barbie_demo_dataset/val.txt")
val_set = data_utils.SingleGameDataset("Barbie", val_sessions, transform=transform, cols_to_predict=cols_to_predict_value)

train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_set, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)

# Set device (GPU if available, else CPU)
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x should have shape (batch_size, seq_length, input_size)
        lstm_out, _ = self.lstm(x)  # Output of LSTM
        lstm_out = lstm_out[:, -1, :]  # Get the output of the last time step
        out = self.fc(lstm_out)  # Pass through the fully connected layer
        return out

# Define model parameters
input_size = img_size * img_size * 3  # Assuming RGB images
hidden_size = 256
num_layers = 2
output_size = len(cols_to_predict_value)

# Instantiate the LSTM model
model = LSTMModel(input_size, hidden_size, num_layers, output_size).to(device)

# Set all parameters to be trainable
for param in model.parameters():
    param.requires_grad = True

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
        images, targets = images.to(device), targets.to(device)

        # Reshape images for LSTM: (batch_size, seq_length, input_size)
        # Here we treat each image as a sequence of pixels
        seq_length = 1  # Treat each image as a single time step
        images = images.view(images.size(0), seq_length, -1)  # Reshape to (batch_size, seq_length, input_size)

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
            images, targets = images.to(device), targets.to(device)

            # Reshape images for LSTM: (batch_size, seq_length, input_size)
            images = images.view(images.size(0), seq_length, -1)  # Reshape to (batch_size, seq_length, input_size)
            outputs = model(images)
            loss = criterion(outputs, targets)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)

    print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_val_loss}")
    writer.add_scalar('Validation Loss', avg_val_loss, epoch)

# Close TensorBoard writer
writer.close()
