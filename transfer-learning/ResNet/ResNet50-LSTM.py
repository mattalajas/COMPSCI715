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

class ResNet_LSTM_Model(nn.Module):
    def __init__(self, pre_trained_model, lstm_hidden_size=512, lstm_num_layers=1, num_classes=4, num_frames=12):
        super(ResNet_LSTM_Model, self).__init__()

        # Extract frame features
        self.resnet = pre_trained_model
        num_resnet_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()  # Remove final FC layer to get feature vector

        # LSTM for sequence modeling
        self.lstm = nn.LSTM(input_size=num_resnet_features, hidden_size=lstm_hidden_size, num_layers=lstm_num_layers, batch_first=True)

        # Final fully connected layer to predict thumbstick values
        self.fc = nn.Linear(lstm_hidden_size, num_classes)

        # Store number of frames
        self.num_frames = num_frames

    def forward(self, x):
        batch_size, seq_len, c, h, w = x.size()  # x is of shape (batch_size, num_frames, channels, height, width)

        # Process each frame through ResNet
        resnet_features = []
        for t in range(seq_len):
            frame = x[:, t, :, :, :]  # Extract the t-th frame from each sequence
            features = self.resnet(frame)  # Pass each frame through Resnet
            resnet_features.append(features)

        # Stack the features along the time dimension
        resnet_features = torch.stack(resnet_features, dim=1)  # Shape: (batch_size, num_frames, num_resnet_features)

        # Initialize hidden and cell states
        batch_size = x.size(0)
        h0 = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(x.device)
        
        # Pass the sequence of features through the LSTM
        lstm_out, (hn, cn) = self.lstm(resnet_features, (h0, c0))  # Shape of lstm_out: (batch_size, num_frames, lstm_hidden_size)

        # Take the last output of the LSTM (or use mean pooling over time)
        lstm_last_output = lstm_out[:, -1, :]  # Shape: (batch_size, lstm_hidden_size)

        # Pass through final fully connected layer to get predictions
        output = self.fc(lstm_last_output)  # Shape: (batch_size, num_classes)

        return output



img_size = 224

transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((img_size, img_size))
])

#setup train and validation sets

frames = 12
cols_to_predict_value = ["thumbstick_left_x", "thumbstick_left_y", "thumbstick_right_x", "thumbstick_right_y"]

train_sessions = data_utils.DataUtils.read_txt("/data/ysun209/app/0_git/COMPSCI715/datasets/barbie_demo_dataset/train.txt")
train_set = data_utils.SingleGameDataset("Barbie", train_sessions, transform=transform, frame_count=frames, cols_to_predict=cols_to_predict_value)

val_sessions = data_utils.DataUtils.read_txt("/data/ysun209/app/0_git/COMPSCI715/datasets/barbie_demo_dataset/val.txt")
val_set = data_utils.SingleGameDataset("Barbie", val_sessions, transform=transform, frame_count=frames, cols_to_predict=cols_to_predict_value)


train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=16, pin_memory=True)
val_loader = DataLoader(val_set, batch_size=64, shuffle=True, num_workers=16, pin_memory=True)

# Set device (GPU if available, else CPU)
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")



# Initialize the model
pre_trained_model = timm.create_model('resnet50', pretrained=True)
model = ResNet_LSTM_Model(pre_trained_model=pre_trained_model, lstm_hidden_size=512, lstm_num_layers=1, num_classes=len(cols_to_predict_value), num_frames=12)
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
        images, targets = images.to(device), targets.to(device)

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
            outputs = model(images)
            loss = criterion(outputs, targets)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)

    print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_val_loss}")
    writer.add_scalar('Validation Loss', avg_val_loss, epoch)

# Save model after training
torch.save(model.state_dict(), f"models/{current_file_name}.pth")
