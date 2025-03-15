import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets.RodoSolDataset import RodoSolDataset
from models import LicensePlateDetector

# Training Configs
BATCH_SIZE = 16
NUM_EPOCHS = 30
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load Dataset and DataLoader
train_dataset = RodoSolDataset(data_dir="../data/RodoSol-ALPR")
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

# Initialize Model, Loss Function, Optimizer
model = LicensePlateDetector().to(DEVICE)
criterion = nn.SmoothL1Loss()  # Huber loss for bounding boxes
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training Loop
def train():
    best_loss = float("inf")

    for epoch in range(NUM_EPOCHS):
        model.train()  # Set model to training mode
        epoch_loss = 0

        for batch_idx, (images, targets) in enumerate(train_loader):
            images, targets = images.to(DEVICE), targets.to(DEVICE)  # Move to GPU if available

            # Forward Pass
            predictions = model(images)  # Shape: (batch, 4)
            loss = criterion(predictions, targets)  # Compute loss

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # Print Progress
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Step [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}")

        # Calculate Average Loss for Epoch
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Completed - Avg Loss: {avg_loss:.4f}")

        #Save Best Model 
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "best_model.pth")
            print("Saved model.")

if __name__ == "__main__":
    train()