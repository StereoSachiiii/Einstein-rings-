import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import models, transforms
from dataset import GalaxyZooDataset
import os
import time
import numpy as np

def train_model():
    # 1. Hardware Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 2. Hyperparameters
    BATCH_SIZE = 128 # Increased for Cloud GPU (T4/P100/A100)
    LEARNING_RATE = 0.001
    EPOCHS = 15 # Increased for a deeper scientific fit
    NUM_WORKERS = 4 # High concurrency for Data Pipeline
    
    # 3. Data Preparation (Full Luxury Augmentation)
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print("Initializing FULL dataset...")
    full_dataset = GalaxyZooDataset(
        labels_path='gz2_hart16.csv.gz',
        mapping_path='gz2_filename_mapping.csv',
        img_dir=os.path.join('images_gz2', 'images'),
        transform=data_transform
    )

    # Split into Train (90%) and Validation (10%) of the FULL dataset
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    print(f"Training on FULL SET: {train_size} images")
    print(f"Validating on: {val_size} images")

    # Ultra-High-Performance DataLoader Pipeline
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=NUM_WORKERS,
        pin_memory=True,          
        prefetch_factor=2,        
        persistent_workers=True   
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True
    )

    # 4. Model Definition (Transfer Learning)
    print("Building ResNet-18 model...")
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2) 
    model = model.to(device)

    # 5. Loss, Optimizer, and Scheduler
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.1)

    # 6. Training Loop (FULL SPEED)
    print(f"\nStarting HIGH-SPEED training for {EPOCHS} epochs (Full Dataset)...")
    best_val_loss = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        start_time = time.time()
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            if (i + 1) % 50 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        epoch_loss = running_loss / len(train_loader)
        
        # 7. Validation Step
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1} Summary: Train Loss: {epoch_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Time: {time.time() - start_time:.2f}s")
        
        # Update Scheduler
        scheduler.step(avg_val_loss)

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'galaxy_model_best.pth')
            print(f"New best model saved! (Val Loss: {avg_val_loss:.4f})")

    print("\nRefined Training Finished!")

if __name__ == "__main__":
    train_model()
