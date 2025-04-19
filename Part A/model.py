# Import necessary libraries
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os
import copy
import wandb
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
import ast

# Device configuration - use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Custom CNN model class
class ConvNN(nn.Module):
    def __init__(self, numberOfFilters, filterOrganisation, activation, 
                 hidden_size, dropout, cnn_layers=5, kernel_size=3, num_classes=10):
        super().__init__()
        
        # Create a list of convolutional blocks
        self.conv_blocks = nn.ModuleList()
        in_channels = 3  # Starting with 3 channels for RGB images
        
        # Build each convolutional block
        for i in range(cnn_layers):
            # Calculate output channels with organizational pattern
            out_channels = int(numberOfFilters * (filterOrganisation ** i))
            # Ensure channels stay within reasonable bounds
            out_channels = max(32, min(100, out_channels))
            
            # Each block contains Conv2d, BatchNorm, Activation, and MaxPool
            block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, padding=1),
                nn.BatchNorm2d(out_channels),
                activationFunction(activation),
                nn.MaxPool2d(2, 2)  # Downsample by factor of 2
            )
            self.conv_blocks.append(block)
            in_channels = out_channels  # Update input channels for next block
        
        # Classifier head with adaptive pooling and fully connected layers
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Global average pooling
            nn.Flatten(),
            nn.Linear(in_channels, hidden_size),
            activationFunction(activation),
            nn.Dropout(dropout),  # Regularization
            nn.Linear(hidden_size, num_classes)  # Final classification layer
        )

    def forward(self, x):
        # Pass input through all convolutional blocks
        for block in self.conv_blocks:
            x = block(x)
        # Final classification
        return self.classifier(x)

# Helper function to get activation functions
def activationFunction(activation_name):
    # Dictionary mapping activation names to PyTorch implementations
    return {
        "ReLU": nn.ReLU(),
        "GeLU": nn.GELU(),
        "SiLU": nn.SiLU(),
        "Mish": nn.Mish(),
        "LeakyReLU": nn.LeakyReLU(0.1)
    }[activation_name]

# Function to create image transforms with/without augmentation
def get_transforms(data_augmentation=False):
    # Standard ImageNet normalization
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    if data_augmentation:
        # Training transforms with augmentation
        return transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),  # Random crop
            transforms.RandomHorizontalFlip(),  # Random flip
            transforms.RandomRotation(15),  # Small rotation
            transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Color variation
            transforms.ToTensor(),
            normalize
        ])
    
    # Validation/test transforms without augmentation
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),  # Center crop
        transforms.ToTensor(),
        normalize
    ])

# Function to prepare datasets and data loaders
def prepare_datasets(data_dir, batch_size, data_augmentation=True, num_workers=1):
    # Get transforms for training and validation
    train_transform = get_transforms(data_augmentation=True)
    val_transform = get_transforms(data_augmentation=False)

    # Load full training set
    full_train_set = datasets.ImageFolder(root=os.path.join(data_dir, 'train'),transform=train_transform)

    # Split into training and validation sets (80/20)
    train_size = int(0.8 * len(full_train_set))
    val_size = len(full_train_set) - train_size
    train_set, val_set = random_split(full_train_set, [train_size, val_size])

    # Create separate validation set with different transforms
    val_set.dataset = copy.deepcopy(val_set.dataset)
    val_set.dataset.transform = val_transform

    # Load test set with validation transforms
    test_set = datasets.ImageFolder(root=os.path.join(data_dir, 'val'),transform=val_transform)

    # Create data loaders with specified batch size and workers
    return (
        DataLoader(train_set, batch_size, shuffle=True, num_workers=num_workers, pin_memory=True),
        DataLoader(val_set, batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
        DataLoader(test_set, batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    )

# Main training function
def train_model(model, num_epochs, device, learning_rate, batch_size, dataAugmentation, iswandb=False):
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # Learning rate scheduler that reduces on plateau
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.1, patience=2)
    # Gradient scaler for mixed precision training
    scaler = torch.amp.GradScaler(enabled=(device.type == 'cuda'))

    # Get data loaders
    train_loader, val_loader, test_loader = prepare_datasets("./inaturalist_12K", batch_size, dataAugmentation)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0

        # Process each batch
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Mixed precision training for efficiency
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == 'cuda')):
                outputs = model(images)
                loss = criterion(outputs, labels)

            # Backpropagation with gradient scaling
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Track metrics
            epoch_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        # Calculate epoch metrics
        train_acc = 100 * correct / total
        train_loss = epoch_loss / len(train_loader)
        # Evaluate on validation set
        val_acc, val_loss = evaluate_accuracy(model, val_loader, device, criterion)
        # Adjust learning rate based on performance
        scheduler.step(train_acc)

        # Log to wandb if enabled
        if iswandb:
            wandb.log({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_acc": train_acc,
                "val_acc": val_acc,
                "lr": optimizer.param_groups[0]['lr']  # Log learning rate
            })

        # Print progress
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.2f}%")

# Evaluation function
def evaluate_accuracy(model, dataloader, device, criterion=None):
    model.eval()  # Set to evaluation mode
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # Disable gradient computation
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            
            # Calculate loss if criterion provided
            if criterion:
                loss = criterion(outputs, labels)
                total_loss += loss.item()
            
            # Get predictions
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted ==labels).sum().item()

    # Return accuracy and average loss
    acc = 100 * correct / total
    return (acc, total_loss/len(dataloader)) if criterion else acc

# Function to log predictions to wandb
def log_predictions(model, data_dir="./inaturalist_12K"):
    # Initialize wandb run
    wandb.init(project="inaturalist-classification", job_type="evaluation")
    
    # Prepare data loaders
    _, _, test_loader = prepare_datasets(data_dir, batch_size=30, data_augmentation=False)
    dataset = ImageFolder(root=os.path.join(data_dir, 'val'))
    # Create mapping from class indices to names
    class_to_name = {i: name.split('/')[-1] for i, name in enumerate(dataset.classes)}
    
    # Get one batch of 30 images
    images, labels = next(iter(test_loader))
    images, labels = images.to(device), labels.to(device)
    
    # Get model predictions
    with torch.no_grad():
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
    
    # Prepare images for visualization
    wandb_images = []
    for idx in range(30):
        # Denormalize image for display
        img = images[idx].cpu().numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)  # Ensure valid pixel values
        
        # Create caption showing prediction vs actual
        actual_name = class_to_name[labels[idx].item()]
        pred_name = class_to_name[preds[idx].item()]
        correct = labels[idx] == preds[idx]
        caption = (f"✅ Correct" if correct else f"❌ Wrong") + \
                 f"\nPred: {pred_name}\nActual: {actual_name}"
        
        # Create wandb image with caption
        wandb_images.append(wandb.Image(
            img, 
            caption=caption
        ))
    
    # Log images as a carousel
    wandb.log({
        "Test_Predictions": wandb_images,
        "Test_Accuracy": 100*torch.sum(labels==preds)/30  # Batch accuracy
    })
    
    # Additional detailed logging as table
    wandb.log({
        "Detailed_Predictions": wandb.Table(
            columns=["Image", "Prediction", "Label", "Correct"],
            data=[
                [wandb_images[i], 
                 class_to_name[preds[i].item()], 
                 class_to_name[labels[i].item()], 
                 bool(preds[i] == labels[i])]
                for i in range(30)
            ]
        )
    })