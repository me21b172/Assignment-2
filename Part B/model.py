# Import necessary libraries
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision import models
from torch.utils.data import DataLoader, random_split
import os
import copy
import ast
import argparse
import wandb

# Configuration Management
def read_config_file():
    """Read and parse configuration file"""
    config_path = "config.txt"
    with open(config_path, "r") as file:
        return ast.literal_eval(file.read())

def get_config_values():
    """Extract configuration parameters"""
    config = read_config_file()
    return int(config['num_classes']), config['data_dir']

# Data Preparation
def create_transforms(data_augmentation=False):
    """Create image transforms with basic data augmentation"""
    # Base transforms applied to all images
    base_transforms = [
        transforms.Resize((224, 224)),  # Resize to standard size
        transforms.ToTensor(),  # Convert to tensor
        # Normalize with ImageNet stats
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
    
    # Add augmentation if enabled
    if data_augmentation:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),  # Random horizontal flip
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # Color variations
            *base_transforms
        ])
    return transforms.Compose(base_transforms)

def prepare_datasets(data_dir, batch_size, data_augmentation=True):
    """Prepare data loaders with proper validation"""
    # Create different transforms for training and validation
    train_transform = create_transforms(data_augmentation)
    val_transform = create_transforms(False)

    # Load full training set and split into train/val
    full_train = datasets.ImageFolder(os.path.join(data_dir, 'train'), train_transform)
    train_size = int(0.8 * len(full_train))
    train_set, val_set = random_split(full_train, [train_size, len(full_train)-train_size])
    
    # Clone validation set and apply validation transforms
    val_set.dataset = copy.deepcopy(val_set.dataset)
    val_set.dataset.transform = val_transform

    # Load test set with validation transforms
    test_set = datasets.ImageFolder(os.path.join(data_dir, 'val'), val_transform)

    # Create data loaders with pinned memory for faster GPU transfer
    return (
        DataLoader(train_set, batch_size, shuffle=True, pin_memory=True),
        DataLoader(val_set, batch_size, shuffle=False, pin_memory=True),
        DataLoader(test_set, batch_size, shuffle=False, pin_memory=True)
    )

# === MODEL DEFINITION ===
def build_model(num_classes, device):
    """Create and configure GoogLeNet model"""
    # Load pretrained ResNet50 (note: original comment mentioned GoogLeNet but code uses ResNet)
    model = models.resnet50(pretrained=True)

    # Freeze all layers initially
    for param in model.parameters():
        param.requires_grad = False

    # Replace final fully connected layer for our task
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # Unfreeze only the final layer for training
    for param in model.fc.parameters():
        param.requires_grad = True
    
    return model.to(device)

# === TRAINING CORE ===
def train_model(model, num_epochs, batch_size, device, data_dir, lr=0.0001, use_wandb=False):
    """Simplified training loop with stability improvements"""
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=lr, weight_decay=1e-4)
    
    # Get data loaders once instead of per epoch
    train_loader, val_loader, test_loader = prepare_datasets(data_dir, batch_size)
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        # Batch processing
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass with gradient clipping
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Prevent exploding gradients
            optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        # Calculate epoch metrics
        train_loss = total_loss / len(train_loader)
        train_acc = 100 * correct / total
        
        # Validation metrics
        val_acc, val_loss = evaluate_accuracy(model, val_loader, device, criterion)
        test_acc, test_loss = evaluate_accuracy(model, test_loader, device, criterion)
        
        # Print progress
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
        
        # Log to wandb if enabled
        if use_wandb:
            wandb.log({
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc
            })
    
    # Final test metrics
    print(f"Test Loss: {val_loss:.4f}, Acc: {test_acc:.2f}%")

# === EVALUATION ===
def evaluate_accuracy(model, loader, device, criterion=None):
    """Simplified evaluation function"""
    model.eval()  # Set to evaluation mode
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():  # Disable gradient computation
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            
            # Calculate loss if criterion provided
            if criterion:
                loss = criterion(outputs, labels)
                total_loss += loss.item()
            
            # Get predictions
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    # Calculate accuracy and average loss
    acc = 100 * correct / total
    loss = total_loss / len(loader) if criterion else 0
    return acc, loss

# === MAIN ===
def main(args):
    """Main execution function"""
    # Get configuration values
    num_classes, data_dir = get_config_values()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Build and prepare model
    model = build_model(num_classes, device)
    
    # Initialize wandb if enabled
    if args.wandb:
        wandb.init(project="CNN-Hyperparameter-Tuning")
        wandb.watch(model)
    
    # Start training
    train_model(
        model=model,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        device=device,
        data_dir=data_dir,
        lr=0.001,
        use_wandb=args.wandb
    )

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--batch_size", type=int, default=64)
    parser.add_argument("-e", "--epochs", type=int, default=7)
    parser.add_argument("-w", "--wandb", type=bool, default=False)
    args = parser.parse_args()
    main(args)