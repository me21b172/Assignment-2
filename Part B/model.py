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
    base_transforms = [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
    
    if data_augmentation:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            *base_transforms
        ])
    return transforms.Compose(base_transforms)

def prepare_datasets(data_dir, batch_size, data_augmentation=True):
    """Prepare data loaders with proper validation"""
    train_transform = create_transforms(data_augmentation)
    val_transform = create_transforms(False)

    # Load and split datasets
    full_train = datasets.ImageFolder(os.path.join(data_dir, 'train'), train_transform)
    train_size = int(0.8 * len(full_train))
    train_set, val_set = random_split(full_train, [train_size, len(full_train)-train_size])
    
    # Clone and transform validation set
    val_set.dataset = copy.deepcopy(val_set.dataset)
    val_set.dataset.transform = val_transform

    # Test dataset
    test_set = datasets.ImageFolder(os.path.join(data_dir, 'val'), val_transform)

    return (
        DataLoader(train_set, batch_size, shuffle=True, pin_memory=True),
        DataLoader(val_set, batch_size, shuffle=False, pin_memory=True),
        DataLoader(test_set, batch_size, shuffle=False, pin_memory=True)
    )

# === MODEL DEFINITION ===
def build_model(num_classes, device):
    """Create and configure GoogLeNet model"""
    model = models.resnet50(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    # Replace only the final classifier layer
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # Enable training only on the new final layer
    for param in model.fc.parameters():
        param.requires_grad = True

    # model = models.googlenet(pretrained=True)
    
    # # Freeze all parameters
    # for param in model.parameters():
    #     param.requires_grad = False

    # # Replace final layer with better initialization
    # model.fc = nn.Sequential(
    #     nn.Linear(model.fc.in_features, num_classes)
    # )
    
    
    return model.to(device)

# === TRAINING CORE ===
def train_model(model, num_epochs, batch_size, device, data_dir, lr=0.0001, use_wandb=False):
    """Simplified training loop with stability improvements"""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=lr, weight_decay=1e-4)
    
    # Get data loaders once instead of per epoch
    train_loader, val_loader, test_loader = prepare_datasets(data_dir, batch_size)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass with gradient clipping
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        # Calculate epoch metrics
        train_loss = total_loss / len(train_loader)
        train_acc = 100 * correct / total
        
        # Validation
        val_acc, val_loss = evaluate_accuracy(model, val_loader, device, criterion)
        test_acc, test_loss = evaluate_accuracy(model, test_loader, device, criterion)
        
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
        print(f"Test Loss: {val_loss:.4f}, Acc: {test_acc:.2f}%")
        
        if use_wandb:
            wandb.log({
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc
            })

# === EVALUATION ===
def evaluate_accuracy(model, loader, device, criterion=None):
    """Simplified evaluation function"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            
            if criterion:
                loss = criterion(outputs, labels)
                total_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    acc = 100 * correct / total
    loss = total_loss / len(loader) if criterion else 0
    return acc, loss

# === MAIN ===
def main(args):
    num_classes, data_dir = get_config_values()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = build_model(num_classes, device)
    
    if args.wandb:
        wandb.init(project="CNN-Hyperparameter-Tuning")
        wandb.watch(model)
    
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
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--batch_size", type=int, default=64)
    parser.add_argument("-e", "--epochs", type=int, default=7)
    parser.add_argument("-w", "--wandb", type=bool, default=False)
    args = parser.parse_args()
    main(args)