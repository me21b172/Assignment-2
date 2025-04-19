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

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ConvNN(nn.Module):
    def __init__(self, numberOfFilters, filterOrganisation, activation, 
                 hidden_size, dropout, cnn_layers=5, kernel_size=3, num_classes=10):
        super().__init__()
        
        self.conv_blocks = nn.ModuleList()
        in_channels = 3
        
        for i in range(cnn_layers):
            out_channels = int(numberOfFilters * (filterOrganisation ** i))
            out_channels = max(32, min(100, out_channels))
            
            block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, padding=1),
                nn.BatchNorm2d(out_channels),
                activationFunction(activation),
                nn.MaxPool2d(2, 2)
            )
            self.conv_blocks.append(block)
            in_channels = out_channels
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, hidden_size),
            activationFunction(activation),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        for block in self.conv_blocks:
            x = block(x)
        return self.classifier(x)

def activationFunction(activation_name):
    return {
        "ReLU": nn.ReLU(),
        "GeLU": nn.GELU(),
        "SiLU": nn.SiLU(),
        "Mish": nn.Mish(),
        "LeakyReLU": nn.LeakyReLU(0.1)
    }[activation_name]

def get_transforms(data_augmentation=False):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    if data_augmentation:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            normalize
        ])
    
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])

def prepare_datasets(data_dir, batch_size, data_augmentation=True, num_workers=1):
    train_transform = get_transforms(data_augmentation=True)
    val_transform = get_transforms(data_augmentation=False)

    full_train_set = datasets.ImageFolder(root=os.path.join(data_dir, 'train'),transform=train_transform)

    train_size = int(0.8 * len(full_train_set))
    val_size = len(full_train_set) - train_size
    train_set, val_set = random_split(full_train_set, [train_size, val_size])

    val_set.dataset = copy.deepcopy(val_set.dataset)
    val_set.dataset.transform = val_transform

    test_set = datasets.ImageFolder(root=os.path.join(data_dir, 'val'),transform=val_transform)

    return (
        DataLoader(train_set, batch_size, shuffle=True, num_workers=num_workers, pin_memory=True),
        DataLoader(val_set, batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
        DataLoader(test_set, batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    )

def train_model(model, num_epochs, device, learning_rate, batch_size, dataAugmentation, iswandb=False):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # scheduler to prevent the make the learning rate adaptive
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.1, patience=2)
    # To Prevent gradient underflow/overflow
    scaler = torch.amp.GradScaler(enabled=(device.type == 'cuda'))

    train_loader, val_loader, test_loader = prepare_datasets("./inaturalist_12K", batch_size, dataAugmentation)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == 'cuda')):
                outputs = model(images)
                loss = criterion(outputs, labels)

            optimizer.zero_grad()
            #Scalar to prevent underflow/overflow of gradients
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_acc = 100 * correct / total
        train_loss = epoch_loss / len(train_loader)
        val_acc, val_loss = evaluate_accuracy(model, val_loader, device, criterion)
        test_acc, test_loss = evaluate_accuracy(model, test_loader, device, criterion)
        scheduler.step(train_acc)

        if iswandb:
            wandb.log({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_acc": train_acc,
                "val_acc": val_acc,
                "lr": optimizer.param_groups[0]['lr']
            })

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.2f}%")
        print(f"Test Loss: {test_loss:.4f} | Acc: {test_acc:.2f}%")

    # test_acc = evaluate_accuracy(model, test_loader, device)
    # print(f"\nFinal Test Accuracy: {test_acc:.2f}%")

def evaluate_accuracy(model, dataloader, device, criterion=None):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            
            if criterion:
                loss = criterion(outputs, labels)
                total_loss += loss.item()
            
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted ==labels).sum().item()

    acc = 100 * correct / total
    return (acc, total_loss/len(dataloader)) if criterion else acc

def log_predictions(model, data_dir="./inaturalist_12K"):
    # Initialize wandb
    wandb.init(project="inaturalist-classification", job_type="evaluation")
    
    # Prepare data
    _, _, test_loader = prepare_datasets(data_dir, batch_size=30, data_augmentation=False)
    dataset = ImageFolder(root=os.path.join(data_dir, 'val'))
    class_to_name = {i: name.split('/')[-1] for i, name in enumerate(dataset.classes)}
    
    # Get one batch of 30 images
    images, labels = next(iter(test_loader))
    images, labels = images.to(device), labels.to(device)
    
    with torch.no_grad():
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
    
    # Create a list of wandb.Image objects with captions
    wandb_images = []
    for idx in range(30):
        # Denormalize image
        img = images[idx].cpu().numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)
        
        # Create caption with prediction info
        actual_name = class_to_name[labels[idx].item()]
        pred_name = class_to_name[preds[idx].item()]
        correct = labels[idx] == preds[idx]
        caption = (f"✅ Correct" if correct else f"❌ Wrong") + \
                 f"\nPred: {pred_name}\nActual: {actual_name}"
        
        # Create wandb.Image with caption
        wandb_images.append(wandb.Image(
            img, 
            caption=caption
        ))
    
    # Log as a single image carousel
    wandb.log({
        "Test_Predictions": wandb_images,
        "Test_Accuracy": 100*torch.sum(labels==preds)/30
    })
    
    # Additional table logging for reference
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