import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader,random_split
import os
import wandb
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("Using device:", device)

class ConvNN(nn.Module):
    def __init__(self, numberOfFilters,filterOrganisation, activation,hidden_size, dropout, cnn_layers = 5 ,kernel_size = 3 ,num_classes=10):
        super().__init__()
        self.conv = nn.ModuleList()
        in_channel = 3
        for i in range(cnn_layers):
            self.conv.append(nn.Conv2d(in_channels = in_channel, out_channels=numberOfFilters, kernel_size=kernel_size, stride=1))
            self.conv.append(nn.BatchNorm2d(numberOfFilters))
            self.conv.append(activationFunction(activation))
            self.conv.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channel = numberOfFilters
            numberOfFilters = min(100,max(32, int(numberOfFilters * filterOrganisation)))
        self.conv.append(nn.AdaptiveAvgPool2d(1))
        self.conv.append(nn.Flatten())
        # Look into it
        # self.conv.append(activationFunction(activation))
        self.conv.append(nn.Linear(in_channel, hidden_size))
        self.conv.append(activationFunction(activation))
        self.conv.append(nn.Dropout(dropout))
        self.conv.append(nn.Linear(hidden_size, num_classes))

    def forward(self, x):
        for i in range(len(self.conv)):
            x = self.conv[i](x)
        return x

def activationFunction(x):
    activation = {"ReLU": nn.ReLU(), "GeLU": nn.GELU(), "SiLU": nn.SiLU(), "Mish": nn.Mish(), "LeakyReLU": nn.LeakyReLU()}
    return activation[x]

def evaluate_accuracy(model, dataloader, device):
    correct = 0
    total = 0
    total_loss = 0
    criterion = nn.CrossEntropyLoss()
    for images, labels in dataloader:
        with torch.no_grad():
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = 100 * correct / total
    # print(f"Validation Accuracy: {acc:.2f}%")
    return acc, total_loss/len(dataloader)

def train_model(model,num_epochs,device,learning_rate, batch_size, dataAugmentation,iswandb=False):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max')
    prev_Acc = 0
    for epoch in range(num_epochs):
        total_loss = 0
        correct = 0
        total = 0
        train_loader,val_loader,test_loader = data_loader(batch_size=batch_size,data_dir = "./inaturalist_12K",dataAugmentation=dataAugmentation)
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            with torch.autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(images)
                loss = criterion(outputs, labels)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_acc = (correct/total)*100
            total_loss += loss.item()
            del loss, outputs

        train_loss = total_loss / len(train_loader)
        val_acc,val_loss = evaluate_accuracy(model, val_loader, device)
        test_acc,test_loss = evaluate_accuracy(model, test_loader, device)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}, Val_Acc: {val_acc}, Test_acc : {test_acc}")
        if iswandb:
            wandb.log({"train_loss": train_loss,"val_loss": val_loss, "val_accuracy": val_acc, "train_accuracy": train_acc})
        if train_acc - prev_Acc < 0.5:
            learning_rate = learning_rate * 0.1
            print(f"Reducing learning rate to {learning_rate}")
        prev_Acc = train_acc

def transform_image(dataAugmentation=False):
    if dataAugmentation:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        return transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    return transform

# def data_loader(data_dir, batch_size,dataAugmentation, num_workers=3):
#     # transform = transform_image(dataAugmentation=True)
#     train_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'train'), transform=transform_image(dataAugmentation=dataAugmentation))
#     train_dataset, val_dataset = random_split(train_dataset, [0.8, 0.2])
#     val_dataset.dataset.transform = transform_image() # No data augmentation for validation set
#     test_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'val'), transform=transform_image())

#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
#     val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
#     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

#     return train_loader, val_loader, test_loader

def data_loader(data_dir, batch_size, dataAugmentation, num_workers=1):
    full_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, 'train'),
        transform=transform_image(dataAugmentation=dataAugmentation)
    )
    
    # Extract targets (class labels) for stratification
    targets = [s[1] for s in full_dataset.samples]

    # Perform stratified split
    train_idx, val_idx = train_test_split(
        range(len(targets)),
        test_size=0.2,
        stratify=targets,
        random_state=42
    )

    # Create Subsets
    train_dataset = Subset(full_dataset, train_idx)
    val_dataset = Subset(full_dataset, val_idx)

    # Overwrite the transform for validation set (no data augmentation!)
    val_dataset.dataset.transform = transform_image()  

    # Test set
    test_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, 'val'),
        transform=transform_image()
    )

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader

