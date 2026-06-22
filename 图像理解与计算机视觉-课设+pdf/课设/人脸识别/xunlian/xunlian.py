import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import math
import time

class ArcMarginProduct(nn.Module):
    """ArcFace margin product layer for metric learning"""
    def __init__(self, in_features, out_features, s=30.0, m=0.50):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input_features, labels):
        cosine = F.linear(F.normalize(input_features), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        output = torch.where(cosine > self.th, phi, cosine - self.mm)
        
        one_hot = torch.zeros(cosine.size(), device=input_features.device)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        output = (one_hot * output) + ((1.0 - one_hot) * cosine)
        output *= self.s
        
        return output

class LFWFeatureExtractor(nn.Module):
    """ResNet18-based feature extractor with dropout for regularization"""
    def __init__(self, embedding_dim=512):
        super(LFWFeatureExtractor, self).__init__()
        self.backbone = models.resnet18(pretrained=True)
        
        for param in self.backbone.parameters():
            param.requires_grad = True
        
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, embedding_dim)
        )

    def forward(self, x):
        return self.backbone(x)

class LFWDataset(Dataset):
    """LFW dataset wrapper for PyTorch DataLoader"""
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_np = self.images[idx].astype(np.uint8)
        img_pil = transforms.ToPILImage()(img_np)
        
        if self.transform:
            img_tensor = self.transform(img_pil)
        return img_tensor, self.labels[idx]

def get_dataloaders(min_faces=10, batch_size=64):
    print(f"\n[1/3] Loading LFW dataset (min_faces={min_faces})...")
    lfw = fetch_lfw_people(min_faces_per_person=min_faces, color=True, resize=1.0)
    
    images = lfw.images * 255.0
    labels = lfw.target
    num_classes = len(lfw.target_names)
    print(f"Loaded {len(images)} images, {num_classes} identities")

    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

    train_transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    train_dataset = LFWDataset(X_train, y_train, train_transform)
    test_dataset = LFWDataset(X_test, y_test, test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, test_loader, num_classes

def plot_metrics(history):
    print("\n[3/3] Generating training metrics plot...")
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    plt.plot(epochs, history['val_loss'], 'r--', label='Validation Loss')
    plt.title('Training and Validation Loss (ArcFace)', fontsize=14)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy')
    plt.plot(epochs, history['val_acc'], 'r--', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy', fontsize=14)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig('training_metrics.png', dpi=300)
    print("-> Saved to 'training_metrics.png'")
    plt.show()

def main():
    train_loader, test_loader, num_classes = get_dataloaders(min_faces=10, batch_size=64)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = LFWFeatureExtractor(embedding_dim=512).to(device)
    arcface = ArcMarginProduct(in_features=512, out_features=num_classes, s=20, m=0.35).to(device)
    
    criterion = nn.CrossEntropyLoss()
    trainable_params = [p for p in model.parameters() if p.requires_grad] + list(arcface.parameters())
    optimizer = optim.Adam(trainable_params, lr=0.0005, weight_decay=1e-4)
    
    epochs = 100
    print(f"\n[2/3] Training network for {epochs} epochs...")
    
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    for epoch in range(epochs):
        start_time = time.time()
        
        model.train()
        arcface.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            features = model(images)
            outputs = arcface(features, labels)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total
        
        model.eval()
        arcface.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                features = model(images)
                outputs = arcface(features, labels)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_loss = val_loss / len(test_loader)
        val_acc = 100. * correct / total
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        time_elapsed = time.time() - start_time
        print(f"Epoch [{epoch+1}/{epochs}] | Time: {time_elapsed:.1f}s | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")

    print("\nTraining completed! Saving feature extractor weights...")
    torch.save(model.backbone.state_dict(), 'my_lfw_resnet18_weights.pth')
    print("-> Saved to 'my_lfw_resnet18_weights.pth'")
    
    plot_metrics(history)

if __name__ == "__main__":
    main()
