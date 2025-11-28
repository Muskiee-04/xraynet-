import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

class NIHXRayDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None):
        self.dataframe = dataframe
        self.image_dir = image_dir
        self.transform = transform
        
        # NIH ChestX-ray14 classes
        self.classes = [
            'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 
            'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation',
            'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia', 'No Finding'
        ]
        
        # Convert Finding Labels to multi-hot encoding
        self.labels = []
        for idx, row in dataframe.iterrows():
            label_vector = [0] * len(self.classes)
            findings = row['Finding Labels'].split('|')
            for finding in findings:
                if finding in self.classes:
                    label_vector[self.classes.index(finding)] = 1
            self.labels.append(label_vector)
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.dataframe.iloc[idx]['Image Index'])
        
        try:
            image = Image.open(img_name).convert('RGB')
        except:
            # If image doesn't exist, create a black image
            image = Image.new('RGB', (224, 224), color='black')
        
        label = torch.FloatTensor(self.labels[idx])
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class EfficientNetModel:
    def __init__(self, num_classes=15, model_name='efficientnet_b0'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🚀 Using device: {self.device}")
        self.num_classes = num_classes
        
        # Load pre-trained EfficientNet
        if model_name == 'efficientnet_b0':
            self.model = models.efficientnet_b0(pretrained=True)
            # Replace classifier
            self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, num_classes)
        
        self.model = self.model.to(self.device)
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', patience=5, factor=0.5)
        
    def train(self, train_loader, val_loader, epochs=30):
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        
        print("🎯 Starting training...")
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
                
                if batch_idx % 100 == 0:
                    print(f'Epoch: {epoch+1}/{epochs} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                          f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
            
            train_loss /= len(train_loader)
            train_losses.append(train_loss)
            
            # Validation phase
            val_loss = self.validate(val_loader)
            val_losses.append(val_loss)
            
            self.scheduler.step(val_loss)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': best_val_loss,
                    'classes': train_loader.dataset.classes
                }, 'models/saved/best_model.pth')
                print(f'💾 New best model saved! Validation Loss: {val_loss:.4f}')
            
            print(f'Epoch {epoch+1}/{epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {self.optimizer.param_groups[0]["lr"]:.6f}')
        
        # Plot training history
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training History')
        plt.savefig('training_history.png')
        print("📊 Training history plot saved!")
        
    def validate(self, val_loader):
        self.model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                val_loss += self.criterion(output, target).item()
        
        return val_loss / len(val_loader)

def prepare_data():
    """Prepare NIH dataset for training"""
    print("📁 Loading NIH dataset...")
    
    # Load data entry
    df = pd.read_csv('data/Data_Entry_2017.csv')
    print(f"📊 Total samples: {len(df)}")
    
    # Load train/val split
    with open('data/train_val_list_NIH.txt', 'r') as f:
        train_val_files = [line.strip() for line in f]
    
    with open('data/test_list_NIH.txt', 'r') as f:
        test_files = [line.strip() for line in f]
    
    # Split into train and validation
    train_files, val_files = train_test_split(train_val_files, test_size=0.2, random_state=42)
    
    train_df = df[df['Image Index'].isin(train_files)]
    val_df = df[df['Image Index'].isin(val_files)]
    test_df = df[df['Image Index'].isin(test_files)]
    
    print(f"🎯 Training samples: {len(train_df)}")
    print(f"🔍 Validation samples: {len(val_df)}")
    print(f"🧪 Test samples: {len(test_df)}")
    
    return train_df, val_df, test_df

def main():
    # Create models directory
    os.makedirs('models/saved', exist_ok=True)
    
    # Prepare data
    train_df, val_df, test_df = prepare_data()
    
    # Data transformations
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Try different image directories
    image_dirs = ['data/images_224/', 'data/images/']
    image_dir = None
    
    for dir_path in image_dirs:
        if os.path.exists(dir_path):
            image_dir = dir_path
            print(f"📁 Using image directory: {image_dir}")
            break
    
    if not image_dir:
        raise Exception("❌ No image directory found!")
    
    # Create datasets
    train_dataset = NIHXRayDataset(train_df, image_dir, train_transform)
    val_dataset = NIHXRayDataset(val_df, image_dir, val_transform)
    
    # Create data loaders
    batch_size = 32 if torch.cuda.is_available() else 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    print(f"🔧 Batch size: {batch_size}")
    print(f"🔧 Training batches: {len(train_loader)}")
    print(f"🔧 Validation batches: {len(val_loader)}")
    
    # Initialize and train model
    model = EfficientNetModel(num_classes=15)
    model.train(train_loader, val_loader, epochs=30)

if __name__ == "__main__":
    main()