import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import models
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image

#custom dataset wrapper implemented with albumentations (data augmentation but faster :))
class AlbumentationsDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform = None):
        self.dataset = dataset
        self.transform = transform
    
    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        img = np.array(img)

        if self.transform:
            img = self.transform(image = img)['image']
        
        return img, label

    def __len__(self):
        return len(self.dataset)
    
def main():
    device = torch.device('cuda')
    data_dir = r"Flower_Classification_Model\flower_photos"
    img_size = 224
    batch_size = 32
    num_workers = os.cpu_count()//2 #max parallelism (at your own risk)

    #Albumentations 
    train_transform = A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p = 0.5),
        A.RandomBrightnessContrast(p = 0.2),
        A.Rotate(limit = 15),
        A.ShiftScaleRotate(p = 0.2),
        A.Normalize(),
        ToTensorV2()
    ])


    test_transform = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(),
        ToTensorV2()
    ])

    #load with PIL and wrap in albumentationsdataset class
    base_dataset = ImageFolder(root = data_dir)
    class_names = base_dataset.classes
    num_classes = len(class_names)


    #Split into training, cross val, testing
    train_len = int(0.6*len(base_dataset))
    cv_len = int(0.2*len(base_dataset))
    test_len = len(base_dataset) - train_len - cv_len
    

    train_data, cv_data, test_data = random_split(base_dataset, 
                                                  [train_len, cv_len, test_len], 
                                                  generator = torch.Generator().manual_seed(42)
                                                  )
    
    train_dataset = AlbumentationsDataset(train_data, train_transform)
    cv_dataset = AlbumentationsDataset(cv_data, test_transform)
    test_dataset = AlbumentationsDataset(test_data, test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    cv_loader = DataLoader(cv_dataset, batch_size=batch_size, shuffle = False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle = False, num_workers=num_workers)

    #Enhancing the model via transfer learning, utilizing the pretrained
    #resnet18

    model = models.resnet18(pretrained = True)
    for param in model.parameters():
        param.required_grad = False #freezing pretraining
    
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, num_classes)
    )
    model.to(device)


    #training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr = 0.001)
    best_val_loss = float('inf')
    patience = 3
    counter = 0

    train_acc_list, val_acc_list, train_loss_list, val_loss_list = [], [], [], []

    for epoch in range(50):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for images, labels in tqdm(train_loader, desc = f"Epoch {epoch + 1}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        
        train_acc = correct/total
        train_loss = running_loss/len(train_loader)

        #cross validation set
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in cv_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
        val_acc = correct/total
        val_loss /= len(cv_loader)

        train_acc_list.append(train_acc)
        train_loss_list.append(train_loss)
        val_acc_list.append(val_acc)
        val_loss_list.append(val_loss)

        print(f"Epoch {epoch + 1}: Train Acc {train_acc:.4f} Val acc: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict()
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered")
                break
    model.load_state_dict(best_model)

    #plotting and evaluation
    epochs_range = range(1, len(train_acc_list) + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_acc_list, label='Train Acc')
    plt.plot(epochs_range, val_acc_list, label='Val Acc')
    plt.title('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_loss_list, label='Train Loss')
    plt.plot(epochs_range, val_loss_list, label='Val Loss')
    plt.title('Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    test_acc = accuracy_score(all_labels, all_preds)
    print(f"Test Accuracy: {test_acc:.4f}")
    print(classification_report(all_labels, all_preds, target_names=class_names))
if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.freeze_support()
    main()
