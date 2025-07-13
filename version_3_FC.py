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
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
from torch.amp import GradScaler, autocast
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="albumentations")


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
    num_workers = 2 #max parallelism os.cpu_count()(at your own risk)

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

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        persistent_workers=True, #more faster -> really important, 
        pin_memory=True #speeds up GPU transfer
    )
    cv_loader = DataLoader(
        cv_dataset, 
        batch_size=batch_size, 
        shuffle = False, 
        num_workers=num_workers,
        persistent_workers=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle = False, 
        num_workers=num_workers,
        persistent_workers=True
    )

    #Enhancing the model via transfer learning, utilizing the pretrained
    #resnet18

    model = models.resnet50(weights = ResNet50_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False #freezing pretraining
    
    #unfreezing layer 4 and the fully connected layer
    for name, param in model.named_parameters():
        if "layer4" in name or "fc" in name:
            param.requires_grad = True
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, num_classes)
    )
    model.to(device)


    #training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        filter(lambda p : p.requires_grad, model.parameters()), 
        lr = 0.0003,
        weight_decay=1e-4 #similar to punishing the model by adding a squared loss term in terms of the weight paramete
    )

    #utilizing a learning rate scheduler to adjust the learning rate of the model optimizer depending 
    #upon it's performance
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode = 'min', 
        factor = 0.3, 
        patience = 2, 
    )

    """
    Utilzing..
    Automatic Mixed Precision : Switches between FP16 and FP32 depending upon the computation increasing efficiency
    and decreasing the load
    Gradient Scaling : In order to prevent loss of information due to switch between FP32 to FP16, the 
    FP16 gradient values are multiplied by a large number to prevent loss
    """
    scaler = GradScaler("cuda")

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
            with autocast("cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        
        train_acc = correct/total
        train_loss = running_loss/len(train_loader)

        #cross validation set
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad(), autocast("cuda"):
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
    torch.save(best_model, "version_2_resnet50.pth")
    #clearing cache to prevent memory spikes
    torch.cuda.empty_cache()
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
    with torch.no_grad(), autocast("cuda"):
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
