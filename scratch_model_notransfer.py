import os 
import numpy as np
import torch
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import ImageFolder
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

def main():
    device = torch.device('cuda')

    data_dir = r"Flower_Classification_Model\flower_photos"
    img_height = 180
    img_width = 180
    batch_size = 32
    num_workers = 2

    #preprocessing and data augmentation
    train_transforms = transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomCrop(0.1),
        transforms.RandomAutocontrast(),
        transforms.ToTensor()
    ])

    test_transforms = transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.ToTensor()
    ])

    #loading the data set and splitting it
    full_dataset = ImageFolder(root=data_dir, transform=train_transforms)
    class_names = full_dataset.classes
    num_classes = len(class_names)

    train_size = int(0.6*len(full_dataset))
    cv_size = int(0.2*len(full_dataset))
    test_size = len(full_dataset) - train_size - cv_size

    train_dataset, cv_dataset, test_dataset = random_split(
        full_dataset, [train_size, cv_size, test_size],
        generator = torch.Generator().manual_seed(42)
    )

    cv_dataset.dataset.transform = test_transforms
    test_dataset.dataset.transform = test_transforms

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle = True, num_workers=num_workers)
    cv_loader = DataLoader(cv_dataset, batch_size = batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    #visualize samples
    def visualize_samples(dataset, title):
        plt.figure(figsize=(10, 10))

        for i in range(9):
            img, label = dataset[i]
            plt.subplot(3, 3, i + 1)
            plt.imshow(np.transpose(img.numpy(), (1, 2, 0)))
            plt.title(class_names[label])
            plt.axis('off')
        plt.suptitle(title)
        plt.show()

    visualize_samples(train_dataset, "Training Samples")

    #CNN Model

    class FlowerCNN(nn.Module):
        def __init__(self, num_classes):
            super(FlowerCNN, self).__init__()
            self.net = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding = 1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding = 1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(64, 64, 3, padding = 1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding = 1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Flatten(),
                nn.Linear(128*11*11, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, num_classes)
            )
        def forward(self, x):
            return self.net(x)
    """
    Explanation of the PyTorch Sequential Model
    Below is a detailed, line-by-line explanation of the neural network defined using nn.Sequential in PyTorch:

    1. nn.Conv2d(3, 32, 3, padding=1)
    2D Convolutional Layer

    Input channels: 3 (e.g., RGB image)

    Output channels: 32 (number of filters)

    Kernel size: 3x3

    Padding: 1 (keeps spatial dimensions unchanged)

    Purpose: Extracts low-level features from the input image.

    2. nn.ReLU()
    Activation Function

    Type: Rectified Linear Unit (ReLU)

    Purpose: Introduces non-linearity, enabling the network to learn complex patterns.

    3. nn.MaxPool2d(2)
    Max Pooling Layer

    Kernel size: 2x2

    Purpose: Reduces spatial dimensions by half, retaining the most important features and reducing computational load.

    4. nn.Conv2d(32, 64, 3, padding=1)
    2D Convolutional Layer

    Input channels: 32

    Output channels: 64

    Kernel size: 3x3

    Padding: 1

    Purpose: Learns more complex features from previous layer’s outputs.

    5. nn.ReLU()
    Activation Function

    Purpose: Adds non-linearity.

    6. nn.MaxPool2d(2)
    Max Pooling Layer

    Purpose: Further reduces spatial dimensions.

    7. nn.Conv2d(64, 64, 3, padding=1)
    2D Convolutional Layer

    Input channels: 64

    Output channels: 64

    Purpose: Deepens feature extraction.

    8. nn.ReLU()
    Activation Function

    9. nn.MaxPool2d(2)
    Max Pooling Layer

    10. nn.Conv2d(64, 128, 3, padding=1)
    2D Convolutional Layer

    Input channels: 64

    Output channels: 128

    Purpose: Extracts even higher-level features.

    11. nn.ReLU()
    Activation Function

    12. nn.MaxPool2d(2)
    Max Pooling Layer

    13. nn.Flatten()
    Flatten Layer

    Purpose: Converts the multi-dimensional output from the previous layer into a 1D vector, preparing it for the fully connected layers.

    14. nn.Linear(128*11*11, 256)
    Fully Connected (Dense) Layer

    Input features: 1281111 (assuming the input image and pooling operations result in this size)

    Output features: 256

    Purpose: Learns complex combinations of the extracted features.

    15. nn.ReLU()
    Activation Function

    16. nn.Dropout(0.5)
    Dropout Layer

    Dropout probability: 0.5

    Purpose: Randomly sets 50% of the input units to zero during training to prevent overfitting.

    17. nn.Linear(256, num_classes)
    Output Layer

    Input features: 256

    Output features: num_classes (number of possible output classes)

    Purpose: Produces the final classification scores for each class.
    """

    #fitting the model, specifying loss and optimizers
    model = FlowerCNN(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = 0.001)
    best_val_loss = float('inf')
    patience = 3
    counter = 0

    #training loop
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

        #validation set
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
        val_acc_list.append(val_acc)
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)

        print(f"Epoch {epoch + 1}: Train Acc : {train_acc:.4f}, Val Acc = {val_acc:.4f}")

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


    #plotting the accuracy and loss
    epochs_range = range(1, len(train_acc_list) + 1)
    plt.figure(figsize = (12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_acc_list, label = 'Training accuracy')
    plt.plot(epochs_range, val_acc_list, label = "Val Acc")
    plt.title("accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_loss_list, label = "Train Loss")
    plt.plot(epochs_range, val_loss_list, label = "Val Loss")
    plt.title('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

    #Evaluating on test data set
    model.eval()

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim = 1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
        
    test_acc = accuracy_score(all_labels, all_preds)
    print(f"Test accuracy: {test_acc:.4f}")
    print(classification_report(all_labels, all_preds, target_names=class_names))

if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.freeze_support()
    main()