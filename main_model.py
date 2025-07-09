import tensorflow as tf
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.utils import img_to_array, load_img
from tqdm import tqdm
import os
data_dir = r"Flower_Classification_Model\flower_photos"

img_height = 180
img_width = 180
class_names = sorted(os.listdir(data_dir))
image_paths = []
labels = []

for label_index, class_name in enumerate(class_names):
    class_dir = os.path.join(data_dir, class_name)
    if os.path.isdir(class_dir):
        for fname in os.listdir(class_dir):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(class_dir, fname))
                labels.append(label_index)

X = []
for path in tqdm(image_paths, desc = "Loading images"):
    img = load_img(path, target_size = (img_height, img_width))
    img_array = img_to_array(img)/255.0
    X.append(img_array)

X = np.array(X)
y = np.array(labels)

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)

X_cv, X_test, y_cv, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

print(f"X train shape : {X_train.shape} y train : {y_train.shape}")
print(f"X_cv shape: {X_cv.shape}, Y_cv shape : {y_cv.shape}")
print(f"X_test shape : {X_test.shape}, y_test shape : {y_test.shape}")

def visualize_samples(X, y, title, class_names):
    plt.figure(figsize = (10, 10))
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(X[i])
        plt.title(class_names[y[i]])
        plt.axis('off')
    plt.suptitle(title)
    plt.show()

visualize_samples(X_train, y_train, "Training Set", class_names)
visualize_samples(X_cv, y_cv, "Cross-Validation Set", class_names)
visualize_samples(X_test, y_test, "Test Set", class_names)