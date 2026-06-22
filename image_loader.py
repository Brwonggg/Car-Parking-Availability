import cv2 as cv
import os
import numpy as np

def load_images(train_data):
    images = []
    labels = []
    valid_extensions = ('.png', '.jpg', '.jpeg') 

    for i, folder in enumerate(train_data):
        label = i
        for filename in os.listdir(folder):
            if not filename.lower().endswith(valid_extensions):   
                continue
            try:
                img = cv.imread(os.path.join(folder, filename))
                if img is None:
                    print(f"Skipping unreadable file: {filename}")
                    continue
                img = cv.resize(img, (48, 48))
                images.append(img)
                labels.append(label)
            except Exception as e:
                print(f"Error loading {filename}: {e}")

    return np.array(images), np.array(labels)