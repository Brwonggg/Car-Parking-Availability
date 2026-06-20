import cv2 as cv
import os
import numpy as np
import glob

train_data = ['/Users/brandon/Downloads/matchbox_cars_parkinglot/empty',
              '/Users/brandon/Downloads/matchbox_cars_parkinglot/occupied']

def load_images(train_data):
    images = []
    labels = []
    for i, folder in enumerate(train_data):
        label = i
        for filename in os.listdir(folder):
            try:
                img = cv.imread(os.path.join(folder, filename), cv.IMREAD_GRAYSCALE)
                img = cv.resize(img, (48, 48))
                images.append(img)
                labels.append(label)
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    return np.array(images), np.array(labels)