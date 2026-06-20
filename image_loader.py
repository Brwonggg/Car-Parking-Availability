import cv2 as cv
import os
import numpy as np

base_path = '/Users/brandon/Downloads/archive/spots'
empty_folder = [os.path.join(base_path, 'empty')]
occupied_folder = [os.path.join(base_path, 'parked')]

train_data = empty_folder + occupied_folder

def load_images(train_data):
    images = []
    labels = []
    for i, folder in enumerate(train_data):
        label = i
        for filename in os.listdir(folder):
            try:
                img = cv.imread(os.path.join(folder, filename))
                img = cv.resize(img, (48, 48))
                images.append(img)
                labels.append(label)
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    return np.array(images), np.array(labels)