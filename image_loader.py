import cv2 as cv
import os
import numpy as np

def load_images(train_data, augment=True):
    images = []
    labels = []
    for i, folder in enumerate(train_data):
        label = i
        for filename in os.listdir(folder):
            try:
                img = cv.imread(os.path.join(folder,filename), cv.IMREAD_GRAYSCALE)
                img = cv.resize(img, (48,48))
                images.append(img)
                labels.append(label)

                if augment:
                    # horizontal flip
                    images.append(cv.flip(img, 1))
                    labels.append(label)

                    # slight rotation
                    M = cv.getRotationMatrix2D((24, 24), 10, 1.0)
                    rotated = cv.warpAffine(img, M, (48, 48))
                    images.append(rotated)
                    labels.append(label)

                    # brightness adjustment
                    brighter = cv.convertScaleAbs(img, alpha=1.2, beta=10)
                    images.append(brighter)
                    labels.append(label)

            except Exception as e:
                print(f"Error loading image {os.path.join(folder,filename)}: {e}")
    return np.array(images), np.array(labels)