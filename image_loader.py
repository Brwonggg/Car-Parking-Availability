import os
from sklearn.model_selection import train_test_split
import numpy as np
import cv2 as cv
from torch import nn
import torch.nn.functional as F
import torch
from model import Model
import tqdm
import time

train_data = ['/Users/brandon/Downloads/matchbox_cars_parkinglot/empty',
              '/Users/brandon/Downloads/matchbox_cars_parkinglot/occupied'
              ]

def load_images(train_data):
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
            except Exception as e:
                print(f"Error loading image {os.path.join(folder,filename)}: {e}")
    return np.array(images), np.array(labels)

images, labels = load_images(train_data)
X_train, y_train, X_test, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

X_train = X_train.reshape(X_train.shape[0],(48, 48), 1).astype('float32') / 255
X_test = X_test.reshape(X_test.shape[0],(48, 48), 1).astype('float32') / 255

y_train_long = torch.from_numpy(y_train).long()
y_test_long = torch.from_numpy(y_test).long()
y_train = F.one_hot(y_train_long)
y_test = F.one_hot(y_test_long)

print(X_train.dtype)

epochs = 30 
model = Model()
optimizer = torch.optim.Adam(lr=0.01)
loss_fn = nn.BCELoss()

#training loop 
for epoch in range(epochs):
    model.train
    
    y_pred = model(X_train)

    loss = loss_fn(y_pred, y_train)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

#testing loop 
model.eval()
with torch.inference_mode:
    test_pred = model(X_test)

    test_loss = loss_fn(y_pred, y_test)
