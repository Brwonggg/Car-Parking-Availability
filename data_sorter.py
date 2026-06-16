from sklearn.model_selection import train_test_split
import torch
from image_loader import load_images

train_data = ['/Users/brandon/Downloads/matchbox_cars_parkinglot/empty',
              '/Users/brandon/Downloads/matchbox_cars_parkinglot/occupied'
              ]

def organise_data():
    images, labels = load_images(train_data)
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

    X_train = X_train.reshape(X_train.shape[0], 1, 48, 48)
    X_test = X_test.reshape(X_test.shape[0], 1, 48, 48)
    X_train = torch.FloatTensor(X_train) / 255
    X_test = torch.FloatTensor(X_test) / 255

    y_train = torch.from_numpy(y_train).long()
    y_test = torch.from_numpy(y_test).long()

    return X_train, X_test, y_train, y_test






