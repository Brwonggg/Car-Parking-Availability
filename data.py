from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import torch
from model import Model
from image_loader import load_images

train_data = ['/Users/brandon/Downloads/matchbox_cars_parkinglot/empty',
              '/Users/brandon/Downloads/matchbox_cars_parkinglot/occupied'
              ]

images, labels = load_images(train_data)
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

X_train = X_train.reshape(X_train.shape[0], 1, 48, 48)
X_test = X_test.reshape(X_test.shape[0], 1, 48, 48)
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)

y_train_long = torch.from_numpy(y_train).long()
y_test_long = torch.from_numpy(y_test).long()
y_train = F.one_hot(y_train_long)
y_test = F.one_hot(y_test_long)






