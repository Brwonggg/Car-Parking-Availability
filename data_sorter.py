from sklearn.model_selection import train_test_split
import torch
from image_loader import load_images
from torchvision import transforms
from torch.utils.data import Dataset

train_data = ['/Users/brandon/Downloads/parking/clf-data/empty',
              '/Users/brandon/Downloads/parking/clf-data/not_empty'
              ]

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
])

class ParkingDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        img = torch.FloatTensor(img).permute(2,0,1) / 255  # shape (1, 48, 48)

        if self.transform:
            img = self.transform(img)

        label = self.labels[idx]
        return img, label

def organise_data():
    images, labels = load_images(train_data)
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

    # X_train = X_train.reshape(X_train.shape[0], 1, 48, 48)
    # X_test = X_test.reshape(X_test.shape[0], 1, 48, 48)
    # X_train = torch.FloatTensor(X_train) / 255
    # X_test = torch.FloatTensor(X_test) / 255

    y_train = torch.from_numpy(y_train).long()
    y_test = torch.from_numpy(y_test).long()

    return X_train, X_test, y_train, y_test







