from sklearn.model_selection import train_test_split
import torch
from image_loader import load_images
from torchvision import transforms
from torch.utils.data import Dataset

train_data = ['/Users/brandon/Downloads/matchbox_cars_parkinglot/empty',
              '/Users/brandon/Downloads/matchbox_cars_parkinglot/occupied']

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
        img = torch.FloatTensor(img).unsqueeze(0) / 255 

        if self.transform:
            img = self.transform(img)

        label = self.labels[idx]
        return img, label

def organise_data():
    images, labels = load_images(train_data)
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
    y_train = torch.from_numpy(y_train).long()
    y_test = torch.from_numpy(y_test).long()
    return X_train, X_test, y_train, y_test







