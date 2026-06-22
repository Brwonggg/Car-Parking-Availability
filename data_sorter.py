from sklearn.model_selection import train_test_split
import torch
from image_loader import load_images
from torchvision import transforms
from torch.utils.data import Dataset

class ParkingDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        img = torch.FloatTensor(img).permute(2, 0, 1) / 255 

        if self.transform:
            img = self.transform(img)

        img = self.normalize(img)

        label = self.labels[idx]
        return img, label

def organise_data(train_data):
    images, labels = load_images(train_data)  
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
    y_train = torch.from_numpy(y_train).long()
    y_test = torch.from_numpy(y_test).long()
    return X_train, X_test, y_train, y_test







