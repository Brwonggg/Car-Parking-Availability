import time
from tqdm import tqdm
import torch
from torch import nn
from model import Model
from torchmetrics import MeanMetric
from torch.utils.data import DataLoader
from data_sorter import organise_data, ParkingDataset
from torchvision import transforms
import glob, os

train_data = ['/Users/brandon/Downloads/matchbox_cars_parkinglot/empty',
              '/Users/brandon/Downloads/matchbox_cars_parkinglot/occupied']

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
])

epochs = 20
model = Model()
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()
xent_metric = MeanMetric()  

X_train, X_test, y_train, y_test = organise_data()
train_dataset = ParkingDataset(X_train, y_train, transform=train_transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

def train_step(X_train, y_train):
    start_time = time.time()

    for epoch in tqdm(range(epochs)):
        total_correct = 0
        total_samples = 0
        model.train()
        xent_metric.reset()
        
        for X_batch, y_batch in train_loader:
            y_pred = model(X_batch)

            loss = loss_fn(y_pred, y_batch)

            xent_metric.update(loss)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            total_correct += (y_pred.argmax(dim=1) == y_batch).sum().item()
            total_samples += y_batch.size(0)
            epoch_accuracy = total_correct / total_samples

        if (epoch + 1) % 3 == 0:
            epoch_loss = xent_metric.compute()
            tqdm.write(f"Epoch {epoch + 1} | Loss: {epoch_loss.item()} | Accuracy: {epoch_accuracy*100:.2f}")

    end_time = time.time()
    time_taken = end_time - start_time
    print(f"{time_taken:.1f}")

