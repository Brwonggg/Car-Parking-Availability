import time
from tqdm import tqdm
import torch
from model import Model
from torchmetrics import MeanMetric
from torch.utils.data import DataLoader
from data_sorter import organise_data, ParkingDataset
from torchvision import transforms
import os
from torch import nn

base_path = '/Users/brandon/Downloads/archive/spots'
empty_folder = [os.path.join(base_path, 'empty')]
occupied_folder = [os.path.join(base_path, 'parked')]

train_data = empty_folder + occupied_folder

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
])

epochs = 20
model = Model()
device = torch.device("cpu")
model = model.to(device)
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.0001, momentum=0.9, weight_decay=1e-4)
loss_fn = nn.CrossEntropyLoss()
xent_metric = MeanMetric()  
xent_metric = xent_metric.to(device)

X_train, X_test, y_train, y_test = organise_data()
train_dataset = ParkingDataset(X_train, y_train, transform=train_transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

def train_step(X_train, y_train, loss_fn):
    start_time = time.time()

    for epoch in tqdm(range(epochs)):
        total_correct = 0
        total_samples = 0
        best_test_acc = 0
        best_epoch = 0
        patience = 3
        epochs_without_improvement = 0
        model.train()
        xent_metric.reset()
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            y_pred = model(X_batch)

            loss = loss_fn(y_pred, y_batch)

            xent_metric.update(loss)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            total_correct += (y_pred.argmax(dim=1) == y_batch).sum().item()
            total_samples += y_batch.size(0)
            epoch_accuracy = total_correct / total_samples
            train_acc = total_correct / total_samples

        model.eval()
        test_correct = 0
        test_total = 0
        with torch.inference_mode():
            test_dataset = ParkingDataset(X_test, y_test, transform=None)

            test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                pred = model(X_batch)
                test_correct += (pred.argmax(dim=1) == y_batch).sum().item()
                test_total += y_batch.size(0)
        test_acc = test_correct / test_total
        epoch_loss = xent_metric.compute()
        tqdm.write(f"Epoch {epoch+1} | Loss: {epoch_loss.item():.4f} | Train Acc: {train_acc*100:.2f} | Test Acc: {test_acc*100:.2f}")

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch+1}. Best was epoch {best_epoch} with {best_test_acc*100:.2f}% test accuracy")
                break

    print(f"Best test accuracy: {best_test_acc*100:.2f}% at epoch {best_epoch}")

    end_time = time.time()
    time_taken = end_time - start_time
    print(f"{time_taken:.1f}")

train_step(X_train, y_train, loss_fn)

