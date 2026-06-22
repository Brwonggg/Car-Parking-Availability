import torch, time
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchmetrics import MeanMetric
from tqdm import tqdm
from model import Model
from data_sorter import organise_data, ParkingDataset


def train_step(train_data, device, model_path, epochs=20):
    model = Model().to(device)
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.0001, momentum=0.9, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    xent_metric = MeanMetric().to(device)

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    ])

    X_train, X_test, y_train, y_test = organise_data(train_data)
    train_dataset = ParkingDataset(X_train, y_train, transform=train_transform)
    test_dataset = ParkingDataset(X_test, y_test, transform=None)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    start_time = time.time()
    best_test_acc = 0
    best_epoch = 0
    patience = 3
    epochs_without_improvement = 0

    for epoch in tqdm(range(epochs)):
        total_correct, total_samples = 0, 0
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

        train_acc = total_correct / total_samples
        epoch_loss = xent_metric.compute()

        model.eval()
        test_correct, test_total = 0, 0
        with torch.inference_mode():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                pred = model(X_batch)
                test_correct += (pred.argmax(dim=1) == y_batch).sum().item()
                test_total += y_batch.size(0)
        test_acc = test_correct / test_total

        tqdm.write(f"Epoch {epoch+1} | Loss: {epoch_loss.item():.4f} | Train Acc: {train_acc*100:.2f} | Test Acc: {test_acc*100:.2f}")

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            torch.save(model.state_dict(), model_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}. Best was epoch {best_epoch} with {best_test_acc*100:.2f}%")
                break

    print(f"{time.time() - start_time:.1f}s | Best test accuracy: {best_test_acc*100:.2f}% at epoch {best_epoch}")
    return model, X_test, y_test, test_loader, loss_fn, xent_metric  