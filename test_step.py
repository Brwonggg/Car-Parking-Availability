import tqdm, time
import torch
from torch import nn
from model import Model
from torchmetrics import MeanMetric
from torch.utils.data import DataLoader, TensorDataset
from data_sorter import organise_data
from train_step import train_step

train_data = ['/Users/brandon/Downloads/matchbox_cars_parkinglot/empty',
              '/Users/brandon/Downloads/matchbox_cars_parkinglot/occupied'
              ]

epochs = 30 
model = Model(train_data)
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.0001)
loss_fn = nn.CrossEntropyLoss()
xent_metric = MeanMetric()  

X_train, X_test, y_train, y_test = organise_data()
test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

#testing loop 
def test_step(X_test, y_test):
    model.eval()
    xent_metric.reset()
    total_correct = 0
    total_samples = 0

    with torch.inference_mode():
        for X_batch, y_batch in test_loader:
            test_pred = model(X_batch)

            test_loss = loss_fn(test_pred, y_batch)

            xent_metric.update(test_loss)

            total_correct += (test_pred.argmax(dim=1) == y_batch).sum().item()
            total_samples += y_batch.size(0)

    avg_loss = xent_metric.compute()
    accuracy = total_correct / total_samples

    print(f"Test Loss: {avg_loss.item():.4f} | Test Accuracy: {accuracy * 100:.2f}%")

train_step(X_train, y_train)
test_step(X_test, y_test)