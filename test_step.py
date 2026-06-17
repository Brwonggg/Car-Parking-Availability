import torch
from torch import nn
from model import Model
from torchmetrics import MeanMetric
from torch.utils.data import DataLoader
from data_sorter import organise_data, ParkingDataset
from torchvision import transforms
from sklearn.metrics import confusion_matrix


train_data = ['/Users/brandon/Downloads/parking/clf-data/empty',
              '/Users/brandon/Downloads/parking/clf-data/not_empty'
              ]

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
])

epochs = 30 
model = Model()
loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.0]))
xent_metric = MeanMetric()  

X_train, X_test, y_train, y_test = organise_data()
test_dataset = ParkingDataset(X_test, y_test, transform=None)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

def test_step(X_test, y_test):
    model.eval()
    xent_metric.reset()
    total_correct = 0
    total_samples = 0
    all_preds = []
    all_labels = []

    with torch.inference_mode():
        for X_batch, y_batch in test_loader:
            test_pred = model(X_batch)

            test_loss = loss_fn(test_pred, y_batch)

            xent_metric.update(test_loss)

            predicted_classes = test_pred.argmax(dim=1)
            total_correct += (predicted_classes == y_batch).sum().item()
            total_samples += y_batch.size(0)

            all_preds.append(predicted_classes)
            all_labels.append(y_batch)

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    
    print(confusion_matrix(all_labels.numpy(), all_preds.numpy()))

    avg_loss = xent_metric.compute()
    accuracy = total_correct / total_samples

    print(f"Test Loss: {avg_loss.item():.4f} | Test Accuracy: {accuracy * 100:.2f}%")
