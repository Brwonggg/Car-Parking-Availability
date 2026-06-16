from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import torch
from torch import nn
import time
from tqdm import tqdm
from torchmetrics import MeanMetric
from model import Model
from image_loader import load_images
# from train_step import train_step
from torch.utils.data import DataLoader, TensorDataset

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

train_dataset = TensorDataset(X_train, y_train_long)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

epochs = 30 
model = Model(train_data)
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()
xent_metric = MeanMetric()  

#training loop 
def train_step(X_train, y_train):
    start_time = time.time()
    accuracy = 0

    for epoch in tqdm(range(epochs)):
        model.train()
        xent_metric.reset()
        
        for X_batch, y_batch in train_loader:
            y_pred = model(X_batch)

            loss = loss_fn(y_pred, y_batch)

            xent_metric.update(loss)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

        if (epoch + 1) % 3 == 0:
            epoch_loss = xent_metric.compute()
            tqdm.write(f"Epoch {epoch + 1} | Average Epoch Cross Entropy: {epoch_loss.item()}")

    end_time = time.time()
    time_taken = end_time - start_time
    print(time_taken)
    
train_step(X_train, y_train_long)




