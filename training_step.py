import tqdm, time
import torch
from torch import nn
from model import Model
from data import 

epochs = 30 
model = Model()
optimizer = torch.optim.Adam(lr=0.01)
loss_fn = nn.BCELoss()


#training loop 
for epoch in range(epochs):
    model.train
    
    y_pred = model(X_train)

    loss = loss_fn(y_pred, y_train)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()