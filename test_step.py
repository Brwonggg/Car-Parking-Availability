import tqdm, time
import torch
from torch import nn
from model import Model

epochs = 30 
model = Model()
optimizer = torch.optim.Adam(lr=0.01)
loss_fn = nn.BCELoss()

#testing loop 
def test_step(X_test, y_test):
    model.eval()
    with torch.inference_mode:
        test_pred = model(X_test)

        test_loss = loss_fn(test_pred, y_test)