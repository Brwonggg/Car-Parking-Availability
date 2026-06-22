import torch


def test_step(model, test_loader, loss_fn, xent_metric, device, model_path):
    model.load_state_dict(torch.load(model_path))
    model.eval()
    xent_metric.reset()
    total_correct, total_samples = 0, 0

    with torch.inference_mode():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            test_pred = model(X_batch)
            test_loss = loss_fn(test_pred, y_batch)
            xent_metric.update(test_loss)
            predicted_classes = test_pred.argmax(dim=1)
            total_correct += (predicted_classes == y_batch).sum().item()
            total_samples += y_batch.size(0)
        
    print(f"Test Loss: {xent_metric.compute().item():.4f} | Test Accuracy: {total_correct/total_samples*100:.2f}%")
    return model