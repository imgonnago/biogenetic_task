import numpy as np
import torch

def validate(model, dataset):
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for batch_x, batch_y in dataset:
            batch_x = batch_x.to(torch.device)
            batch_y = batch_y.to(torch.device)

            outputs = model(batch_x)
            total += batch_y.size(0)
            _, predicted = torch.max(outputs.data, 1)   
            correct += (predicted == batch_y).sum().item()
    accuracy = 100 * correct / total
    return print(f"Validation Accuracy: {accuracy:.2f}%")