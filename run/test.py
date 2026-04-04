import numpy as np
import torch

def test(model, dataset, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    model.eval()
    total = 0
    with torch.no_grad():
        for batch_x, batch_y in dataset:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            outputs = model(batch_x)
            _, predicted = torch.max(outputs.data, 1)

            total += batch_y.size(0)
            correct = (predicted == batch_y).sum().item()
        
    accuracy = 100 * correct / total            
    return print(f"Test Accuracy: {accuracy:.2f}%")