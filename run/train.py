import numpy as np
import torch
def train(model, dataset, optimizer, criterion = torch.nn.CrossEntropyLoss(), learning_rate = 1e-4):
    model.train()
    total_loss = 0
    num = 0
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for batch_x, batch_y in dataset:
        batch_x = batch_x.to(torch.device)
        batch_y = batch_y.to(torch.device)

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num += 1
        print(f'='*20)
        print(f"Batch {num}, Loss: {loss.item()}")
        print(f'='*20)

    return print(f"Training Loss: {total_loss / len(x)}")

