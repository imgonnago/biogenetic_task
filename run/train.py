import numpy as np
import torch
import os

def train(
        model, 
        dataset, 
        optimizer, 
        criterion = torch.nn.CrossEntropyLoss(), 
        num_epochs = 10,
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        save_dir = './checkpoints'
        ):
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Move model to device
    model = model.to(device)
    model.train()
    total_loss = []     
    best_loss = float('inf')
    num = 0
    learning_rate = optimizer.param_groups[0]['lr']
    for epoch in range(1, num_epochs + 1):
        print(f"Epoch {epoch} started.")
        for batch_x, batch_y in dataset:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            outputs, attn_w = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()    

            total_loss.append(loss.item())
            print(f'='*20)
            print(f"Epoch {epoch}, Batch {num}, Loss: {loss.item()}")
            print(f'='*20)

        if total_loss[-1] < best_loss:
            best_loss = total_loss[-1]
            checkpoint_path = os.path.join(save_dir, f'best_model_loss_{best_loss:.4f}_epoch{epoch}.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print(f"ㄴ----New best loss: {best_loss:.4f} at epoch {epoch}, batch {num}")
            print(f"Model saved to: {checkpoint_path}") 
                 
            num += 1
    print(f"Training started on device: {device}")
    print(f"Number of epochs: {num_epochs}, Learning rate: {learning_rate}, optimizer: {optimizer.__class__.__name__}, criterion: {criterion.__class__.__name__}")
    return print(f"Training Loss: {best_loss:.4f} at batch {total_loss.index(best_loss)}")



def Fusion_train(
        model, 
        dataset, 
        optimizer, 
        criterion = torch.nn.CrossEntropyLoss(), 
        num_epochs = 10,
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        save_dir = './checkpoints'
        ):
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Move model to device
    model = model.to(device)
    model.train()
    total_loss = []     
    best_loss = float('inf')
    num = 0
    learning_rate = optimizer.param_groups[0]['lr']
    for epoch in range(1, num_epochs + 1):
        print(f"Epoch {epoch} started.")
        for snp, gm, label in dataset:
            snp = snp.to(device)
            gm = gm.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            outputs, attn_w = model(snp, gm)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()    

            total_loss.append(loss.item())

            print(f'='*20)
            print(f"Epoch {epoch}, Batch {num}, Loss: {loss.item()}")
            print(f'='*20)
            
        if total_loss[-1] < best_loss:
            best_loss = total_loss[-1]
            checkpoint_path = os.path.join(save_dir, f'best_model_loss_{best_loss:.4f}_epoch{epoch}.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print(f"ㄴ----New best loss: {best_loss:.4f} at epoch {epoch}, batch {num}")
            print(f"Model saved to: {checkpoint_path}") 
                 
            num += 1
    
    print(f"Training started on device: {device}")
    print(f"Number of epochs: {num_epochs}, Learning rate: {learning_rate}, optimizer: {optimizer.__class__.__name__}, criterion: {criterion.__class__.__name__}")
    return print(f"Training Loss: {best_loss:.4f} at batch {total_loss.index(best_loss)}")
