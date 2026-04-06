import numpy as np
import torch
from sklearn.metrics import confusion_matrix, classification_report

def validate(
        model, 
        dataset,
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ):
    
    model.eval()
    total = 0
    correct = 0
    all_preds  = []    
    all_labels = []
    with torch.no_grad():
        for batch_x, batch_y in dataset:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            outputs,attn_w = model(batch_x)
            total += batch_y.size(0)
            _, predicted = torch.max(outputs.data, 1)   
            correct += (predicted == batch_y).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

    accuracy = 100 * correct / total
    confusion = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds)

    return print(f"Validation Accuracy: {accuracy:.2f}% \nConfusion Matrix: \n{confusion}\nClassification Report: \n{report}")

def Fusion_validate(
        model, 
        dataset,
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ):
    
    model.eval()
    total = 0
    correct = 0
    all_preds  = []    
    all_labels = []
    with torch.no_grad():
        for snp, gm, label in dataset:
            snp = snp.to(device)
            gm = gm.to(device)
            label = label.to(device)

            outputs,attn_w = model(snp, gm)
            total += label.size(0)
            _, predicted = torch.max(outputs.data, 1)   
            correct += (predicted == label).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(label.cpu().numpy())

    accuracy = 100 * correct / total
    confusion = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds)
    return print(f"Validation Accuracy: {accuracy:.2f}% \nConfusion Matrix: \n{confusion}\nClassification Report: \n{report}")