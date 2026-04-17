import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import torch

def test(
        model, 
        dataset,
        ckpt_path,                                          # ★ 추가: 베스트 체크포인트 경로
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ):
    
    # ★ 베스트 파라미터 로드
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)

    model.eval()
    total = 0
    correct = 0
    all_preds  = []    
    all_labels = []
    with torch.no_grad():
        for batch_x, batch_y in dataset:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            outputs, attn_w = model(batch_x)
            _, predicted = torch.max(outputs.data, 1)

            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
        
    accuracy = 100 * correct / total
    confusion = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds)
    return print(f"Test Accuracy: {accuracy:.2f}% \nConfusion Matrix:\n{confusion}\nClassification Report: \n{report}")


def Fusion_test(
        model, 
        dataset,
        ckpt_path,                                          # ★ 추가
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ):
    
    # ★ 베스트 파라미터 로드
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)

    model.eval()
    total = 0
    correct = 0
    all_preds  = []    
    all_labels = []
    with torch.no_grad():
        for snp, gm, label in dataset:
            snp   = snp.to(device)
            gm    = gm.to(device)
            label = label.to(device)

            outputs, attn_w = model(snp, gm)
            _, predicted = torch.max(outputs.data, 1)

            total += label.size(0)
            correct += (predicted == label).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(label.cpu().numpy())

    accuracy = 100 * correct / total 
    confusion = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds)           
    return print(f"Test Accuracy: {accuracy:.2f}% \nConfusion Matrix: \n{confusion}\nClassification Report: \n{report}")
