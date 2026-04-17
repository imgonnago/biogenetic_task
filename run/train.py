import numpy as np
import torch
import os

def train(
        model, 
        dataset,
        val_dataset,                                        # ★ 추가: val loss 기준 저장용
        optimizer, 
        criterion = torch.nn.CrossEntropyLoss(), 
        num_epochs = 10,
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        save_dir = './checkpoints'
        ):
    
    os.makedirs(save_dir, exist_ok=True)
    model = model.to(device)
    best_val_loss = float('inf')                            # ★ train loss → val loss 기준
    best_ckpt_path = None
    learning_rate = optimizer.param_groups[0]['lr']

    for epoch in range(1, num_epochs + 1):
        # ── Train ──────────────────────────────────────────
        model.train()
        for batch_x, batch_y in dataset:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            outputs, attn_w = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

        # ── Validation (val loss 계산) ──────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_dataset:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                outputs, _ = model(batch_x)
                val_loss += criterion(outputs, batch_y).item()
        val_loss /= len(val_dataset)
        model.train()

        print(f"Epoch {epoch}/{num_epochs}  val_loss: {val_loss:.4f}")

        # ── Best 저장 (val loss 기준) ───────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_ckpt_path = os.path.join(save_dir, f'best_model_valloss_{best_val_loss:.4f}_epoch{epoch}.pth')
            torch.save(model.state_dict(), best_ckpt_path)
            print(f"  └─ New best val_loss: {best_val_loss:.4f} → {best_ckpt_path}")

    print(f"\nTraining done | device={device} | epochs={num_epochs} | lr={learning_rate}")
    print(f"Best val_loss: {best_val_loss:.4f}")
    return best_ckpt_path                                   # ★ 경로 반환


def Fusion_train(
        model, 
        dataset,
        val_dataset,                                        # ★ 추가
        optimizer, 
        criterion = torch.nn.CrossEntropyLoss(), 
        num_epochs = 10,
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        save_dir = './checkpoints'
        ):
    
    os.makedirs(save_dir, exist_ok=True)
    model = model.to(device)
    best_val_loss = float('inf')                            # ★ val loss 기준
    best_ckpt_path = None
    learning_rate = optimizer.param_groups[0]['lr']

    for epoch in range(1, num_epochs + 1):
        # ── Train ──────────────────────────────────────────
        model.train()
        for snp, gm, label in dataset:
            snp   = snp.to(device)
            gm    = gm.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            outputs, attn_w = model(snp, gm)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()

        # ── Validation ──────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for snp, gm, label in val_dataset:
                snp   = snp.to(device)
                gm    = gm.to(device)
                label = label.to(device)
                outputs, _ = model(snp, gm)
                val_loss += criterion(outputs, label).item()
        val_loss /= len(val_dataset)

        print(f"Epoch {epoch}/{num_epochs}  val_loss: {val_loss:.4f}")

        # ── Best 저장 (val loss 기준) ───────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_ckpt_path = os.path.join(save_dir, f'best_model_valloss_{best_val_loss:.4f}_epoch{epoch}.pth')
            torch.save(model.state_dict(), best_ckpt_path)
            print(f"  └─ New best val_loss: {best_val_loss:.4f} → {best_ckpt_path}")

    print(f"\nTraining done | device={device} | epochs={num_epochs} | lr={learning_rate}")
    print(f"Best val_loss: {best_val_loss:.4f}")
    return best_ckpt_path                                   # ★ 경로 반환
