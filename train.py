# phishserve/train.py
import os, math, argparse
from tqdm import tqdm
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from utils import load_splits
from model import PhishingClassifier
from utils import seed_everything, device, save_vocab

def train(args):

    seed_everything(42)
    dev = device()
    os.makedirs(args.out_dir, exist_ok=True)

    train_ds, val_ds, test_ds, stoi, itos = load_splits(args.csv, min_freq=2, max_len=args.max_len, balance_classes=args.balance_classes)
    
    # save vocab
    save_vocab(itos, os.path.join(args.out_dir, "itos.txt"))

    # loaders
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size*2, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size*2, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)

    # class weights (imbalance)
    import numpy as np
    labels = np.array(train_ds.labels)
    pos = (labels == 1).sum(); neg = (labels == 0).sum()
    w_pos = neg / max(pos, 1); w_neg = 1.0
    class_weights = torch.tensor([w_neg, w_pos], dtype=torch.float32, device=dev)

    model = PhishingClassifier(vocab_size=len(itos), emb_dim=args.emb_dim, hid=args.hid, num_classes=2, pad_idx=stoi["<pad>"]).to(dev)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler(enabled=torch.cuda.is_available())

    best_val = 1e9
    for epoch in range(1, args.epochs+1):
        model.train()
        tot, correct, loss_sum = 0, 0, 0.0
        for xb, yb in tqdm(train_loader, ncols=100, desc=f"train {epoch}"):
            xb, yb = xb.to(dev, non_blocking=True), yb.to(dev, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast():
                logits = model(xb)
                loss = criterion(logits, yb)
            scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
            loss_sum += loss.item() * xb.size(0)
            correct += (logits.argmax(1) == yb).sum().item()
            tot += xb.size(0)
        train_loss, train_acc = loss_sum/tot, correct/tot

        # eval
        model.eval()
        with torch.no_grad():
            tot, correct, loss_sum = 0, 0, 0.0
            for xb, yb in val_loader:
                xb, yb = xb.to(dev, non_blocking=True), yb.to(dev, non_blocking=True)
                with autocast():
                    logits = model(xb)
                    loss = criterion(logits, yb)
                loss_sum += loss.item() * xb.size(0)
                correct += (logits.argmax(1) == yb).sum().item()
                tot += xb.size(0)
            val_loss, val_acc = loss_sum/tot, correct/tot

        scheduler.step()
        print(f"epoch {epoch:02d} | train {train_loss:.4f}/{train_acc:.3f} | val {val_loss:.4f}/{val_acc:.3f}")

        # save best
        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "stoi": stoi,
                "itos": itos,
                "max_len": args.max_len,
                "emb_dim": args.emb_dim,
                "hid": args.hid
            }, os.path.join(args.out_dir, "best.pt"))

    print("Training done. Best val loss:", best_val)
    return best_val

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="data/malicious_phish.csv", help="path to csv file")
    parser.add_argument("--epochs", type=int, default=10, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=512, help="batch size")
    parser.add_argument("--max_len", type=int, default=64, help="max length of url")
    parser.add_argument("--emb_dim", type=int, default=128, help="embedding dimension")
    parser.add_argument("--hid", type=int, default=128, help="hidden dimension")
    parser.add_argument("--lr", type=float, default=2e-3, help="learning rate")
    parser.add_argument("--wd", type=float, default=1e-2, help="weight decay")
    parser.add_argument("--out_dir", type=str, default="artifacts", help="output directory")
    parser.add_argument("--balance_classes", action="store_true", help="balance classes by undersampling majority class")
    args = parser.parse_args()
    train(args)
