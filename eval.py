# phishserve/eval.py
import os, json, argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, average_precision_score, roc_auc_score
from dataset import PhishingDataset
from model import PhishingClassifier
from utils import device, load_vocab
import pandas as pd

def evaluate(args):
    dev = device()
    df = pd.read_csv(args.csv).dropna(subset=["text","label"])
    # load artifacts
    ck = torch.load(args.ckpt, map_location=dev)
    
    itos = load_vocab(os.path.join(os.path.dirname(args.ckpt), "itos.txt"))
    stoi = {s:i for i,s in enumerate(itos)}

    max_len = ck["max_len"]; emb_dim=ck["emb_dim"]; hid=ck["hid"]

    ds = PhishingDataset(df["text"].tolist(), df["label"].tolist(), stoi, max_len=max_len)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = PhishingClassifier(len(itos), emb_dim=emb_dim, hid=hid, num_classes=2, pad_idx=stoi["<pad>"]).to(dev)
    model.load_state_dict(ck["model"]); model.eval()

    y_true, y_prob = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(dev, non_blocking=True)
            logits = model(xb)
            prob = torch.softmax(logits, dim=1)[:,1].cpu().numpy()
            y_prob.append(prob)
            y_true.append(yb.numpy())
    y_true = np.concatenate(y_true); y_prob = np.concatenate(y_prob)
    y_pred = (y_prob >= 0.5).astype(int)

    print(classification_report(y_true, y_pred, digits=4))
    print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))
    print("PR-AUC:", average_precision_score(y_true, y_prob))
    print("ROC-AUC:", roc_auc_score(y_true, y_prob))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="data/malicious_phish.csv", help="path to csv file")
    parser.add_argument("--ckpt", type=str, default="artifacts/best.pt", help="path to model checkpoint")
    parser.add_argument("--batch_size", type=int, default=1024, help="batch size")
    args = parser.parse_args()
    evaluate(args)
