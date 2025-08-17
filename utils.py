# phishserve/utils.py
import os 
import numpy as np 
import random
import torch 
import pandas as pd
from collections import Counter
from dataset import PhishingDataset, clean_text
from sklearn.model_selection import train_test_split

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    

def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_vocab(texts, min_freq=2,specials=('<pad>','<unk>')):
    counter = Counter()
    for text in texts:
        counter.update(clean_text(text))
        
    itos = list(specials)
    for k, v in counter.items():
        if v >= min_freq:
            itos.append(k)
    stoi = {t:i for i, t in enumerate(itos)}
    return stoi, itos

def load_splits(csv_path: str, test_size=0.2, val_size=0.1, min_freq=2, max_len=64):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["text", "label"])
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(int)

    # stratified split
    train_df, test_df = train_test_split(df, test_size=test_size, stratify=df["label"], random_state=42)
    train_df, val_df  = train_test_split(train_df, test_size=val_size, stratify=train_df["label"], random_state=42)

    stoi, itos = build_vocab(train_df["text"].tolist(), min_freq=min_freq)

    # ensure specials present
    for sp in ('<pad>', '<unk>'):
        if sp not in stoi:
            idx = len(stoi)
            stoi[sp] = idx
            itos.append(sp)

    train_ds = PhishingDataset(train_df["text"].tolist(), train_df["label"].tolist(), stoi, max_len=max_len)
    val_ds   = PhishingDataset(val_df["text"].tolist(),   val_df["label"].tolist(),   stoi, max_len=max_len)
    test_ds  = PhishingDataset(test_df["text"].tolist(),  test_df["label"].tolist(),  stoi, max_len=max_len)
    return train_ds, val_ds, test_ds, stoi, itos

def save_vocab(itos, path):
    with open(path, 'w') as f:
        for s in itos:
            f.write(f"{s}\n")

def load_vocab(path):
    with open(path, 'r') as f:
        itos = [line.strip() for line in f]
    return itos
