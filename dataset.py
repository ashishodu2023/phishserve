# phishserve/dataset.py
import torch
from torch.utils.data import Dataset
import re

def clean_text(text):
    # remove protocol
    text = re.sub(r'^https?://', '', text)
    # remove www
    text = re.sub(r'^www.', '', text)
    # split by special characters
    tokens = re.split(r'[/.?=_-]', text)
    # remove empty tokens
    tokens = [token for token in tokens if token]
    return tokens

class PhishingDataset(Dataset):
    def __init__(self, texts, labels, stoi, max_len=64, pad_token="<pad>", unk_token="<unk>"):
        self.texts = texts
        self.labels = labels
        self.stoi = stoi
        self.max_len = max_len
        self.pad_idx = stoi[pad_token]
        self.unk_idx = stoi[unk_token]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = self.texts[index]
        label = self.labels[index]
        tokens = clean_text(text)
        ids = [self.stoi.get(token, self.unk_idx) for token in tokens][:self.max_len]
        if len(ids) < self.max_len:
            ids += [self.pad_idx] * (self.max_len - len(ids))
        
        x = torch.tensor(ids, dtype=torch.long)
        y = torch.tensor(int(label), dtype=torch.long)
        return x, y
            
            