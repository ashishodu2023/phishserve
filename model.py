# phishserve/model.py
import torch
import torch.nn as nn

class PhishingClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim=128, hid=128, num_classes=2, pad_idx=0, dropout=0.2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(emb_dim, hid, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hid*2, num_classes)

    def forward(self, x):
        e = self.emb(x)              # [B,T,E]
        o, _ = self.gru(e)           # [B,T,2H]
        h = o[:, -1]                 # last time step
        h = self.dropout(h)
        return self.fc(h)            # [B,C]
