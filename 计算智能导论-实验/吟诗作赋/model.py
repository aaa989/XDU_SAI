"""字符级LSTM语言模型。"""
import torch
import torch.nn as nn


class CharLSTM(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int = 128, hidden: int = 512, num_layers: int = 2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hidden, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden, vocab_size)

    def forward(self, x: torch.Tensor, hidden=None):
        e = self.emb(x)
        out, hidden = self.lstm(e, hidden)
        logits = self.fc(out)
        return logits, hidden
