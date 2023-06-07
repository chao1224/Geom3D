from collections import *
from tkinter import X
from torch import nn


class CNN(nn.Module):
    def __init__(
        self, vocab_size, pad_size, hidden_size, num_tasks,
        out_channels=16, kernel_size=8, dropout=0.3,
    ):
        super(CNN, self).__init__()
        self.vocab_size = vocab_size
        self.pad_size = pad_size
        self.hidden_size = hidden_size

        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dropout = dropout

        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.layer1 = nn.Sequential(
            nn.Conv1d(self.pad_size, self.out_channels, kernel_size=self.kernel_size),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )

        self.intermediate_dim = self.hidden_size - self.kernel_size + 1
        self.layer2 = nn.Sequential(
            nn.Linear(self.intermediate_dim * self.out_channels, hidden_size),
            nn.ReLU()
        )

        self.pred_layer = nn.Sequential(
            nn.Linear(hidden_size, num_tasks)
        )
        print(self)
        return

    def forward(self, x):
        x = self.embedding(x)
        repr = self.layer1(x)
        repr = repr.view(-1, self.out_channels * self.intermediate_dim)
        repr = self.layer2(repr)
        out = self.pred_layer(repr)
        return out