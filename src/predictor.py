"""
Prediction models for ANOS.

Provides three model families:
    - NetLSTM        : LSTM-based predictor with temporal memory.
    - NetMLP         : Multi-layer perceptron (no temporal memory).
    - NetTransformer : Transformer encoder predictor.

All models accept input of shape (batch, seq_len, input_dim) and
produce output of shape (batch, 1).
"""
import torch
import torch.nn as nn


class NetLSTM(nn.Module):
    """LSTM-based predictor with temporal memory."""

    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=0.1,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        out, _ = self.lstm(x)
        # Take the last time-step's output
        return self.fc(out[:, -1, :])


class NetMLP(nn.Module):
    """Multi-layer perceptron predictor without temporal memory."""

    def __init__(self, input_dim, seq_len, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim * seq_len, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        # Flatten (batch, seq_len, input_dim) -> (batch, seq_len * input_dim)
        x_flat = x.reshape(x.size(0), -1)
        return self.net(x_flat)


class NetTransformer(nn.Module):
    """Transformer encoder predictor."""

    def __init__(self, input_dim, hidden_dim=64, nhead=4, num_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            batch_first=True,
            dim_feedforward=hidden_dim * 2,
            dropout=0.1,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.input_proj(x)
        out = self.transformer(x)
        # Take the last time-step's output
        return self.fc(out[:, -1, :])
