"""
Bidirectional LSTM model for time series classification
"""

import torch
import torch.nn as nn


class BiLSTM(nn.Module):
    """Bidirectional LSTM for multivariate time series classification"""

    def __init__(self, n_channels, n_classes):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=n_channels,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3,
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, 128),  # 256 = 128 * 2 (bidirectional)
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize model weights"""
        for name, param in self.lstm.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0)

        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Forward pass"""
        # x: (batch, channels, time) -> (batch, time, channels)
        x = x.transpose(1, 2)

        # LSTM forward pass
        lstm_out, (h_n, _) = self.lstm(x)

        # Use last hidden state from both directions
        # h_n: (num_layers * num_directions, batch, hidden_size)
        h_forward = h_n[-2]  # Last layer, forward direction
        h_backward = h_n[-1]  # Last layer, backward direction
        x = torch.cat([h_forward, h_backward], dim=1)

        # Classification
        x = self.classifier(x)
        return x
