"""
Transformer model for time series classification
"""

import torch
import torch.nn as nn


class TransformerClassifier(nn.Module):
    """Transformer encoder for multivariate time series classification"""

    def __init__(self, n_channels, n_classes, d_model=128, nhead=8, num_layers=4):
        super().__init__()

        self.d_model = d_model

        # Input projection
        self.input_projection = nn.Linear(n_channels, d_model)

        # Positional encoding (learnable)
        self.pos_embedding = nn.Parameter(torch.randn(1, 1000, d_model) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=512,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(d_model, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, n_classes),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize model weights"""
        nn.init.kaiming_normal_(self.input_projection.weight)
        nn.init.constant_(self.input_projection.bias, 0)

        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Forward pass"""
        # x: (batch, channels, time) -> (batch, time, channels)
        x = x.transpose(1, 2)
        batch_size, seq_len, _ = x.shape

        # Limit sequence length for memory efficiency
        max_len = min(seq_len, 1000)
        x = x[:, :max_len, :]

        # Project to model dimension
        x = self.input_projection(x)

        # Add positional encoding
        x = x + self.pos_embedding[:, :max_len, :]

        # Transformer encoding
        x = self.transformer(x)

        # Global average pooling
        x = x.mean(dim=1)

        # Classification
        x = self.classifier(x)
        return x
