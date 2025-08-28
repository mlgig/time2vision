"""
Vision Transformer adapter for time series classification
"""

import numpy as np
import torch
import torch.nn as nn


class ViTAdapter(nn.Module):
    """Vision Transformer adapter for time series data"""

    def __init__(self, n_classes, channel_strategy="first", use_pretrained=False):
        super().__init__()

        self.channel_strategy = channel_strategy
        self.use_pretrained = use_pretrained

        # Try to load ViT, fallback to CNN if failed
        try:
            from transformers import ViTConfig, ViTModel

            if use_pretrained:
                try:
                    self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224")
                    hidden_size = 768
                    self.use_vit = True
                    print("✅ Using pretrained ViT")
                except:
                    # Create smaller ViT from scratch
                    config = ViTConfig(
                        image_size=224,
                        patch_size=16,
                        num_channels=3,
                        hidden_size=384,
                        num_hidden_layers=6,
                        num_attention_heads=6,
                    )
                    self.vit = ViTModel(config)
                    hidden_size = 384
                    self.use_vit = True
                    print("Using custom ViT (pretrained failed)")
            else:
                # Small ViT config for efficiency
                config = ViTConfig(
                    image_size=224,
                    patch_size=16,
                    num_channels=3,
                    hidden_size=384,
                    num_hidden_layers=6,
                    num_attention_heads=6,
                )
                self.vit = ViTModel(config)
                hidden_size = 384
                self.use_vit = True

            self.classifier = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(hidden_size, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(256, n_classes),
            )

        except Exception as e:
            print(f"⚠️ ViT failed, using CNN backbone: {e}")
            self.use_vit = False
            self._create_cnn_backbone(n_classes)

        self._init_weights()

    def _create_cnn_backbone(self, n_classes):
        """Create CNN backbone as ViT fallback"""
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((7, 7)),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, n_classes),
        )

    def _init_weights(self):
        """Initialize weights"""
        if not self.use_vit:
            for m in self.modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    nn.init.kaiming_normal_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
        else:
            # Only initialize classifier for ViT
            for m in self.classifier.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight)
                    nn.init.constant_(m.bias, 0)

    def _select_channels(self, x, n_channels):
        """Select channels based on strategy"""
        if n_channels >= 3:
            if self.channel_strategy == "first":
                return x[:, :3, :]
            elif self.channel_strategy == "random":
                # Fixed random selection for reproducibility
                torch.manual_seed(42)
                indices = torch.randperm(n_channels)[:3]
                return x[:, indices, :]
            elif self.channel_strategy == "pca":
                # Simple PCA-like: take first 3 channels for now
                # Could implement actual PCA here if needed
                return x[:, :3, :]
            else:
                return x[:, :3, :]
        else:
            # Repeat channels to get 3
            repeat_factor = 3 // n_channels + 1
            x = x.repeat(1, repeat_factor, 1)
            return x[:, :3, :]

    def _convert_to_image(self, x):
        """Convert time series to 2D image"""
        batch_size, n_channels, seq_len = x.shape

        # Convert to square image
        img_size = int(np.ceil(np.sqrt(seq_len)))
        target_size = img_size * img_size

        # Pad or truncate to square
        if seq_len < target_size:
            x = torch.nn.functional.pad(x, (0, target_size - seq_len))
        elif seq_len > target_size:
            x = x[:, :, :target_size]

        # Reshape to square image
        x = x.reshape(batch_size, n_channels, img_size, img_size)

        # Resize to 224x224 for ViT
        if img_size != 224:
            x = torch.nn.functional.interpolate(
                x, size=(224, 224), mode="bilinear", align_corners=False
            )

        return x

    def forward(self, x):
        """Forward pass"""
        batch_size, n_channels, seq_len = x.shape

        # Channel selection
        x = self._select_channels(x, n_channels)

        # Convert to image
        x = self._convert_to_image(x)

        # Forward pass
        if self.use_vit:
            try:
                outputs = self.vit(pixel_values=x)
                pooled = outputs.pooler_output
                return self.classifier(pooled)
            except Exception as e:
                print(f"ViT forward failed: {e}, falling back to CNN")
                # Fallback to CNN if ViT fails during forward
                if not hasattr(self, "backbone"):
                    self._create_cnn_backbone(self.classifier[-1].out_features)
                return self._forward_cnn(x)
        else:
            return self._forward_cnn(x)

    def _forward_cnn(self, x):
        """CNN forward pass"""
        x = self.backbone(x)
        x = x.flatten(1)
        return self.classifier(x)
