import torch
import torch.nn as nn


class CNNBiLSTM(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # CNN frontend
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),
            nn.Dropout(0.2),

            nn.Conv2d(32, 64, kernel_size=(3, 1)),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),
            nn.Dropout(0.2),
        )

        # After CNN, we flatten into 512-dim vectors per timestep
        # This matches checkpoint: LSTM input_size = 512
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )

        # Classifier head (exactly from checkpoint)
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        """
        Input shape:
            x: (B, T, J, C)
        We reshape to:
            (B, 1, T, J*C) = (B, 1, T, 66)
        """

        B, T, J, C = x.shape
        x = x.view(B, 1, T, J * C)  # (B, 1, T, 66)

        x = self.cnn(x)             # (B, 64, T', W')

        # Force width to 8 so that 64 * 8 = 512 (as in training)
        x = torch.nn.functional.adaptive_avg_pool2d(x, (x.size(2), 8))
        # Now: (B, 64, T', 8)

        x = x.permute(0, 2, 1, 3).contiguous()  # (B, T', 64, 8)
        x = x.view(x.size(0), x.size(1), -1)    # (B, T', 512)

        out, _ = self.lstm(x)       # (B, T', 256)
        out = out[:, -1, :]         # (B, 256)

        return self.classifier(out)


