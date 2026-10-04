"""Member 3 - CNN-LSTM (recurrent hybrid).

The image is cut into a grid of 32x32 patches (7x7 = 49 for a 224x224 input).
A small CNN turns every patch into a 32-d feature vector, the 49 vectors form a
sequence, and an LSTM reads that sequence. Classification uses the final
hidden state.
"""
import torch
import torch.nn as nn


class CNNLSTM(nn.Module):
    def __init__(self, num_classes=4, patch_size=32, img_size=224, hidden_size=128):
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        self.patch_size = patch_size
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1)
        )
        self.lstm = nn.LSTM(input_size=32, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        B = x.size(0)
        p = self.patch_size
        patches = x.unfold(2, p, p).unfold(3, p, p)          # B,C,nH,nW,p,p
        patches = patches.contiguous().view(B, 3, -1, p, p).permute(0, 2, 1, 3, 4)  # B,N,C,p,p
        N = patches.size(1)
        feats = [self.cnn(patches[:, i]).view(B, -1) for i in range(N)]
        seq = torch.stack(feats, dim=1)  # B, N, 32
        _, (h_n, _) = self.lstm(seq)
        return self.fc(h_n[-1])
