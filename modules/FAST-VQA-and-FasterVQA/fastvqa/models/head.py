import torch.nn as nn


class VQAHead(nn.Module):
    def __init__(self, in_channels=768, hidden_channels=64, dropout_ratio=0.5, **kwargs):
        del kwargs
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_ratio) if dropout_ratio else nn.Identity()
        self.fc_hid = nn.Conv3d(in_channels, hidden_channels, (1, 1, 1))
        self.fc_last = nn.Conv3d(hidden_channels, 1, (1, 1, 1))
        self.gelu = nn.GELU()

    def forward(self, x, rois=None):
        del rois
        x = self.dropout(x)
        return self.fc_last(self.dropout(self.gelu(self.fc_hid(x))))
