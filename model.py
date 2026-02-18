import torch
import torch.nn as nn
import numpy as np

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class SamePadConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        left = (kernel_size - 1) // 2
        right = kernel_size - 1 - left
        self.pad = nn.ConstantPad1d((left, right), 0.0)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=0)
        
    def forward(self, x):
        return self.conv(self.pad(x))

class MultiScaleCNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=(3, 24, 168), dropout=0.2):
        super().__init__()
        self.convs = nn.ModuleList([
            SamePadConv1d(in_channels, out_channels, kernel_size=k)
            for k in kernel_sizes
        ])
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = x.transpose(1, 2)
        outputs = []
        for conv in self.convs:
            out = self.relu(conv(x))
            out = self.dropout(out)
            outputs.append(out)
        x = torch.cat(outputs, dim=1)
        return x.transpose(1, 2)

class HybridCNNTransformerQuantile(nn.Module):
    def __init__(self, feature_dim, hidden_dim=64, num_heads=4, num_layers=2, dropout=0.2):
        super().__init__()
        self.cnn_block = MultiScaleCNNBlock(feature_dim, hidden_dim, dropout=dropout)
        cnn_output_dim = hidden_dim * len(self.cnn_block.convs)
        
        self.input_projection = nn.Linear(cnn_output_dim, hidden_dim)
        self.positional_encoding = PositionalEncoding(hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation='relu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.extreme_fusion = nn.Linear(hidden_dim + 1, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, 3)  # Output 3 quantiles
        
    def forward(self, x):
        extreme_flags = x[:, -1, -1].unsqueeze(1)  # Extract extreme flags
        x = self.cnn_block(x)
        x = self.input_projection(x)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)
        x = x.transpose(1, 2)
        x = self.pool(x).squeeze(-1)
        x_with_extreme = torch.cat([x, extreme_flags], dim=1)
        x_fused = torch.relu(self.extreme_fusion(x_with_extreme))
        return self.output_layer(x_fused)

class LSTMBaseline(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=1, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers=num_layers,
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out).squeeze(-1)

class QuantileLoss(nn.Module):
    def __init__(self, quantiles=(0.1, 0.5, 0.9)):
        super().__init__()
        self.quantiles = quantiles
        
    def forward(self, preds, target):
        losses = []
        for i, q in enumerate(self.quantiles):
            errors = target - preds[:, i]
            losses.append(torch.max((q-1)*errors, q*errors).mean())
        return sum(losses)