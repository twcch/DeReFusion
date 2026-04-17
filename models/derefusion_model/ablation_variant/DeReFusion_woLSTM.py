import torch
import torch.nn as nn
from layers.RevIN import RevIN


# ========== 1. Building Blocks ==========
class MovingAvg(nn.Module):
    def __init__(self, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        front = x[:, :1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x_padded = torch.cat([front, x, end], dim=1)
        return self.avg(x_padded.permute(0, 2, 1)).permute(0, 2, 1)


class SeriesDecomposition(nn.Module):
    def __init__(self, kernel_size: int = 25):
        super().__init__()
        self.moving_avg = MovingAvg(kernel_size)

    def forward(self, x: torch.Tensor):
        trend = self.moving_avg(x)
        return x - trend, trend


# ========== 2. Branches ==========
class DLinearBranch(nn.Module):
    def __init__(self, seq_len: int, pred_len: int, kernel_size: int = 25):
        super().__init__()
        self.decomp = SeriesDecomposition(kernel_size)
        self.linear_seasonal = nn.Linear(seq_len, pred_len)
        self.linear_trend = nn.Linear(seq_len, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seasonal, trend = self.decomp(x)
        seasonal_out = self.linear_seasonal(seasonal.permute(0, 2, 1))
        trend_out = self.linear_trend(trend.permute(0, 2, 1))
        return (seasonal_out + trend_out).permute(0, 2, 1)


class TransformerResidualBranch(nn.Module):
    """Transformer-only residual branch (no LSTM)."""

    def __init__(self, enc_in: int, d_model: int, seq_len: int, pred_len: int, n_heads: int = 4):
        super().__init__()
        self.feature_proj = nn.Linear(enc_in, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4, batch_first=True, dropout=0.3,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.time_proj = nn.Linear(seq_len, pred_len)
        self.channel_proj = nn.Linear(d_model, enc_in)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_emb = self.feature_proj(x)
        tf_out = self.transformer(x_emb)
        out = self.time_proj(tf_out.permute(0, 2, 1)).permute(0, 2, 1)
        return self.channel_proj(out)


# ========== 3. Top-level Model (woLSTM) ==========
class Model(nn.Module):
    """
    Ablation: woLSTM — DLinear + Transformer-only residual + direct add.
    Tests the contribution of the LSTM component in the hybrid branch.
    """

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.d_model = configs.d_model
        kernel_size = getattr(configs, "moving_avg", 25)

        self.revin = RevIN(self.enc_in)
        self.dlinear_branch = DLinearBranch(self.seq_len, self.pred_len, kernel_size)
        self.res_branch = TransformerResidualBranch(
            self.enc_in, self.d_model, self.seq_len, self.pred_len
        )

    def forecast(self, x_enc: torch.Tensor) -> torch.Tensor:
        x_norm = self.revin(x_enc, "norm")
        base = self.dlinear_branch(x_norm)
        residual = self.res_branch(x_norm)
        fused = base + residual
        return self.revin(fused, "denorm")

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        return self.forecast(x_enc)