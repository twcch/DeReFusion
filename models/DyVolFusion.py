import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.RevIN import RevIN


# 1. Building Blocks
class MovingAvg(nn.Module):
    """Moving average block to highlight the trend of time series."""

    def __init__(self, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, C]
        Returns:
            smoothed: [B, L, C]
        """
        # Pad front and end to keep the same length
        front = x[:, :1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x_padded = torch.cat([front, x, end], dim=1)
        # AvgPool1d expects [B, C, L]
        smoothed = self.avg(x_padded.permute(0, 2, 1)).permute(0, 2, 1)
        return smoothed


class SeriesDecomposition(nn.Module):
    """Decompose a time series into seasonal and trend components."""

    def __init__(self, kernel_size: int = 25):
        super().__init__()
        self.moving_avg = MovingAvg(kernel_size)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [B, L, C]
        Returns:
            seasonal: [B, L, C]
            trend:    [B, L, C]
        """
        trend = self.moving_avg(x)
        seasonal = x - trend
        return seasonal, trend


# 2. Sub-modules (Branches)
class DLinearBranch(nn.Module):
    """
    DLinear-style branch: decompose → project seasonal & trend independently.
    """

    def __init__(self, seq_len: int, pred_len: int, kernel_size: int = 25):
        super().__init__()
        self.decomp = SeriesDecomposition(kernel_size)
        # Both projections map along the time axis: seq_len → pred_len
        self.linear_seasonal = nn.Linear(seq_len, pred_len)
        self.linear_trend = nn.Linear(seq_len, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, seq_len, C]  (already normalised)
        Returns:
            base_pred: [B, pred_len, C]
        """
        seasonal, trend = self.decomp(x)
        # [B, C, seq_len] → Linear → [B, C, pred_len]
        seasonal_out = self.linear_seasonal(seasonal.permute(0, 2, 1))
        trend_out = self.linear_trend(trend.permute(0, 2, 1))
        # [B, pred_len, C]
        return (seasonal_out + trend_out).permute(0, 2, 1)


class LSTMResidualBranch(nn.Module):
    """
    LSTM branch that captures residual dynamics beyond DLinear.
    """

    def __init__(self, enc_in: int, d_model: int, seq_len: int, pred_len: int):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=enc_in,
            hidden_size=d_model,
            num_layers=1,
            batch_first=True,
        )
        # Project along time and channel back to original space
        self.time_proj = nn.Linear(seq_len, pred_len)
        self.channel_proj = nn.Linear(d_model, enc_in)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, seq_len, C]  (already normalised)
        Returns:
            residual: [B, pred_len, C]
        """
        lstm_out, _ = self.lstm(x)                           # [B, seq_len, d_model]
        lstm_out = self.time_proj(lstm_out.permute(0, 2, 1)) # [B, d_model, pred_len]
        lstm_out = lstm_out.permute(0, 2, 1)                 # [B, pred_len, d_model]
        residual = self.channel_proj(lstm_out)               # [B, pred_len, C]
        return residual


class TransformerResidualBranch(nn.Module):
    """
    Transformer branch alternative for capturing residual dynamics.
    """
    
    def __init__(self, enc_in: int, d_model: int, seq_len: int, pred_len: int, n_heads: int = 4):
        super().__init__()
        self.feature_proj = nn.Linear(enc_in, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=n_heads, 
            dim_feedforward=d_model * 4, 
            batch_first=True,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.time_proj = nn.Linear(seq_len, pred_len)
        self.channel_proj = nn.Linear(d_model, enc_in)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, seq_len, C]  (already normalised)
        Returns:
            residual: [B, pred_len, C]
        """
        x_emb = self.feature_proj(x)                         # [B, seq_len, d_model]
        tf_out = self.transformer(x_emb)                     # [B, seq_len, d_model]
        tf_out = self.time_proj(tf_out.permute(0, 2, 1))     # [B, d_model, pred_len]
        tf_out = tf_out.permute(0, 2, 1)                     # [B, pred_len, d_model]
        residual = self.channel_proj(tf_out)                 # [B, pred_len, C]
        return residual

class TransformerLSTMResidualBranch(nn.Module):
    """
    Hybrid branch: LSTM captures local sequence patterns and acts as implicit positional encoding,
    followed by Transformer to capture global cross-time dependencies.
    """
    
    def __init__(self, enc_in: int, d_model: int, seq_len: int, pred_len: int, n_heads: int = 4):
        super().__init__()
        # 1. 將輸入特徵維度映射至 d_model
        self.feature_proj = nn.Linear(enc_in, d_model)
        
        # 2. LSTM 提取局部時序特徵
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=1,
            batch_first=True,
        )
        
        # 3. Transformer 提取全局依賴
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=n_heads, 
            dim_feedforward=d_model * 4, 
            batch_first=True,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        
        # 4. 維度還原映射
        self.time_proj = nn.Linear(seq_len, pred_len)
        self.channel_proj = nn.Linear(d_model, enc_in)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, seq_len, C] -> [B, seq_len, d_model]
        x_emb = self.feature_proj(x)
        
        # LSTM 處理: [B, seq_len, d_model]
        lstm_out, _ = self.lstm(x_emb)
        
        # Transformer 處理 (直接接收 LSTM 的輸出): [B, seq_len, d_model]
        tf_out = self.transformer(lstm_out)
        
        # 時間與特徵維度投影
        out = self.time_proj(tf_out.permute(0, 2, 1))     # [B, d_model, pred_len]
        out = out.permute(0, 2, 1)                        # [B, pred_len, d_model]
        residual = self.channel_proj(out)                 # [B, pred_len, C]
        
        return residual

# 3. Fusion Mechanism
class SigmoidGate(nn.Module):
    """
    Learnable sigmoid gate that fuses the base prediction with a residual.
    gate ∈ (0, 1):  output = (1 - gate) * base + gate * residual
    """

    def __init__(self, seq_len: int, pred_len: int, init_bias: float = -2.0):
        super().__init__()
        self.proj = nn.Linear(seq_len, pred_len)
        # Negative bias → gate ≈ 0 at init → model starts from DLinear baseline
        nn.init.constant_(self.proj.bias, init_bias)

    def forward(
        self,
        x_enc: torch.Tensor,
        base: torch.Tensor,
        residual: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x_enc:    [B, seq_len, C]  — raw (normalised) encoder input
            base:     [B, pred_len, C] — DLinear prediction
            residual: [B, pred_len, C] — Residual prediction
        Returns:
            fused: [B, pred_len, C]
        """
        gate = torch.sigmoid(self.proj(x_enc.permute(0, 2, 1)))  # [B, C, pred_len]
        gate = gate.permute(0, 2, 1)                             # [B, pred_len, C]
        fused = (1.0 - gate) * base + gate * residual
        return fused


# 4. Top-level Model
class Model(nn.Module):
    """
    DyVolFusion — Dynamic Volatility Fusion

    Combines a DLinear decomposition branch with a residual branch (LSTM/Transformer)
    via a learnable sigmoid gate, wrapped in RevIN normalisation.

    Compatible with TSLib's standard forward signature:
        forward(x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None)
    """

    def __init__(self, configs):
        super().__init__()
        self.seq_len: int = configs.seq_len
        self.pred_len: int = configs.pred_len
        self.enc_in: int = configs.enc_in
        self.d_model: int = configs.d_model
        
        # 動態決定要使用的殘差分支，預設為 LSTM
        self.res_branch_type: str = getattr(configs, "res_branch_type", "TRANSFORMER_LSTM")
        kernel_size: int = getattr(configs, "moving_avg", 25)

        # Normalisation
        self.revin = RevIN(self.enc_in)

        # Branches
        self.dlinear_branch = DLinearBranch(self.seq_len, self.pred_len, kernel_size)
        
        if self.res_branch_type == "TRANSFORMER_LSTM":
            self.res_branch = TransformerLSTMResidualBranch(
                self.enc_in, self.d_model, self.seq_len, self.pred_len
            )
        elif self.res_branch_type == "TRANSFORMER":
            self.res_branch = TransformerResidualBranch(
                self.enc_in, self.d_model, self.seq_len, self.pred_len
            )
        else:
            self.res_branch = LSTMResidualBranch(
                self.enc_in, self.d_model, self.seq_len, self.pred_len
            )

        # Fusion
        self.gate = SigmoidGate(self.seq_len, self.pred_len, init_bias=-2.0)

    # Task-specific heads
    def forecast(self, x_enc: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_enc: [B, seq_len, C]
        Returns:
            dec_out: [B, pred_len, C]
        """
        # A. RevIN Normalisation
        x_norm = self.revin(x_enc, "norm")

        # B. DLinear base prediction
        base = self.dlinear_branch(x_norm)

        # C. Residual dynamics (LSTM or Transformer)
        residual = self.res_branch(x_norm)

        # D. Gated fusion
        fused = self.gate(x_norm, base, residual)

        # E. RevIN De-normalisation
        dec_out = self.revin(fused, "denorm")
        return dec_out

    # TSLib standard forward
    def forward(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor,
        x_dec: torch.Tensor,
        x_mark_dec: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # 這裡直接回傳 forecast 結果即可，因為維度已經被投影成 [B, pred_len, C]
        dec_out = self.forecast(x_enc)
        return dec_out