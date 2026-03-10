"""
DyVolFusion w/o Dy (Ablation)

Remove the dynamic (input-dependent) gating mechanism.
Replace SigmoidGate with a static learnable scalar gate that does NOT depend on input.
This tests whether the data-driven dynamic weighting is essential.
"""
import torch
import torch.nn as nn
from layers.RevIN import RevIN
from models.DyVolFusion import (
    DLinearBranch,
    LSTMResidualBranch,
    TransformerResidualBranch,
    TransformerLSTMResidualBranch,
)


class StaticGate(nn.Module):
    """
    Static gate: a single learnable scalar (per channel) that does NOT
    depend on the input.  gate = sigmoid(bias), broadcast over B and T.
    """

    def __init__(self, enc_in: int, init_bias: float = -2.0):
        super().__init__()
        # One learnable parameter per channel
        self.bias = nn.Parameter(torch.full((1, 1, enc_in), init_bias))

    def forward(
        self,
        x_enc: torch.Tensor,
        base: torch.Tensor,
        residual: torch.Tensor,
    ) -> torch.Tensor:
        gate = torch.sigmoid(self.bias)          # [1, 1, C]
        fused = (1.0 - gate) * base + gate * residual
        return fused


class Model(nn.Module):
    """DyVolFusion w/o Dy — static gate replaces dynamic SigmoidGate."""

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.d_model = configs.d_model

        self.res_branch_type = getattr(configs, "res_branch_type", "TRANSFORMER_LSTM")
        kernel_size = getattr(configs, "moving_avg", 25)

        self.revin = RevIN(self.enc_in)
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

        # >>> ABLATION: static gate instead of dynamic SigmoidGate <<<
        self.gate = StaticGate(self.enc_in, init_bias=-2.0)

    def forecast(self, x_enc: torch.Tensor) -> torch.Tensor:
        x_norm = self.revin(x_enc, "norm")
        base = self.dlinear_branch(x_norm)
        residual = self.res_branch(x_norm)
        fused = self.gate(x_norm, base, residual)
        dec_out = self.revin(fused, "denorm")
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        return self.forecast(x_enc)
