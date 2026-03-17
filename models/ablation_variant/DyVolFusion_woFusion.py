"""
DyVolFusion w/o Fusion (Ablation)

Replace the learnable SigmoidGate fusion with naive addition (base + residual).
This tests whether the gated fusion mechanism is better than simple combination.
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


class Model(nn.Module):
    """DyVolFusion w/o Fusion — simple addition replaces gated fusion."""

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

        # >>> ABLATION: no gate, just a simple addition <<<

    def forecast(self, x_enc: torch.Tensor) -> torch.Tensor:
        x_norm = self.revin(x_enc, "norm")
        base = self.dlinear_branch(x_norm)
        residual = self.res_branch(x_norm)
        # Simple addition instead of gated fusion
        fused = base + residual
        dec_out = self.revin(fused, "denorm")
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        return self.forecast(x_enc)
