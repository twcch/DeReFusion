"""
DyVolFusion w/o Vol (Ablation)

Remove the volatility-aware DLinear decomposition branch (trend/seasonal).
The residual branch output is used directly without decomposition-based base prediction.
This tests whether the volatility decomposition contributes to performance.
"""
import torch
import torch.nn as nn
from layers.RevIN import RevIN
from models.DyVolFusion import (
    LSTMResidualBranch,
    TransformerResidualBranch,
    TransformerLSTMResidualBranch,
)


class Model(nn.Module):
    """DyVolFusion w/o Vol — no DLinear decomposition branch, residual branch only."""

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.d_model = configs.d_model

        self.res_branch_type = getattr(configs, "res_branch_type", "TRANSFORMER_LSTM")

        self.revin = RevIN(self.enc_in)

        # >>> ABLATION: no DLinear branch, no gate — residual branch only <<<
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

    def forecast(self, x_enc: torch.Tensor) -> torch.Tensor:
        x_norm = self.revin(x_enc, "norm")
        dec_out = self.res_branch(x_norm)
        dec_out = self.revin(dec_out, "denorm")
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        return self.forecast(x_enc)
