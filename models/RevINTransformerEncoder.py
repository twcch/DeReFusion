import torch.nn as nn
import torch.nn.functional as F
from layers.RevIN import RevIN
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding


class Model(nn.Module):
    """
    RevINTransformerEncoder — Encoder-only Transformer with Reversible Instance Normalization

    Unlike the encoder-decoder RevINTransformer, this model has no decoder: the
    Transformer encoder produces a contextual representation of the input window,
    and a flatten-and-project head maps it directly to the forecast horizon.

    RevIN normalizes each instance before encoding and denormalizes the output,
    helping the model handle non-stationary time series with distribution shifts.

    Forward signature follows TSLib convention:
        forward(x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None)
    """

    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in

        # RevIN
        self.revin = RevIN(configs.enc_in)

        # Transformer Encoder
        self.enc_embedding = DataEmbedding(
            configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout
        )
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor,
                                      attention_dropout=configs.dropout,
                                      output_attention=False),
                        configs.d_model, configs.n_heads
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                ) for _ in range(configs.e_layers)
            ],
            norm_layer=nn.LayerNorm(configs.d_model),
        )

        # Task-specific heads
        if self.task_name in ('long_term_forecast', 'short_term_forecast'):
            # Flatten the encoded [seq_len, d_model] map and project to the
            # full forecast horizon, then reshape to [pred_len, enc_in].
            self.flatten = nn.Flatten(start_dim=1)
            self.dropout = nn.Dropout(configs.dropout)
            self.head = nn.Linear(configs.d_model * configs.seq_len,
                                   configs.pred_len * configs.enc_in)
        if self.task_name == 'imputation':
            self.head = nn.Linear(configs.d_model, configs.enc_in)
        if self.task_name == 'anomaly_detection':
            self.head = nn.Linear(configs.d_model, configs.enc_in)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.head = nn.Linear(configs.d_model * configs.seq_len, configs.num_class)

    # ------------------------------------------------------------------
    # Task-specific forward paths
    # ------------------------------------------------------------------
    def forecast(self, x_enc, x_mark_enc):
        # RevIN: normalize encoder input (stores mean / stdev for later denorm)
        x_enc = self.revin(x_enc, 'norm')

        # Transformer Encoder
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, _ = self.encoder(enc_out)          # [B, seq_len, d_model]

        # Flatten + project to the forecast horizon
        out = self.flatten(enc_out)                 # [B, seq_len * d_model]
        out = self.dropout(out)
        out = self.head(out)                        # [B, pred_len * enc_in]
        out = out.reshape(out.shape[0], self.pred_len, self.enc_in)

        # RevIN: denormalize
        out = self.revin(out, 'denorm')
        return out

    def imputation(self, x_enc, x_mark_enc):
        x_enc = self.revin(x_enc, 'norm')
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, _ = self.encoder(enc_out)
        dec_out = self.head(enc_out)
        dec_out = self.revin(dec_out, 'denorm')
        return dec_out

    def anomaly_detection(self, x_enc):
        x_enc = self.revin(x_enc, 'norm')
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, _ = self.encoder(enc_out)
        dec_out = self.head(enc_out)
        dec_out = self.revin(dec_out, 'denorm')
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        # RevIN norm only (no denorm — output is class logits)
        x_enc = self.revin(x_enc, 'norm')
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, _ = self.encoder(enc_out)

        output = self.act(enc_out)
        output = self.dropout(output)
        output = output * x_mark_enc.unsqueeze(-1)
        output = output.reshape(output.shape[0], -1)
        output = self.head(output)
        return output

    # ------------------------------------------------------------------
    # TSLib standard forward
    # ------------------------------------------------------------------
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ('long_term_forecast', 'short_term_forecast'):
            dec_out = self.forecast(x_enc, x_mark_enc)
            return dec_out[:, -self.pred_len:, :]
        if self.task_name == 'imputation':
            return self.imputation(x_enc, x_mark_enc)
        if self.task_name == 'anomaly_detection':
            return self.anomaly_detection(x_enc)
        if self.task_name == 'classification':
            return self.classification(x_enc, x_mark_enc)
        return None
