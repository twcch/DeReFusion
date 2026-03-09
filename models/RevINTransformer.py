import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.RevIN import RevIN
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding


class Model(nn.Module):
    """
    RevINTransformer — Encoder-Decoder Transformer with Reversible Instance Normalization

    RevIN normalizes each instance before encoding and denormalizes after decoding,
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

        # Encoder
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

        # Forecasting: decoder projects to enc_in so RevIN denorm dimensions match
        if self.task_name in ('long_term_forecast', 'short_term_forecast'):
            self.dec_embedding = DataEmbedding(
                configs.dec_in, configs.d_model, configs.embed, configs.freq, configs.dropout
            )
            self.decoder = Decoder(
                [
                    DecoderLayer(
                        AttentionLayer(
                            FullAttention(True, configs.factor,
                                          attention_dropout=configs.dropout,
                                          output_attention=False),
                            configs.d_model, configs.n_heads
                        ),
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
                    )
                    for _ in range(configs.d_layers)
                ],
                norm_layer=nn.LayerNorm(configs.d_model),
                projection=nn.Linear(configs.d_model, configs.enc_in, bias=True),
            )

        if self.task_name == 'imputation':
            self.projection = nn.Linear(configs.d_model, configs.enc_in, bias=True)
        if self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(configs.d_model, configs.enc_in, bias=True)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(configs.d_model * configs.seq_len, configs.num_class)

    # ------------------------------------------------------------------
    # Helpers: normalize x_dec with the same statistics stored by RevIN
    # ------------------------------------------------------------------
    def _normalize_dec(self, x_dec):
        """Normalize decoder input using encoder's RevIN statistics."""
        x = (x_dec - self.revin.mean) / self.revin.stdev
        if self.revin.affine:
            x = x * self.revin.affine_weight + self.revin.affine_bias
        return x

    # ------------------------------------------------------------------
    # Task-specific heads
    # ------------------------------------------------------------------
    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # RevIN: normalize encoder input (stores mean / stdev for later denorm)
        x_enc = self.revin(x_enc, 'norm')
        x_dec = self._normalize_dec(x_dec)

        # Encoder
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, _ = self.encoder(enc_out)

        # Decoder  ->  [B, label_len + pred_len, enc_in]
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None)

        # RevIN: denormalize
        dec_out = self.revin(dec_out, 'denorm')
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        x_enc = self.revin(x_enc, 'norm')

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, _ = self.encoder(enc_out)
        dec_out = self.projection(enc_out)

        dec_out = self.revin(dec_out, 'denorm')
        return dec_out

    def anomaly_detection(self, x_enc):
        x_enc = self.revin(x_enc, 'norm')

        enc_out = self.enc_embedding(x_enc, None)
        enc_out, _ = self.encoder(enc_out)
        dec_out = self.projection(enc_out)

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
        output = self.projection(output)
        return output

    # ------------------------------------------------------------------
    # TSLib standard forward
    # ------------------------------------------------------------------
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ('long_term_forecast', 'short_term_forecast'):
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out
        return None
