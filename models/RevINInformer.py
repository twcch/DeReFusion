import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.RevIN import RevIN
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from layers.SelfAttention_Family import ProbAttention, AttentionLayer
from layers.Embed import DataEmbedding


class Model(nn.Module):
    """
    RevINInformer — Informer with Reversible Instance Normalization

    The Informer backbone (ProbSparse attention in O(LlogL) and the distilling
    encoder) is identical to TSLib's Informer. RevIN normalizes each instance
    before the embedding and denormalizes after decoding, helping the model
    handle non-stationary time series with distribution shifts.

    Forward signature follows TSLib convention:
        forward(x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None)
    """

    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.label_len = configs.label_len
        self.enc_in = configs.enc_in

        # RevIN
        self.revin = RevIN(configs.enc_in)

        # Embedding
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        ProbAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            [
                ConvLayer(
                    configs.d_model
                ) for l in range(configs.e_layers - 1)
            ] if configs.distil and ('forecast' in configs.task_name) else None,
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        # Decoder — projects to enc_in so RevIN denorm dimensions match
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        ProbAttention(True, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    AttentionLayer(
                        ProbAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.enc_in, bias=True)
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
    # Helper: normalize x_dec with the same statistics stored by RevIN
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

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None)

        # RevIN: denormalize
        dec_out = self.revin(dec_out, 'denorm')
        return dec_out  # [B, L, D]

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        x_enc = self.revin(x_enc, 'norm')

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        dec_out = self.projection(enc_out)

        dec_out = self.revin(dec_out, 'denorm')
        return dec_out

    def anomaly_detection(self, x_enc):
        x_enc = self.revin(x_enc, 'norm')

        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        dec_out = self.projection(enc_out)

        dec_out = self.revin(dec_out, 'denorm')
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        # RevIN norm only (no denorm — output is class logits)
        x_enc = self.revin(x_enc, 'norm')

        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

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
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        return None
