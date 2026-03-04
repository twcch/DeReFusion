import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.RevIN import RevIN
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding


class LSTMDecoder(nn.Module):
    """LSTM decoder that uses Transformer encoder memory as initial hidden state."""

    def __init__(self, dec_in, d_model, num_layers=1, dropout=0.1):
        super().__init__()
        self.embedding = nn.Linear(dec_in, d_model)
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

    def forward(self, x_dec, memory):
        """
        Args:
            x_dec:  [B, label_len + pred_len, dec_in]
            memory: [B, seq_len, d_model]  (Transformer encoder output)
        Returns:
            dec_out: [B, label_len + pred_len, d_model]
        """
        dec_emb = self.embedding(x_dec)

        # Use encoder's last time-step as LSTM initial hidden state
        last_memory = memory[:, -1, :]
        h_0 = last_memory.unsqueeze(0).expand(self.num_layers, -1, -1).contiguous()
        c_0 = torch.zeros_like(h_0)

        dec_out, _ = self.lstm(dec_emb, (h_0, c_0))
        return dec_out


class Model(nn.Module):
    """
    RevINTransLSTM — Transformer Encoder + LSTM Decoder with RevIN

    Encoder: FullAttention Transformer with DataEmbedding (positional + temporal)
    Decoder: LSTM initialized from encoder memory
    RevIN:   normalize before encoding, denormalize after decoding
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

        # LSTM Decoder + projection
        if self.task_name in ('long_term_forecast', 'short_term_forecast'):
            self.decoder = LSTMDecoder(
                dec_in=configs.dec_in,
                d_model=configs.d_model,
                num_layers=configs.d_layers,
                dropout=configs.dropout,
            )
            self.projection = nn.Linear(configs.d_model, configs.enc_in)

        if self.task_name == 'imputation':
            self.head = nn.Linear(configs.d_model, configs.enc_in)
        if self.task_name == 'anomaly_detection':
            self.head = nn.Linear(configs.d_model, configs.enc_in)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.head = nn.Linear(configs.d_model * configs.seq_len, configs.num_class)

    def _normalize_dec(self, x_dec):
        """Normalize decoder input using encoder's stored RevIN statistics."""
        x = (x_dec - self.revin.mean) / self.revin.stdev
        if self.revin.affine:
            x = x * self.revin.affine_weight + self.revin.affine_bias
        return x

    def forecast(self, x_enc, x_mark_enc, x_dec):
        # RevIN norm
        x_enc = self.revin(x_enc, 'norm')
        x_dec = self._normalize_dec(x_dec)

        # Transformer Encoder
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        memory, _ = self.encoder(enc_out)

        # LSTM Decoder
        dec_out = self.decoder(x_dec, memory)
        dec_out = self.projection(dec_out)  # [B, label_len + pred_len, enc_in]

        # RevIN denorm
        dec_out = self.revin(dec_out, 'denorm')
        return dec_out

    def imputation(self, x_enc, x_mark_enc, mask):
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
        x_enc = self.revin(x_enc, 'norm')
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, _ = self.encoder(enc_out)
        output = self.act(enc_out)
        output = self.dropout(output)
        output = output * x_mark_enc.unsqueeze(-1)
        output = output.reshape(output.shape[0], -1)
        output = self.head(output)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ('long_term_forecast', 'short_term_forecast'):
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec)
            return dec_out[:, -self.pred_len:, :]
        if self.task_name == 'imputation':
            return self.imputation(x_enc, x_mark_enc, mask)
        if self.task_name == 'anomaly_detection':
            return self.anomaly_detection(x_enc)
        if self.task_name == 'classification':
            return self.classification(x_enc, x_mark_enc)
        return None
