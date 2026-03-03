import torch
import torch.nn as nn

from layers.Autoformer_EncDec import series_decomp
from layers.Embed import DataEmbedding
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.RevIN import RevIN


class DLinearFeatureExtractor(nn.Module):
    """
    DLinear-style seasonal-trend decomposition as a feature extractor.
    Input:  [B, seq_len, enc_in]
    Output: [B, seq_len, enc_in]  (decomposed & linearly projected)
    """

    def __init__(self, seq_len, enc_in, moving_avg=25, individual=False):
        super().__init__()
        self.seq_len = seq_len
        self.channels = enc_in
        self.individual = individual
        self.decomposition = series_decomp(moving_avg)

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()
            for i in range(self.channels):
                self.Linear_Seasonal.append(nn.Linear(self.seq_len, self.seq_len))
                self.Linear_Trend.append(nn.Linear(self.seq_len, self.seq_len))
                self.Linear_Seasonal[i].weight = nn.Parameter(
                    (1 / self.seq_len) * torch.ones([self.seq_len, self.seq_len])
                )
                self.Linear_Trend[i].weight = nn.Parameter(
                    (1 / self.seq_len) * torch.ones([self.seq_len, self.seq_len])
                )
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.seq_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.seq_len)
            self.Linear_Seasonal.weight = nn.Parameter(
                (1 / self.seq_len) * torch.ones([self.seq_len, self.seq_len])
            )
            self.Linear_Trend.weight = nn.Parameter(
                (1 / self.seq_len) * torch.ones([self.seq_len, self.seq_len])
            )

    def forward(self, x):
        # x: [B, seq_len, enc_in]
        seasonal_init, trend_init = self.decomposition(x)
        # permute to [B, enc_in, seq_len] for linear projection along time axis
        seasonal_init = seasonal_init.permute(0, 2, 1)
        trend_init = trend_init.permute(0, 2, 1)

        if self.individual:
            seasonal_output = torch.zeros_like(seasonal_init)
            trend_output = torch.zeros_like(trend_init)
            for i in range(self.channels):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](trend_init[:, i, :])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)

        # combine and permute back: [B, seq_len, enc_in]
        x = (seasonal_output + trend_output).permute(0, 2, 1)
        return x


class Model(nn.Module):
    """
    RevIN → DLinear (Decomposition Feature Extractor) → Transformer Encoder + LSTM Decoder

    Pipeline:
      1. RevIN: Instance normalization for non-stationary data
      2. DLinear: Seasonal-trend decomposition & linear projection (feature extraction)
      3. Transformer Encoder: Capture global temporal dependencies
      4. LSTM Decoder: Autoregressive decoding with cross-attention to encoder memory
      5. RevIN De-normalization: Restore original scale
    """

    def __init__(self, configs):
        super().__init__()

        # ---- Core lengths ----
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.label_len = configs.label_len
        self.seq_len = configs.seq_len

        # ---- Dims ----
        self.d_model = configs.d_model
        self.c_out = configs.c_out
        self.enc_in = configs.enc_in

        # ---- 1. RevIN ----
        use_revin = getattr(configs, "rev_in", True)
        self.revin = RevIN(configs.enc_in) if use_revin else None

        # ---- 2. DLinear Feature Extractor ----
        moving_avg = getattr(configs, "moving_avg", 25)
        individual = getattr(configs, "individual", False)
        self.dlinear = DLinearFeatureExtractor(
            seq_len=configs.seq_len,
            enc_in=configs.enc_in,
            moving_avg=moving_avg,
            individual=individual,
        )

        # ---- 3. Embeddings ----
        self.enc_embedding = DataEmbedding(
            configs.enc_in,
            configs.d_model,
            configs.embed,
            configs.freq,
            configs.dropout,
        )
        self.dec_embedding = DataEmbedding(
            configs.dec_in,
            configs.d_model,
            configs.embed,
            configs.freq,
            configs.dropout,
        )

        # ---- 4. Transformer Encoder ----
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            mask_flag=False,
                            factor=configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=False,
                        ),
                        configs.d_model,
                        configs.n_heads,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for _ in range(configs.e_layers)
            ],
            norm_layer=nn.LayerNorm(configs.d_model),
        )

        # ---- 5. LSTM Decoder ----
        d_layers = getattr(configs, "d_layers", 1)
        self.lstm = nn.LSTM(
            input_size=configs.d_model,
            hidden_size=configs.d_model,
            num_layers=d_layers,
            batch_first=True,
            dropout=configs.dropout if d_layers > 1 else 0.0,
        )

        # ---- 6. Cross-Attention (LSTM output attends to Encoder memory) ----
        self.cross_attention = AttentionLayer(
            FullAttention(
                mask_flag=False,
                factor=configs.factor,
                attention_dropout=configs.dropout,
                output_attention=False,
            ),
            d_model=configs.d_model,
            n_heads=configs.n_heads,
        )

        # ---- 7. Output Projection ----
        self.projection = nn.Linear(configs.d_model, configs.c_out)

        # ---- 8. Feedback Projection (for autoregressive loop) ----
        self.out_proj = nn.Linear(configs.c_out, configs.d_model)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """
        Long/Short-term Forecast:
        RevIN → DLinear → Transformer Encoder → LSTM AR Decoder → RevIN Denorm
        """
        # 1. RevIN Normalization
        if self.revin is not None:
            x_enc = self.revin(x_enc, "norm")
            # Also normalize the decoder input values
            x_dec_vals = x_dec[:, :, :self.enc_in]
            x_dec_vals = self.revin(x_dec_vals, "norm")
            x_dec = torch.cat([x_dec_vals, x_dec[:, :, self.enc_in:]], dim=-1)

        # 2. DLinear Decomposition Feature Extraction
        x_enc = self.dlinear(x_enc)

        # 3. Transformer Encoder
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)  # [B, seq_len, d_model]

        # 4. Decoder Seed (label_len portion of x_dec)
        dec_input = x_dec[:, :self.label_len, :]
        dec_mark = x_mark_dec[:, :self.label_len, :]
        dec_embed = self.dec_embedding(dec_input, dec_mark)  # [B, label_len, d_model]

        # 5. Autoregressive LSTM Decoding with Cross-Attention
        lstm_input = dec_embed[:, -1:, :]  # [B, 1, d_model]
        hidden = None
        outputs = []

        for _ in range(self.pred_len):
            # A. LSTM step
            lstm_out, hidden = self.lstm(lstm_input, hidden)  # [B, 1, d_model]

            # B. Cross-Attention with encoder memory
            attn_out, _ = self.cross_attention(
                lstm_out, enc_out, enc_out, attn_mask=None
            )  # [B, 1, d_model]

            # C. Project to output space
            pred = self.projection(attn_out)  # [B, 1, c_out]
            outputs.append(pred)

            # D. Feedback: project prediction back to latent space for next step
            pred_feedback = self.out_proj(pred)  # [B, 1, d_model]
            lstm_input = attn_out + pred_feedback  # Residual connection

        dec_out = torch.cat(outputs, dim=1)  # [B, pred_len, c_out]

        # 6. RevIN De-normalization
        if self.revin is not None:
            dec_out = self.revin(dec_out, "denorm")

        # 7. Prepend zeros for x_enc portion (TSLib convention)
        dec_out = torch.cat([torch.zeros_like(x_enc), dec_out], dim=1)

        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == "long_term_forecast" or self.task_name == "short_term_forecast":
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        return None