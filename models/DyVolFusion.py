import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.RevIN import RevIN

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x

class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class Model(nn.Module):
    """
    Hybrid Architecture: RevIN (Forced) + DLinear Base + LSTM Residual + Sigmoid Gating
    TSLib Compatible Interface
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        
        # 1. 強制開啟 RevIN，依賴標準 configs.enc_in
        self.revin = RevIN(self.enc_in)
        
        # 2. DLinear Base
        # 這裡使用固定的 kernel_size=25，若 TSLib 的 configs 中有 moving_avg 參數也可替換
        kernel_size = getattr(configs, 'moving_avg', 25) 
        self.decomp = series_decomp(kernel_size)
        self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
        self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)
        
        # 3. LSTM Residual Branch
        # 使用 getattr 作為防呆機制，若未傳入 d_model 則預設為 64
        self.d_model = getattr(configs, 'd_model', 64)
        self.lstm = nn.LSTM(
            input_size=self.enc_in, 
            hidden_size=self.d_model, 
            num_layers=1, 
            batch_first=True
        )
        self.lstm_time_proj = nn.Linear(self.seq_len, self.pred_len)
        self.lstm_channel_proj = nn.Linear(self.d_model, self.enc_in)
        
        # 4. Sigmoid Gating Mechanism
        self.gate_proj = nn.Linear(self.seq_len, self.pred_len)
        # 初始化偏差，確保訓練初期 LSTM 殘差分支能夠傳遞梯度
        nn.init.constant_(self.gate_proj.bias, 0.5) 

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # TSLib 標準 forward 簽名。DLinear 家族通常只需要過去的序列 x_enc
        # x_enc shape: [Batch, seq_len, Channels]
        
        # --- A. RevIN Normalization (強制執行) ---
        x_enc_norm = self.revin(x_enc, 'norm')
            
        # --- B. DLinear Base Prediction ---
        seasonal_init, trend_init = self.decomp(x_enc_norm)
        
        # 維度轉換 [Batch, Channels, seq_len] 以符合 Linear 層計算
        seasonal_init = seasonal_init.permute(0, 2, 1)
        trend_init = trend_init.permute(0, 2, 1)
        
        seasonal_out = self.Linear_Seasonal(seasonal_init)
        trend_out = self.Linear_Trend(trend_init)
        
        # 基準預測結果 [Batch, pred_len, Channels]
        base_pred = (seasonal_out + trend_out).permute(0, 2, 1)
        
        # --- C. LSTM Residual Prediction ---
        # 輸入經過 RevIN 正規化的特徵
        lstm_out, _ = self.lstm(x_enc_norm) # [Batch, seq_len, d_model]
        
        lstm_out = lstm_out.permute(0, 2, 1) # [Batch, d_model, seq_len]
        res_pred = self.lstm_time_proj(lstm_out) # [Batch, d_model, pred_len]
        res_pred = res_pred.permute(0, 2, 1) # [Batch, pred_len, d_model]
        res_pred = self.lstm_channel_proj(res_pred) # [Batch, pred_len, Channels]
        
        # --- D. Dynamic Gating ---
        # 門控權重計算 [Batch, seq_len, Channels] -> [Batch, Channels, pred_len]
        gate_weight = torch.sigmoid(self.gate_proj(x_enc_norm.permute(0, 2, 1))).permute(0, 2, 1)
        
        # --- E. Fusion ---
        final_pred = base_pred + (gate_weight * res_pred)
        
        # --- F. RevIN Denormalization ---
        final_pred = self.revin(final_pred, 'denorm')
            
        return final_pred # [Batch, pred_len, Channels]




# import torch
# import torch.nn as nn

# from layers.Autoformer_EncDec import series_decomp
# from layers.Embed import DataEmbedding
# from layers.SelfAttention_Family import FullAttention, AttentionLayer
# from layers.Transformer_EncDec import Encoder, EncoderLayer
# from layers.RevIN import RevIN


# class DLinearFeatureExtractor(nn.Module):
#     """
#     DLinear-style seasonal-trend decomposition as a feature extractor.
#     Input:  [B, seq_len, enc_in]
#     Output: [B, seq_len, enc_in]  (decomposed & linearly projected)
#     """

#     def __init__(self, seq_len, enc_in, moving_avg=25, individual=False):
#         super().__init__()
#         self.seq_len = seq_len
#         self.channels = enc_in
#         self.individual = individual
#         self.decomposition = series_decomp(moving_avg)

#         if self.individual:
#             self.Linear_Seasonal = nn.ModuleList()
#             self.Linear_Trend = nn.ModuleList()
#             for i in range(self.channels):
#                 self.Linear_Seasonal.append(nn.Linear(self.seq_len, self.seq_len))
#                 self.Linear_Trend.append(nn.Linear(self.seq_len, self.seq_len))
#                 self.Linear_Seasonal[i].weight = nn.Parameter(
#                     (1 / self.seq_len) * torch.ones([self.seq_len, self.seq_len])
#                 )
#                 self.Linear_Trend[i].weight = nn.Parameter(
#                     (1 / self.seq_len) * torch.ones([self.seq_len, self.seq_len])
#                 )
#         else:
#             self.Linear_Seasonal = nn.Linear(self.seq_len, self.seq_len)
#             self.Linear_Trend = nn.Linear(self.seq_len, self.seq_len)
#             self.Linear_Seasonal.weight = nn.Parameter(
#                 (1 / self.seq_len) * torch.ones([self.seq_len, self.seq_len])
#             )
#             self.Linear_Trend.weight = nn.Parameter(
#                 (1 / self.seq_len) * torch.ones([self.seq_len, self.seq_len])
#             )

#     def forward(self, x):
#         # x: [B, seq_len, enc_in]
#         seasonal_init, trend_init = self.decomposition(x)
#         # permute to [B, enc_in, seq_len] for linear projection along time axis
#         seasonal_init = seasonal_init.permute(0, 2, 1)
#         trend_init = trend_init.permute(0, 2, 1)

#         if self.individual:
#             seasonal_output = torch.zeros_like(seasonal_init)
#             trend_output = torch.zeros_like(trend_init)
#             for i in range(self.channels):
#                 seasonal_output[:, i, :] = self.Linear_Seasonal[i](seasonal_init[:, i, :])
#                 trend_output[:, i, :] = self.Linear_Trend[i](trend_init[:, i, :])
#         else:
#             seasonal_output = self.Linear_Seasonal(seasonal_init)
#             trend_output = self.Linear_Trend(trend_init)

#         # combine and permute back: [B, seq_len, enc_in]
#         x = (seasonal_output + trend_output).permute(0, 2, 1)
#         return x


# class Model(nn.Module):
#     """
#     RevIN → DLinear (Decomposition Feature Extractor) → Transformer Encoder + LSTM Decoder

#     Pipeline:
#       1. RevIN: Instance normalization for non-stationary data
#       2. DLinear: Seasonal-trend decomposition & linear projection (feature extraction)
#       3. Transformer Encoder: Capture global temporal dependencies
#       4. LSTM Decoder: Autoregressive decoding with cross-attention to encoder memory
#       5. RevIN De-normalization: Restore original scale
#     """

#     def __init__(self, configs):
#         super().__init__()

#         # ---- Core lengths ----
#         self.task_name = configs.task_name
#         self.pred_len = configs.pred_len
#         self.label_len = configs.label_len
#         self.seq_len = configs.seq_len

#         # ---- Dims ----
#         self.d_model = configs.d_model
#         self.c_out = configs.c_out
#         self.enc_in = configs.enc_in

#         # ---- 1. RevIN ----
#         use_revin = getattr(configs, "rev_in", True)
#         self.revin = RevIN(configs.enc_in) if use_revin else None

#         # ---- 2. DLinear Feature Extractor ----
#         moving_avg = getattr(configs, "moving_avg", 25)
#         individual = getattr(configs, "individual", False)
#         self.dlinear = DLinearFeatureExtractor(
#             seq_len=configs.seq_len,
#             enc_in=configs.enc_in,
#             moving_avg=moving_avg,
#             individual=individual,
#         )

#         # ---- 3. Embeddings ----
#         self.enc_embedding = DataEmbedding(
#             configs.enc_in,
#             configs.d_model,
#             configs.embed,
#             configs.freq,
#             configs.dropout,
#         )
#         self.dec_embedding = DataEmbedding(
#             configs.dec_in,
#             configs.d_model,
#             configs.embed,
#             configs.freq,
#             configs.dropout,
#         )

#         # ---- 4. Transformer Encoder ----
#         self.encoder = Encoder(
#             [
#                 EncoderLayer(
#                     AttentionLayer(
#                         FullAttention(
#                             mask_flag=False,
#                             factor=configs.factor,
#                             attention_dropout=configs.dropout,
#                             output_attention=False,
#                         ),
#                         configs.d_model,
#                         configs.n_heads,
#                     ),
#                     configs.d_model,
#                     configs.d_ff,
#                     dropout=configs.dropout,
#                     activation=configs.activation,
#                 )
#                 for _ in range(configs.e_layers)
#             ],
#             norm_layer=nn.LayerNorm(configs.d_model),
#         )

#         # ---- 5. LSTM Decoder ----
#         d_layers = getattr(configs, "d_layers", 1)
#         self.lstm = nn.LSTM(
#             input_size=configs.d_model,
#             hidden_size=configs.d_model,
#             num_layers=d_layers,
#             batch_first=True,
#             dropout=configs.dropout if d_layers > 1 else 0.0,
#         )

#         # ---- 6. Cross-Attention (LSTM output attends to Encoder memory) ----
#         self.cross_attention = AttentionLayer(
#             FullAttention(
#                 mask_flag=False,
#                 factor=configs.factor,
#                 attention_dropout=configs.dropout,
#                 output_attention=False,
#             ),
#             d_model=configs.d_model,
#             n_heads=configs.n_heads,
#         )

#         # ---- 7. Output Projection ----
#         self.projection = nn.Linear(configs.d_model, configs.c_out)

#         # ---- 8. Feedback Projection (for autoregressive loop) ----
#         self.out_proj = nn.Linear(configs.c_out, configs.d_model)

#     def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
#         """
#         Long/Short-term Forecast:
#         RevIN → DLinear → Transformer Encoder → LSTM AR Decoder → RevIN Denorm
#         """
#         # 1. RevIN Normalization
#         if self.revin is not None:
#             x_enc = self.revin(x_enc, "norm")
#             # Also normalize the decoder input values
#             x_dec_vals = x_dec[:, :, :self.enc_in]
#             x_dec_vals = self.revin(x_dec_vals, "norm")
#             x_dec = torch.cat([x_dec_vals, x_dec[:, :, self.enc_in:]], dim=-1)

#         # 2. DLinear Decomposition Feature Extraction
#         x_enc = self.dlinear(x_enc)

#         # 3. Transformer Encoder
#         enc_out = self.enc_embedding(x_enc, x_mark_enc)
#         enc_out, _ = self.encoder(enc_out, attn_mask=None)  # [B, seq_len, d_model]

#         # 4. Decoder Seed (label_len portion of x_dec)
#         dec_input = x_dec[:, :self.label_len, :]
#         dec_mark = x_mark_dec[:, :self.label_len, :]
#         dec_embed = self.dec_embedding(dec_input, dec_mark)  # [B, label_len, d_model]

#         # 5. Autoregressive LSTM Decoding with Cross-Attention
#         lstm_input = dec_embed[:, -1:, :]  # [B, 1, d_model]
#         hidden = None
#         outputs = []

#         for _ in range(self.pred_len):
#             # A. LSTM step
#             lstm_out, hidden = self.lstm(lstm_input, hidden)  # [B, 1, d_model]

#             # B. Cross-Attention with encoder memory
#             attn_out, _ = self.cross_attention(
#                 lstm_out, enc_out, enc_out, attn_mask=None
#             )  # [B, 1, d_model]

#             # C. Project to output space
#             pred = self.projection(attn_out)  # [B, 1, c_out]
#             outputs.append(pred)

#             # D. Feedback: project prediction back to latent space for next step
#             pred_feedback = self.out_proj(pred)  # [B, 1, d_model]
#             lstm_input = attn_out + pred_feedback  # Residual connection

#         dec_out = torch.cat(outputs, dim=1)  # [B, pred_len, c_out]

#         # 6. RevIN De-normalization
#         if self.revin is not None:
#             dec_out = self.revin(dec_out, "denorm")

#         # 7. Prepend zeros for x_enc portion (TSLib convention)
#         dec_out = torch.cat([torch.zeros_like(x_enc), dec_out], dim=1)

#         return dec_out

#     def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
#         if self.task_name == "long_term_forecast" or self.task_name == "short_term_forecast":
#             dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
#             return dec_out[:, -self.pred_len:, :]  # [B, L, D]
#         return None