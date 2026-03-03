import torch
import torch.nn as nn

from layers.Embed import DataEmbedding
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.RevIN import RevIN


class CrossAttention(nn.Module):
    """
    Standard Cross-Attention
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, factor: int = 5):
        super().__init__()
        self.attention = AttentionLayer(
            FullAttention(
                mask_flag=False,
                factor=factor,
                attention_dropout=dropout,
                output_attention=False
            ),
            d_model=d_model,
            n_heads=n_heads
        )

    def forward(self, query, key, value, attn_mask=None):
        out, _ = self.attention(query, key, value, attn_mask=attn_mask)
        return out


class Model(nn.Module):
    """
    TransLSTM-AR v2 (Robust Version)
    
    Improvements:
    1. Robust Config Parsing: Fixes 'AttributeError' by using default values.
    2. RevIN: Handles non-stationary stock data.
    3. Feedback Loop: Feeds prediction back into LSTM for better AR decoding.
    """
    def __init__(self, configs):
        super().__init__()

        # ---- core lengths ----
        self.pred_len = configs.pred_len
        self.label_len = configs.label_len

        # ---- dims ----
        self.d_model = configs.d_model
        self.c_out = configs.c_out
        self.enc_in = configs.enc_in
        
        # [CRITICAL FIX] 使用 getattr 設置預設值為 True，防止報錯
        # 如果你的參數檔裡沒有 rev_in，這裡會自動設為 True
        use_revin = getattr(configs, 'rev_in', True) 
        self.revin = RevIN(configs.enc_in) if use_revin else None

        # ---- embedding ----
        self.enc_embedding = DataEmbedding(
            configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout
        )
        self.dec_embedding = DataEmbedding(
            configs.dec_in, configs.d_model, configs.embed, configs.freq, configs.dropout
        )

        # ---- Transformer Encoder ----
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            mask_flag=False,
                            factor=configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=False
                        ),
                        configs.d_model,
                        configs.n_heads
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                )
                for _ in range(configs.e_layers)
            ],
            norm_layer=nn.LayerNorm(configs.d_model)
        )

        # ---- LSTM Decoder ----
        self.lstm = nn.LSTM(
            input_size=configs.d_model,
            hidden_size=configs.d_model,
            num_layers=getattr(configs, "d_layers", 1),
            batch_first=True,
            dropout=configs.dropout if getattr(configs, "d_layers", 1) > 1 else 0.0
        )

        # ---- Cross-Attention ----
        self.cross_attention = CrossAttention(
            d_model=configs.d_model,
            n_heads=configs.n_heads,
            dropout=configs.dropout,
            factor=configs.factor
        )

        # ---- Output head ----
        self.projection = nn.Linear(configs.d_model, configs.c_out)
        
        # [NEW] Output Feedback Projection
        # 用於將預測結果投影回隱藏層維度，增強 LSTM 的連續推理能力
        self.out_proj = nn.Linear(configs.c_out, configs.d_model)

    def forward(
        self,
        x_enc, x_mark_enc,
        x_dec, x_mark_dec,
        enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None
    ):
        """
        Standard Forward Pass with RevIN and Feedback Loop
        """
        
        # 1. [Critical Step] RevIN Normalization (處理股價分佈)
        if self.revin is not None:
            x_enc = self.revin(x_enc, 'norm')
            
            # 對 Decoder 輸入也進行標準化處理，確保特徵分佈一致
            # 這裡只取數值部分進行 Norm，保留可能存在的時間標記 embedding
            x_dec_input_vals = x_dec[:, :, :self.enc_in]
            x_dec_input_vals = self.revin(x_dec_input_vals, 'norm')
            # 將 Norm 後的數值拼回
            x_dec = torch.cat([x_dec_input_vals, x_dec[:, :, self.enc_in:]], dim=-1)

        # 2. Encode
        enc_out = self.enc_embedding(x_enc, x_mark_enc)         
        enc_out, _ = self.encoder(enc_out, attn_mask=enc_self_mask)

        # 3. Prepare Decoder Seed
        dec_input = x_dec[:, :self.label_len, :]                
        dec_mark = x_mark_dec[:, :self.label_len, :]
        dec_embed = self.dec_embedding(dec_input, dec_mark)     

        # LSTM Start Token
        lstm_input = dec_embed[:, -1:, :]                       # [B, 1, D]
        hidden = None
        outputs = []

        # 4. Autoregressive Decoding Loop (Enhanced)
        for i in range(self.pred_len):
            # A. LSTM Step
            lstm_out, hidden = self.lstm(lstm_input, hidden)    # [B, 1, D]

            # B. Cross Attention (Encoder Memory)
            attn_out = self.cross_attention(
                query=lstm_out,
                key=enc_out,
                value=enc_out,
                attn_mask=dec_enc_mask
            )                                                   # [B, 1, D]
            #attn_out = lstm_out

            # C. Projection (Prediction)
            pred = self.projection(attn_out)                    # [B, 1, c_out]
            outputs.append(pred)

            # D. [Feedback Logic]
            # 將預測值投影回 latent space，並與 context vector 相加 (Residual)
            pred_feedback = self.out_proj(pred)                 # [B, 1, D]
            lstm_input = attn_out + pred_feedback               # Residual connection
            # lstm_input = attn_out

        outputs = torch.cat(outputs, dim=1)                     # [B, pred_len, c_out]

        # 5. [Critical Step] RevIN De-Normalization (還原真實股價)
        if self.revin is not None:
            outputs = self.revin(outputs, 'denorm')

        return outputs
