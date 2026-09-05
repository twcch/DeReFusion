import torch
from torch import nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PatchEmbedding
import timesfm

# ---------------------------------------------------------------------------
# Compatibility shim for huggingface_hub >= 0.26, which injects extra kwargs
# (e.g. ``proxies``, ``resume_download``) into the model constructor during
# ``from_pretrained``. ``TimesFM_2p5_200M_torch.__init__`` only accepts
# ``torch_compile``/``config`` and otherwise raises
# ``TypeError: ... got an unexpected keyword argument 'proxies'``.
# See https://github.com/google-research/timesfm/issues/412
# ---------------------------------------------------------------------------
_TimesFM25 = timesfm.TimesFM_2p5_200M_torch
if not getattr(_TimesFM25.__init__, "_ignores_hub_kwargs", False):
    _orig_timesfm_init = _TimesFM25.__init__

    def _timesfm_init(self, torch_compile=True, config=None, **_ignored):
        _orig_timesfm_init(self, torch_compile=torch_compile, config=config)

    _timesfm_init._ignores_hub_kwargs = True
    _TimesFM25.__init__ = _timesfm_init


class Model(nn.Module):
    def __init__(self, configs):
        """
        Load the pretrained TimesFM 2.5 (200M) PyTorch model and compile it
        with a ForecastConfig derived from configs (max_context=seq_len,
        max_horizon=pred_len).
        """
        super().__init__()

        self.model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
        self.model.compile(
            timesfm.ForecastConfig(
                max_context=configs.seq_len,
                max_horizon=configs.pred_len,
                normalize_inputs=True,
                use_continuous_quantile_head=True,
                force_flip_invariance=True,
                infer_is_positive=True,
                fix_quantile_crossing=True,
            )
        )

        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc.sub(means)
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc.div(stdev)

        B, L, C = x_enc.shape
        device = x_enc.device
        x_enc = torch.reshape(x_enc, (B*C, L))

        output, _ = self.model.forecast(
            horizon=self.pred_len,
            inputs=x_enc.cpu().numpy()
        )
        output = torch.Tensor(output).to(device)

        dec_out = torch.reshape(output, (B, output.shape[-1], C)).to(x_enc.device)
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'zero_shot_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out
        return None
