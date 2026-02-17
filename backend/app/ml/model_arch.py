import torch
import torch.nn as nn
import torch.nn.init as init


class RevIN(nn.Module):
    """
    Reversible Instance Normalization.
    Normalizes each input window per-channel, then denormalizes predictions.
    Critical for non-stationary data like stock prices.
    """

    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.mean = None
        self.std = None

    def forward(self, x, mode="norm"):
        if mode == "norm":
            # x: [B, L, C]
            self.mean = x.mean(dim=1, keepdim=True).detach()
            self.std = (x.std(dim=1, keepdim=True) + self.eps).detach()
            return (x - self.mean) / self.std
        else:
            # denorm
            return x * self.std + self.mean


class moving_avg(nn.Module):
    """Moving average block to extract trend component."""

    def __init__(self, kernel_size: int, stride: int = 1):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
        )

    def forward(self, x):
        # x: [B, L, C]
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        return x.permute(0, 2, 1)


class series_decomp(nn.Module):
    """Single-scale series decomposition: seasonal + trend."""

    def __init__(self, kernel_size: int):
        super().__init__()
        self.moving_avg = moving_avg(kernel_size)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        seasonal = x - moving_mean
        return seasonal, moving_mean


class series_decomp_multi(nn.Module):
    """
    Multi-scale series decomposition.
    Averages seasonal and trend from multiple kernel sizes
    to capture short/medium/long-term dynamics.
    """

    def __init__(self, kernel_sizes):
        super().__init__()
        self.decompositions = nn.ModuleList(
            [series_decomp(k) for k in kernel_sizes]
        )

    def forward(self, x):
        seasonal_sum = torch.zeros_like(x)
        trend_sum = torch.zeros_like(x)

        for decomp in self.decompositions:
            seasonal, trend = decomp(x)
            seasonal_sum += seasonal
            trend_sum += trend

        n = len(self.decompositions)
        return seasonal_sum / n, trend_sum / n


class Model(nn.Module):
    """
    Enhanced DLinear v2 — Controlled Architecture Upgrade.

    Improvements over v1:
      - Optional channel mixing layer (cross-feature interaction)
      - Optional nonlinear residual temporal block (GELU activation)
      - Dropout regularization
      - Multi-scale decomposition (kernel 13, 25, 49)
      - Individual linear layers per channel
      - RevIN normalization

    All new modules are optional and controlled via config flags.
    Fully backward-compatible with v1 checkpoints via strict=False loading.
    """

    def __init__(self, configs):
        super().__init__()

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in

        # Feature flags (with safe defaults for backward compat)
        self.use_channel_mixer = getattr(configs, "use_channel_mixer", False)
        self.use_residual_block = getattr(configs, "use_residual_block", False)
        dropout_rate = getattr(configs, "dropout_rate", 0.0)

        # RevIN
        self.revin = RevIN()

        # Optional: Channel Mixer — learns cross-feature interactions
        if self.use_channel_mixer:
            self.channel_mixer = nn.Linear(self.enc_in, self.enc_in)
            init.xavier_uniform_(self.channel_mixer.weight)
            nn.init.zeros_(self.channel_mixer.bias)
            self.mixer_dropout = nn.Dropout(dropout_rate)

        # Multi-scale decomposition
        self.decomposition = series_decomp_multi(kernel_sizes=[13, 25, 49])

        # Individual linear layers per channel (seasonal + trend)
        self.Linear_Seasonal = nn.ModuleList(
            [nn.Linear(self.seq_len, self.pred_len) for _ in range(self.enc_in)]
        )
        self.Linear_Trend = nn.ModuleList(
            [nn.Linear(self.seq_len, self.pred_len) for _ in range(self.enc_in)]
        )

        # Optional: Nonlinear Residual Block per channel
        if self.use_residual_block:
            self.residual_blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(self.seq_len, self.seq_len),
                    nn.GELU(),
                    nn.Linear(self.seq_len, self.pred_len),
                )
                for _ in range(self.enc_in)
            ])

    def forward(self, x):
        """
        x:      [B, seq_len, enc_in]
        output: [B, pred_len, enc_in]
        """

        # ── RevIN normalize ──
        x = self.revin(x, mode="norm")

        # ── Channel Mixer (optional) ──
        if self.use_channel_mixer:
            x = self.channel_mixer(x)          # [B, L, C] → [B, L, C]
            x = self.mixer_dropout(x)

        # ── Multi-scale decomposition ──
        seasonal, trend = self.decomposition(x)

        # ── Permute to [B, C, L] for per-channel linear projection ──
        seasonal = seasonal.permute(0, 2, 1)
        trend = trend.permute(0, 2, 1)

        # ── Per-channel linear mapping ──
        seasonal_out = torch.zeros(
            [seasonal.shape[0], self.enc_in, self.pred_len],
            dtype=seasonal.dtype,
            device=seasonal.device,
        )
        trend_out = torch.zeros_like(seasonal_out)

        for i in range(self.enc_in):
            seasonal_out[:, i, :] = self.Linear_Seasonal[i](seasonal[:, i, :])
            trend_out[:, i, :] = self.Linear_Trend[i](trend[:, i, :])

        out = seasonal_out + trend_out

        # ── Nonlinear Residual (optional) ──
        if self.use_residual_block:
            for i in range(self.enc_in):
                # Input to residual: full normalized channel [B, L]
                res_input = (seasonal[:, i, :] + trend[:, i, :])
                out[:, i, :] = out[:, i, :] + self.residual_blocks[i](res_input)

        # ── Back to [B, pred_len, C] ──
        out = out.permute(0, 2, 1)

        # ── RevIN denormalize ──
        out = self.revin(out, mode="denorm")

        return out
