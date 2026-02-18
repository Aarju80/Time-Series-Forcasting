import torch
import numpy as np
from app.ml.model_arch import Model
from app.config import CHECKPOINT_PATH

_model = None
_stats = None
_checkpoint_data = None


def clear_model_cache():
    """Call after retraining to force reload on next predict."""
    global _model, _stats, _checkpoint_data
    _model = None
    _stats = None
    _checkpoint_data = None


def load_model(num_features: int):
    """
    Load model once and cache it.
    Supports v1, v2, and v3 checkpoints.
    Uses strict=False for backward compatibility with old checkpoints.
    """
    global _model, _stats, _checkpoint_data

    if _model is None:
        _checkpoint_data = torch.load(
            CHECKPOINT_PATH,
            map_location="cpu",
            weights_only=False,
        )

        class Config:
            seq_len = _checkpoint_data.get("seq_len", 128)
            pred_len = _checkpoint_data.get("pred_len", 30)
            enc_in = _checkpoint_data.get("enc_in", num_features)
            # v2 flags (safe defaults for v1 checkpoints)
            use_channel_mixer = _checkpoint_data.get("use_channel_mixer", False)
            use_residual_block = _checkpoint_data.get("use_residual_block", False)
            dropout_rate = _checkpoint_data.get("dropout_rate", 0.0)
            # v3: skip RevIN denorm when trained on log returns
            skip_revin_denorm = _checkpoint_data.get("use_log_returns", False)

        _model = Model(Config)

        # strict=False allows loading old checkpoints into new architecture
        _model.load_state_dict(_checkpoint_data["model"], strict=False)
        _model.eval()

        _stats = (_checkpoint_data["mean"], _checkpoint_data["std"])

    return _model, _stats


def predict_future(series: np.ndarray, horizon: int):
    """
    series:  [T, C] — last SEQ_LEN rows of data (raw scale, with indicators)
    horizon: number of future steps to predict
    returns: [horizon, C] — predictions in original price scale

    Supports three modes based on checkpoint flags:
      1. v2 (default): direct multi-step prediction
      2. v3 log-returns + recursive: 1-step recursive loop
      3. v3 with bias correction + persistence blending
    """
    model, (mean, std) = load_model(series.shape[1])
    ckpt = _checkpoint_data

    # Read v3 flags with safe defaults
    use_log_returns = ckpt.get("use_log_returns", False)
    use_recursive = ckpt.get("use_recursive_forecast", False)
    bias = ckpt.get("bias", None)
    blend_weight = ckpt.get("blend_weight", 1.0)

    seq_len = ckpt.get("seq_len", 128)
    pred_len = ckpt.get("pred_len", 30)

    # Ensure correct seq_len
    if series.shape[0] > seq_len:
        series = series[-seq_len:]
    elif series.shape[0] < seq_len:
        pad_rows = seq_len - series.shape[0]
        padding = np.tile(series[0:1], (pad_rows, 1))
        series = np.concatenate([padding, series], axis=0)

    # Keep last observed price for all modes
    last_observed_price = series[-1].copy()

    # ──────────────────────────────────────
    # MODE A: v3 Recursive 1-step prediction
    # ──────────────────────────────────────
    if use_recursive and pred_len == 1:
        predictions = []
        # Work in normalized space for model input
        norm_window = ((series - mean) / std).astype(np.float32)  # [seq_len, C]
        current_last_price = last_observed_price.copy()

        for step in range(horizon):
            x = torch.tensor(norm_window, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                pred = model(x).squeeze(0).numpy()  # [1, C]

            pred_step = pred[0]  # [C]

            # Apply bias correction
            if bias is not None:
                pred_step = pred_step + bias[0]

            if use_log_returns:
                # pred_step is a log return → convert to price
                next_price = current_last_price * np.exp(pred_step)
            else:
                # pred_step is normalized price → denormalize
                next_price = pred_step * std + mean

            # Persistence blending
            if blend_weight < 1.0:
                next_price = blend_weight * next_price + (1.0 - blend_weight) * current_last_price

            predictions.append(next_price.tolist())

            # Update window: shift left and append new normalized row
            new_norm_row = ((next_price - mean) / std).astype(np.float32)
            norm_window = np.concatenate(
                [norm_window[1:], new_norm_row.reshape(1, -1)],
                axis=0,
            )
            current_last_price = next_price.copy()

        return predictions

    # ──────────────────────────────────────
    # MODE B: v2-style direct multi-step
    # ──────────────────────────────────────
    norm_series = ((series - mean) / std).astype(np.float32)
    x = torch.tensor(norm_series, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        preds = model(x).squeeze(0).numpy()  # [pred_len, C]

    preds = preds[:horizon]

    # Apply bias correction
    if bias is not None:
        bias_slice = bias[:horizon]
        preds = preds + bias_slice

    if use_log_returns:
        # Convert log returns to prices
        from app.training.dataset import reconstruct_prices_from_returns
        preds = reconstruct_prices_from_returns(preds, last_observed_price)
    else:
        # De-normalize
        preds = preds * std + mean

    # Persistence blending for direct mode
    if blend_weight < 1.0:
        for t in range(len(preds)):
            preds[t] = blend_weight * preds[t] + (1.0 - blend_weight) * last_observed_price

    return preds.tolist()
