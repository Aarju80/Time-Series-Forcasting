import torch
import numpy as np
from app.ml.model_arch import Model
from app.config import CHECKPOINT_PATH

_model = None
_stats = None


def clear_model_cache():
    """Call after retraining to force reload on next predict."""
    global _model, _stats
    _model = None
    _stats = None


def load_model(num_features: int):
    """
    Load model once and cache it.
    Supports both v1 (no channel mixer/residual) and v2 checkpoints.
    Uses strict=False for backward compatibility with old checkpoints.
    """
    global _model, _stats

    if _model is None:
        checkpoint = torch.load(
            CHECKPOINT_PATH,
            map_location="cpu",
            weights_only=False,
        )

        class Config:
            seq_len = checkpoint.get("seq_len", 128)
            pred_len = checkpoint.get("pred_len", 30)
            enc_in = checkpoint.get("enc_in", num_features)
            # v2 architecture flags (safe defaults for v1 checkpoints)
            use_channel_mixer = checkpoint.get("use_channel_mixer", False)
            use_residual_block = checkpoint.get("use_residual_block", False)
            dropout_rate = checkpoint.get("dropout_rate", 0.0)

        _model = Model(Config)

        # strict=False allows loading v1 checkpoints into v2 architecture
        # (missing keys for new modules are left at their init values)
        _model.load_state_dict(checkpoint["model"], strict=False)
        _model.eval()

        _stats = (checkpoint["mean"], checkpoint["std"])

    return _model, _stats


def predict_future(series: np.ndarray, horizon: int):
    """
    series: [T, C]  — last SEQ_LEN rows of data (raw scale)
    returns: [horizon, C] — predictions in original scale
    """
    model, (mean, std) = load_model(series.shape[1])

    # Normalize
    series = (series - mean) / std

    x = torch.tensor(series, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        preds = model(x).squeeze(0).numpy()

    preds = preds[:horizon]

    # De-normalize
    preds = preds * std + mean

    return preds.tolist()
