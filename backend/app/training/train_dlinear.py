import sys
import os


def _log(msg: str):
    """Write directly to stdout — guaranteed to show in terminal."""
    sys.stdout.write(f"[TRAIN] {msg}\n")
    sys.stdout.flush()
    log_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "training.log")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"{msg}\n")


def train_model(csv_path):
    import copy
    import random
    import torch
    import numpy as np
    import torch.nn as nn
    from torch.utils.data import DataLoader

    from app.training.dataset import TimeSeriesDataset
    from app.utils.parse_csv import parse_csv
    from app.ml.model_arch import Model
    from app.config import (
        SEQ_LEN, PRED_LEN, EPOCHS, LR, BATCH_SIZE,
        PATIENCE, CHECKPOINT_PATH, CHECKPOINT_DIR,
        WEIGHT_DECAY, GRAD_CLIP_NORM, SEED,
        USE_CHANNEL_MIXER, USE_RESIDUAL_BLOCK, DROPOUT_RATE,
        USE_LOG_RETURNS, USE_RECURSIVE_FORECAST, LOSS_TYPE,
        USE_BIAS_CORRECTION, BLEND_WEIGHT,
    )

    # -------------------
    # Resolve effective pred_len
    # -------------------
    effective_pred_len = 1 if USE_RECURSIVE_FORECAST else PRED_LEN

    # -------------------
    # Reproducibility
    # -------------------
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # -------------------
    # Load raw data
    # -------------------
    _log("=" * 56)
    _log("TRAINING STARTED (v3)")
    _log("=" * 56)
    _log(f"Mode: log_returns={USE_LOG_RETURNS} | "
         f"recursive={USE_RECURSIVE_FORECAST} | "
         f"loss={LOSS_TYPE}")
    _log(f"SEQ_LEN={SEQ_LEN} | PRED_LEN={effective_pred_len} | "
         f"bias_corr={USE_BIAS_CORRECTION} | blend={BLEND_WEIGHT}")

    df = parse_csv(csv_path, add_indicators=False)
    raw_columns = df.columns.tolist()
    raw_data = df.values.astype(np.float32)

    split_idx = int(len(raw_data) * 0.8)
    train_raw = raw_data[:split_idx]
    val_raw = raw_data[split_idx:]

    global_mean = train_raw.mean(axis=0)
    global_std = train_raw.std(axis=0)
    global_std[global_std < 1e-6] = 1.0

    # -------------------
    # Datasets
    # -------------------
    train_dataset = TimeSeriesDataset(
        raw_data=train_raw,
        raw_columns=raw_columns,
        seq_len=SEQ_LEN,
        pred_len=effective_pred_len,
        global_mean=global_mean,
        global_std=global_std,
        use_log_returns=USE_LOG_RETURNS,
    )
    val_dataset = TimeSeriesDataset(
        raw_data=val_raw,
        raw_columns=raw_columns,
        seq_len=SEQ_LEN,
        pred_len=effective_pred_len,
        global_mean=global_mean,
        global_std=global_std,
        use_log_returns=USE_LOG_RETURNS,
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    if len(train_dataset) == 0:
        raise ValueError(
            f"Not enough data for training. Need at least {SEQ_LEN + effective_pred_len} rows, "
            f"got {len(train_raw)} in the training split."
        )

    enc_in = train_dataset.data.shape[1]

    _log(f"Features: {enc_in} ({len(raw_columns)} raw + "
         f"{enc_in - len(raw_columns)} indicators)")
    _log(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")

    # -------------------
    # Model
    # -------------------
    class Config:
        seq_len = SEQ_LEN
        pred_len = effective_pred_len
        use_channel_mixer = USE_CHANNEL_MIXER
        use_residual_block = USE_RESIDUAL_BLOCK
        dropout_rate = DROPOUT_RATE
        skip_revin_denorm = USE_LOG_RETURNS  # skip RevIN denorm when target is log returns

    Config.enc_in = enc_in
    model = Model(Config)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    memory_mb = (total_params * 4) / (1024 * 1024)
    _log(f"Parameters: {total_params:,} total | {trainable_params:,} trainable | "
         f"{memory_mb:.2f} MB")

    # -------------------
    # Optimizer, Scheduler, Loss
    # -------------------
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5, min_lr=1e-6,
    )

    # v3: configurable loss
    if LOSS_TYPE == "mse":
        criterion = nn.MSELoss()
        _log("Loss: MSELoss")
    else:
        criterion = nn.HuberLoss(delta=1.0)
        _log("Loss: HuberLoss(delta=1.0)")

    # -------------------
    # Training loop
    # -------------------
    best_val_loss = float("inf")
    best_model_state = None
    patience_counter = 0

    _log("-" * 56)

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            optimizer.zero_grad()
            preds = model(x)
            loss = criterion(preds, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        if len(val_loader) > 0:
            with torch.no_grad():
                for x, y in val_loader:
                    preds = model(x)
                    loss = criterion(preds, y)
                    val_loss += loss.item()
            val_loss /= len(val_loader)
        else:
            val_loss = train_loss

        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        star = " * best" if val_loss < best_val_loss else ""
        _log(
            f"Epoch [{epoch+1:3d}/{EPOCHS}]  "
            f"Train: {train_loss:.6f}  Val: {val_loss:.6f}  "
            f"LR: {current_lr:.1e}{star}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                _log(f"Early stopping at epoch {epoch+1} "
                     f"(no improvement for {PATIENCE} epochs)")
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # -------------------
    # Bias Correction (v3)
    # -------------------
    bias = None
    if USE_BIAS_CORRECTION:
        _log("Computing bias correction on training set...")
        model.eval()
        all_errors = []
        with torch.no_grad():
            for x, y in train_loader:
                preds = model(x)
                error = (y - preds).mean(dim=0).numpy()  # [pred_len, C]
                all_errors.append(error)
        # Mean bias across all training batches: [pred_len, C]
        bias = np.mean(all_errors, axis=0).astype(np.float32)
        _log(f"Bias correction computed: mean abs = {np.abs(bias).mean():.6f}")

    # -------------------
    # Save checkpoint
    # -------------------
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    checkpoint_data = {
        "model": model.state_dict(),
        "mean": train_dataset.mean,
        "std": train_dataset.std,
        "seq_len": SEQ_LEN,
        "pred_len": effective_pred_len,
        "enc_in": enc_in,
        "features": train_dataset.columns,
        "raw_columns": raw_columns,
        # v2 flags
        "use_channel_mixer": USE_CHANNEL_MIXER,
        "use_residual_block": USE_RESIDUAL_BLOCK,
        "dropout_rate": DROPOUT_RATE,
        # v3 flags
        "use_log_returns": USE_LOG_RETURNS,
        "use_recursive_forecast": USE_RECURSIVE_FORECAST,
        "blend_weight": BLEND_WEIGHT,
    }

    if bias is not None:
        checkpoint_data["bias"] = bias

    torch.save(checkpoint_data, CHECKPOINT_PATH)

    _log("-" * 56)
    _log(f"Model saved. Best val loss: {best_val_loss:.6f}")
    _log(f"Checkpoint: {CHECKPOINT_PATH}")
    _log("=" * 56)
