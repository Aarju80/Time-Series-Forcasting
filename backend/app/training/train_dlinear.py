def train_model(csv_path):
    import os
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
    )

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
    # Load raw data (without indicators — indicators computed per-split)
    # -------------------
    df = parse_csv(csv_path, add_indicators=False)
    raw_columns = df.columns.tolist()
    raw_data = df.values.astype(np.float32)

    # Global normalization stats from raw OHLCV columns (training split only)
    split_idx = int(len(raw_data) * 0.8)

    train_raw = raw_data[:split_idx]
    val_raw = raw_data[split_idx:]

    # Compute global stats from TRAINING data only
    global_mean = train_raw.mean(axis=0)
    global_std = train_raw.std(axis=0)
    global_std[global_std < 1e-6] = 1.0

    # -------------------
    # Create datasets (indicators computed per-split inside dataset)
    # -------------------
    train_dataset = TimeSeriesDataset(
        raw_data=train_raw,
        raw_columns=raw_columns,
        seq_len=SEQ_LEN,
        pred_len=PRED_LEN,
        global_mean=global_mean,
        global_std=global_std,
    )
    val_dataset = TimeSeriesDataset(
        raw_data=val_raw,
        raw_columns=raw_columns,
        seq_len=SEQ_LEN,
        pred_len=PRED_LEN,
        global_mean=global_mean,
        global_std=global_std,
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    if len(train_dataset) == 0:
        raise ValueError(
            f"Not enough data for training. Need at least {SEQ_LEN + PRED_LEN} rows, "
            f"got {len(train_raw)} in the training split."
        )

    # Total feature count (raw + indicators)
    enc_in = train_dataset.data.shape[1]

    print(f"Features: {enc_in} ({len(raw_columns)} raw + "
          f"{enc_in - len(raw_columns)} indicators)")
    print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")

    # -------------------
    # Model
    # -------------------
    class Config:
        seq_len = SEQ_LEN
        pred_len = PRED_LEN
        use_channel_mixer = USE_CHANNEL_MIXER
        use_residual_block = USE_RESIDUAL_BLOCK
        dropout_rate = DROPOUT_RATE

    Config.enc_in = enc_in

    model = Model(Config)

    # --- Parameter report ---
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    memory_mb = (total_params * 4) / (1024 * 1024)
    print(f"Parameters: {total_params:,} total | {trainable_params:,} trainable | "
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
    criterion = nn.HuberLoss(delta=1.0)

    # -------------------
    # Training loop with early stopping + gradient clipping
    # -------------------
    best_val_loss = float("inf")
    best_model_state = None
    patience_counter = 0

    for epoch in range(EPOCHS):
        # --- Train ---
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            optimizer.zero_grad()
            preds = model(x)
            loss = criterion(preds, y)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)

            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # --- Validate ---
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
        print(
            f"Epoch [{epoch+1}/{EPOCHS}] "
            f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
            f"LR: {current_lr:.1e}"
        )

        # --- Early stopping ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch+1} "
                      f"(no improvement for {PATIENCE} epochs)")
                break

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # -------------------
    # Save checkpoint
    # -------------------
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Save full normalization stats (raw global + indicator stats from train set)
    full_mean = train_dataset.mean
    full_std = train_dataset.std

    torch.save(
        {
            "model": model.state_dict(),
            "mean": full_mean,
            "std": full_std,
            "seq_len": SEQ_LEN,
            "pred_len": PRED_LEN,
            "enc_in": enc_in,
            "features": train_dataset.columns,
            "raw_columns": raw_columns,
            # Architecture flags (for safe reload)
            "use_channel_mixer": USE_CHANNEL_MIXER,
            "use_residual_block": USE_RESIDUAL_BLOCK,
            "dropout_rate": DROPOUT_RATE,
        },
        CHECKPOINT_PATH,
    )

    print(f"Model saved. Best val loss: {best_val_loss:.6f}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
