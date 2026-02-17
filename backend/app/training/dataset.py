import torch
import numpy as np
import pandas as pd


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators computed from the 'close' column
    (or the first numeric column if 'close' is not found).
    """

    close_col = None
    for col in df.columns:
        if "close" in col.lower() and "adj" not in col.lower():
            close_col = col
            break
    if close_col is None:
        close_col = df.columns[0]

    close = df[close_col].values.astype(float)

    # SMA
    for window in [7, 14, 21]:
        sma = pd.Series(close).rolling(window=window, min_periods=1).mean().values
        df[f"sma_{window}"] = sma

    # EMA
    for span in [12, 26]:
        ema = pd.Series(close).ewm(span=span, adjust=False).mean().values
        df[f"ema_{span}"] = ema

    # RSI
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(window=14, min_periods=1).mean()
    avg_loss = loss.rolling(window=14, min_periods=1).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    df["rsi_14"] = rsi.values

    # MACD
    ema_12 = pd.Series(close).ewm(span=12, adjust=False).mean()
    ema_26 = pd.Series(close).ewm(span=26, adjust=False).mean()
    macd_line = ema_12 - ema_26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = macd_line - signal_line
    df["macd"] = macd_line.values
    df["macd_signal"] = signal_line.values
    df["macd_hist"] = macd_hist.values

    # Bollinger Bands
    sma_20 = pd.Series(close).rolling(window=20, min_periods=1).mean()
    std_20 = pd.Series(close).rolling(window=20, min_periods=1).std().fillna(0)
    df["bb_upper"] = (sma_20 + 2 * std_20).values
    df["bb_lower"] = (sma_20 - 2 * std_20).values
    df["bb_width"] = ((df["bb_upper"] - df["bb_lower"]) / (sma_20.values + 1e-10))

    return df


class TimeSeriesDataset(torch.utils.data.Dataset):
    """
    Sliding-window time series dataset with per-split indicator computation.

    IMPORTANT: Technical indicators are computed ONLY from the data passed to
    this dataset, preventing information leakage between train and validation
    splits. The raw OHLCV data is split first, then indicators are computed
    independently for each split.

    x: [seq_len, C]     (input window)
    y: [pred_len, C]    (target window)
    """

    def __init__(self, raw_data: np.ndarray, raw_columns: list,
                 seq_len: int, pred_len: int,
                 global_mean: np.ndarray = None,
                 global_std: np.ndarray = None):
        """
        Args:
            raw_data:     NumPy array [T, C_raw] of raw OHLCV values (NOT normalized)
            raw_columns:  List of column names matching raw_data columns
            seq_len:      Lookback window length
            pred_len:     Prediction horizon length
            global_mean:  Per-column mean for z-score normalization
            global_std:   Per-column std for z-score normalization
        """
        self.seq_len = seq_len
        self.pred_len = pred_len

        # Step 1: Build DataFrame from raw data
        df = pd.DataFrame(raw_data, columns=raw_columns)

        # Step 2: Compute technical indicators ONLY from this split's data
        df = add_technical_indicators(df)

        # Step 3: Clean NaN/inf from indicator computation
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(method="bfill").fillna(method="ffill").fillna(0)

        self.columns = df.columns.tolist()
        data = df.values.astype(np.float32)

        # Step 4: Normalize
        if global_mean is not None and global_std is not None:
            # Extend global stats if new indicator columns were added
            c_raw = len(global_mean)
            c_total = data.shape[1]

            if c_total > c_raw:
                # Compute local stats for indicator columns
                indicator_data = data[:, c_raw:]
                ind_mean = indicator_data.mean(axis=0)
                ind_std = indicator_data.std(axis=0)
                ind_std[ind_std < 1e-6] = 1.0

                self.mean = np.concatenate([global_mean, ind_mean])
                self.std = np.concatenate([global_std, ind_std])
            else:
                self.mean = global_mean
                self.std = global_std
        else:
            self.mean = data.mean(axis=0)
            self.std = data.std(axis=0)

        self.std[self.std < 1e-6] = 1.0
        self.data = (data - self.mean) / self.std

        # Safe length calculation
        self.length = max(0, len(self.data) - self.seq_len - self.pred_len + 1)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.length:
            raise IndexError("Index out of range")

        x = self.data[idx: idx + self.seq_len]
        y = self.data[idx + self.seq_len: idx + self.seq_len + self.pred_len]

        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )
