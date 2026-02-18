import torch
import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════
# Utility: Log Returns
# ═══════════════════════════════════════════════════════════

def compute_log_returns(data: np.ndarray) -> np.ndarray:
    """
    Convert price array to log returns per channel.

    data:    [T, C] raw prices
    returns: [T-1, C] log returns  (log(p[t] / p[t-1]))

    Numerically stable: clamps ratios to avoid log(0) or log(negative).
    """
    # Ensure prices are positive
    data_safe = np.maximum(data, 1e-8)
    ratios = data_safe[1:] / data_safe[:-1]
    # Clamp ratios to [0.5, 2.0] to prevent extreme log returns
    ratios = np.clip(ratios, 0.5, 2.0)
    log_ret = np.log(ratios).astype(np.float32)
    # Final NaN/inf guard
    log_ret = np.nan_to_num(log_ret, nan=0.0, posinf=0.0, neginf=0.0)
    return log_ret


def reconstruct_prices_from_returns(
    log_returns: np.ndarray,
    last_prices: np.ndarray,
) -> np.ndarray:
    """
    Convert log-return predictions back to prices.

    log_returns: [H, C] predicted log returns
    last_prices: [C]    last observed price per channel
    returns:     [H, C] reconstructed prices
    """
    prices = np.zeros_like(log_returns)
    prev = last_prices.copy()
    for t in range(len(log_returns)):
        # Clamp returns to prevent extreme price jumps
        clipped = np.clip(log_returns[t], -0.5, 0.5)
        prices[t] = prev * np.exp(clipped)
        prev = prices[t]
    return prices


# ═══════════════════════════════════════════════════════════
# Technical Indicators
# ═══════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════

class TimeSeriesDataset(torch.utils.data.Dataset):
    """
    Sliding-window time series dataset.

    Supports two target modes:
      - Raw prices (v2 default):  y = normalized price window
      - Log returns (v3):         y = log(p[t+1]/p[t]) per channel

    Technical indicators are computed ONLY from the data passed to
    this dataset, preventing information leakage between splits.

    x: [seq_len, C]     (input window, always normalized prices)
    y: [pred_len, C]    (target: normalized prices OR log returns)
    """

    def __init__(self, raw_data: np.ndarray, raw_columns: list,
                 seq_len: int, pred_len: int,
                 global_mean: np.ndarray = None,
                 global_std: np.ndarray = None,
                 use_log_returns: bool = False):
        """
        Args:
            raw_data:        NumPy array [T, C_raw] of raw OHLCV values (NOT normalized)
            raw_columns:     List of column names matching raw_data columns
            seq_len:         Lookback window length
            pred_len:        Prediction horizon length
            global_mean:     Per-column mean for z-score normalization
            global_std:      Per-column std for z-score normalization
            use_log_returns: If True, targets are log returns instead of raw prices
        """
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.use_log_returns = use_log_returns

        # Step 1: Build DataFrame from raw data
        df = pd.DataFrame(raw_data, columns=raw_columns)

        # Step 2: Compute technical indicators ONLY from this split's data
        df = add_technical_indicators(df)

        # Step 3: Clean NaN/inf from indicator computation
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(method="bfill").fillna(method="ffill").fillna(0)

        self.columns = df.columns.tolist()
        data = df.values.astype(np.float32)

        # Store raw (un-normalized) prices for log-return computation
        self.raw_prices = data.copy()

        # Step 4: Normalize
        if global_mean is not None and global_std is not None:
            c_raw = len(global_mean)
            c_total = data.shape[1]

            if c_total > c_raw:
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

        # Pre-compute log returns from raw prices if needed
        if self.use_log_returns:
            self.log_returns = compute_log_returns(self.raw_prices)

        # Safe length calculation
        self.length = max(0, len(self.data) - self.seq_len - self.pred_len + 1)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.length:
            raise IndexError("Index out of range")

        x = self.data[idx: idx + self.seq_len]

        if self.use_log_returns:
            # log_returns[i] = log(price[i+1] / price[i])
            # For prediction window starting at seq_len, we need returns:
            #   from price[idx+seq_len-1] → price[idx+seq_len]
            #   from price[idx+seq_len]   → price[idx+seq_len+1]
            #   etc.
            # log_returns index = price index (for the "from" price)
            # So log_returns[idx+seq_len-1] through log_returns[idx+seq_len+pred_len-2]
            ret_start = idx + self.seq_len - 1
            ret_end = ret_start + self.pred_len
            ret_end = min(ret_end, len(self.log_returns))
            y = self.log_returns[ret_start:ret_end]

            # Pad if needed
            if len(y) < self.pred_len:
                pad = np.zeros((self.pred_len - len(y), y.shape[1]), dtype=np.float32)
                y = np.concatenate([y, pad], axis=0)
        else:
            y = self.data[idx + self.seq_len: idx + self.seq_len + self.pred_len]

        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )
