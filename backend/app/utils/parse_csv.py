import pandas as pd
import numpy as np


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators computed from the 'close' column
    (or the first numeric column if 'close' is not found).
    """

    # Find the close column
    close_col = None
    for col in df.columns:
        if "close" in col.lower() and "adj" not in col.lower():
            close_col = col
            break
    if close_col is None:
        close_col = df.columns[0]

    close = df[close_col].values.astype(float)

    # --- SMA (Simple Moving Averages) ---
    for window in [7, 14, 21]:
        sma = pd.Series(close).rolling(window=window, min_periods=1).mean().values
        df[f"sma_{window}"] = sma

    # --- EMA (Exponential Moving Averages) ---
    for span in [12, 26]:
        ema = pd.Series(close).ewm(span=span, adjust=False).mean().values
        df[f"ema_{span}"] = ema

    # --- RSI (Relative Strength Index, 14-day) ---
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(window=14, min_periods=1).mean()
    avg_loss = loss.rolling(window=14, min_periods=1).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    df["rsi_14"] = rsi.values

    # --- MACD (12/26/9) ---
    ema_12 = pd.Series(close).ewm(span=12, adjust=False).mean()
    ema_26 = pd.Series(close).ewm(span=26, adjust=False).mean()
    macd_line = ema_12 - ema_26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = macd_line - signal_line
    df["macd"] = macd_line.values
    df["macd_signal"] = signal_line.values
    df["macd_hist"] = macd_hist.values

    # --- Bollinger Bands (20-day, 2σ) ---
    sma_20 = pd.Series(close).rolling(window=20, min_periods=1).mean()
    std_20 = pd.Series(close).rolling(window=20, min_periods=1).std().fillna(0)
    df["bb_upper"] = (sma_20 + 2 * std_20).values
    df["bb_lower"] = (sma_20 - 2 * std_20).values
    df["bb_width"] = ((df["bb_upper"] - df["bb_lower"]) / (sma_20.values + 1e-10))

    return df


def parse_csv(path: str, add_indicators: bool = True) -> pd.DataFrame:
    """
    Single source of truth CSV parser.

    Args:
        path: Path to the CSV file.
        add_indicators: If True, append technical indicators.
                        Set to False when training (indicators computed per-split
                        inside TimeSeriesDataset to prevent leakage).
    """

    df = pd.read_csv(path)

    # Normalize column names
    df.columns = df.columns.str.strip().str.lower()

    # Drop date-like columns
    for col in ["date", "datetime", "time", "timestamp"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Convert to numeric
    numeric_cols = []
    for col in df.columns:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(",", "", regex=False)
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")

        if df[col].notna().sum() > 0:
            numeric_cols.append(col)

    if not numeric_cols:
        raise ValueError("No numeric columns found in CSV")

    df = df[numeric_cols]
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()

    if len(df) < 30:
        raise ValueError(
            f"Dataset too small. Need at least 30 rows, got {len(df)}"
        )

    if add_indicators:
        df = add_technical_indicators(df)
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(method="bfill").fillna(method="ffill").fillna(0)

    return df
