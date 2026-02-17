import pandas as pd
import numpy as np

def parse_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Convert all columns to numeric where possible
    for col in df.columns:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(",", "", regex=False)
            .replace(["null", "None", "nan", ""], np.nan)
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Keep numeric columns only
    df = df.select_dtypes(include=[np.number])

    # Drop rows only if ALL values are NaN
    df = df.dropna(how="all")

    if df.shape[1] == 0:
        raise ValueError("CSV has no numeric columns.")

    if df.shape[0] < 30:
        raise ValueError(
            f"Dataset too small after cleaning. Got {df.shape[0]} rows."
        )

    return df
