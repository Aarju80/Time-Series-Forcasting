import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from app.ml.model_arch import Model
from evaluation.metrics import mae, rmse

# ---------------- CONFIG ----------------
SEQ_LEN = 24
PRED_LEN = 30
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "dlinear.pth")
CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "samsung.csv")
# ----------------------------------------


# Load data
df = pd.read_csv(CSV_PATH)
series = df.select_dtypes(include=["number"]).iloc[:, 0]
series = series.replace([np.inf, -np.inf], np.nan).dropna().values

# Split train/test
train_size = int(len(series) * 0.8)
train_data = series[:train_size]
test_data = series[train_size:]


# Normalization (based on train only)
mean = train_data.mean()
std = train_data.std()
if std < 1e-6:
    std = 1.0

train_data = (train_data - mean) / std
test_data = (test_data - mean) / std


# Model config
class Config:
    seq_len = SEQ_LEN
    pred_len = PRED_LEN
    enc_in = 1


model = Model(Config)

# Load from checkpoint dict (matches train_dlinear.py save format)
checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
model.load_state_dict(checkpoint["model"])
model.eval()


# Backtesting (rolling window)
predictions = []
actuals = []

for i in range(0, len(test_data) - SEQ_LEN - PRED_LEN, PRED_LEN):
    x = test_data[i : i + SEQ_LEN]
    y = test_data[i + SEQ_LEN : i + SEQ_LEN + PRED_LEN]

    x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)

    with torch.no_grad():
        pred = model(x_tensor).squeeze(0).squeeze(-1).numpy()

    predictions.extend(pred)
    actuals.extend(y)


# De-normalize
predictions = np.array(predictions) * std + mean
actuals = np.array(actuals) * std + mean


# Metrics
print("MAE :", mae(actuals, predictions))
print("RMSE:", rmse(actuals, predictions))


# Plot
plt.figure(figsize=(12, 5))
plt.plot(actuals, label="Actual")
plt.plot(predictions, label="Predicted")
plt.legend()
plt.title("DLinear Backtesting")
plt.show()
