import threading
from fastapi import APIRouter, UploadFile, File, BackgroundTasks, HTTPException
import shutil
import os
import numpy as np

from app.config import DATA_PATH, CHECKPOINT_PATH, DATA_DIR, SEQ_LEN, PRED_LEN
from app.training.train_dlinear import train_model
from app.ml.inference import predict_future, clear_model_cache
from app.utils.parse_csv import parse_csv

router = APIRouter()

# Thread-safe training status
_status_lock = threading.Lock()
training_status = {"status": "idle", "message": ""}


def set_status(status: str, message: str):
    with _status_lock:
        training_status["status"] = status
        training_status["message"] = message


def get_status():
    with _status_lock:
        return dict(training_status)


@router.post("/upload")
async def upload(file: UploadFile = File(...)):
    os.makedirs(DATA_DIR, exist_ok=True)

    with open(DATA_PATH, "wb") as f:
        shutil.copyfileobj(file.file, f)

    set_status("idle", "File uploaded. Ready to train.")

    return {"message": "File uploaded successfully"}


@router.post("/train")
def train(background_tasks: BackgroundTasks):
    if not os.path.exists(DATA_PATH):
        raise HTTPException(400, "Upload a CSV file first")

    set_status("training", "Training started")

    def run():
        try:
            train_model(DATA_PATH)
            clear_model_cache()
            set_status("done", "Training complete")
        except Exception as e:
            import traceback
            traceback.print_exc()
            set_status("error", str(e))

    background_tasks.add_task(run)
    return {"message": "Training started in background"}


@router.get("/status")
def status():
    return get_status()


@router.post("/predict")
def predict(horizon: int):
    current = get_status()

    if current["status"] == "training":
        raise HTTPException(400, "Training in progress")

    if current["status"] != "done":
        raise HTTPException(400, current["message"])

    if not os.path.exists(CHECKPOINT_PATH):
        raise HTTPException(500, "Model checkpoint not found. Retrain required.")

    if horizon < 1 or horizon > PRED_LEN:
        raise HTTPException(
            400,
            f"Horizon must be between 1 and {PRED_LEN}, got {horizon}"
        )

    df = parse_csv(DATA_PATH)

    if df.shape[0] < SEQ_LEN:
        raise HTTPException(
            400,
            f"Not enough rows. Need at least {SEQ_LEN}"
        )

    series = df.iloc[-SEQ_LEN:].to_numpy()

    try:
        preds = predict_future(series, horizon)
    except Exception as e:
        raise HTTPException(500, f"Inference failed: {str(e)}")

    # Get original column names (OHLCV only, before indicators)
    original_cols = [c for c in df.columns if not _is_indicator(c)]

    return {
        "columns": original_cols,
        "all_columns": df.columns.tolist(),
        "predictions": _filter_original_cols(preds, df.columns.tolist(), original_cols)
    }


@router.post("/backtest")
def backtest():
    """
    Run backtest on the test split (last 20%) of data.
    Returns actual vs predicted for visualization + accuracy metrics.
    """
    current = get_status()

    if current["status"] != "done":
        raise HTTPException(400, "Train the model first")

    if not os.path.exists(CHECKPOINT_PATH):
        raise HTTPException(500, "Model checkpoint not found")

    df = parse_csv(DATA_PATH)
    all_cols = df.columns.tolist()
    original_cols = [c for c in all_cols if not _is_indicator(c)]
    original_indices = [all_cols.index(c) for c in original_cols]

    data = df.values.astype(np.float64)

    # Test split (last 20%)
    split_idx = int(len(data) * 0.8)
    test_data = data[split_idx:]

    if len(test_data) < SEQ_LEN + PRED_LEN:
        raise HTTPException(400, "Not enough test data for backtesting")

    # Rolling-window predictions on test set
    actuals_all = []
    preds_all = []

    step = PRED_LEN  # non-overlapping windows
    for i in range(0, len(test_data) - SEQ_LEN - PRED_LEN + 1, step):
        window = test_data[i : i + SEQ_LEN]
        actual = test_data[i + SEQ_LEN : i + SEQ_LEN + PRED_LEN]

        try:
            pred = predict_future(window, PRED_LEN)
        except Exception:
            continue

        # Extract only original columns
        actual_orig = actual[:, original_indices].tolist()
        pred_orig = [[p[idx] for idx in range(len(original_cols))]
                      for p in pred] if len(pred) > 0 else []

        actuals_all.extend(actual_orig)
        preds_all.extend(pred_orig)

    if not actuals_all:
        raise HTTPException(500, "Backtesting produced no results")

    # Compute metrics per original column
    actuals_arr = np.array(actuals_all)
    preds_arr = np.array(preds_all)

    metrics = {}
    for i, col in enumerate(original_cols):
        a = actuals_arr[:, i]
        p = preds_arr[:, i]

        mae_val = float(np.mean(np.abs(a - p)))
        rmse_val = float(np.sqrt(np.mean((a - p) ** 2)))

        # MAPE (avoid division by zero)
        non_zero = np.abs(a) > 1e-10
        if non_zero.sum() > 0:
            mape_val = float(np.mean(np.abs((a[non_zero] - p[non_zero]) / a[non_zero])) * 100)
        else:
            mape_val = 0.0

        metrics[col] = {
            "mae": round(mae_val, 4),
            "rmse": round(rmse_val, 4),
            "mape": round(mape_val, 2)
        }

    return {
        "columns": original_cols,
        "actuals": actuals_all,
        "predictions": preds_all,
        "metrics": metrics,
        "test_size": len(actuals_all)
    }


# ----- Helpers -----

INDICATOR_NAMES = [
    "sma_7", "sma_14", "sma_21",
    "ema_12", "ema_26",
    "rsi_14",
    "macd", "macd_signal", "macd_hist",
    "bb_upper", "bb_lower", "bb_width"
]


def _is_indicator(col_name: str) -> bool:
    return col_name in INDICATOR_NAMES


def _filter_original_cols(preds, all_cols, original_cols):
    """Filter prediction rows to only include original OHLCV columns."""
    indices = [all_cols.index(c) for c in original_cols]
    return [[row[i] for i in indices] for row in preds]
