# Time Series Forecaster

A full-stack time series forecasting app powered by **DLinear** (PyTorch) with a **FastAPI** backend and a **Tailwind CSS + Chart.js** frontend.

Upload any CSV with numeric columns, train a model, and visualize multi-step predictions.

## Quick Start

### Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Backend runs at `http://127.0.0.1:8000`.

### Frontend

```bash
cd frontend
npm install
npx tailwindcss -i ./style.css -o ./output.css --watch
```

Open `frontend/index.html` in a browser.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/upload` | Upload a CSV file (multipart form) |
| `POST` | `/train` | Start model training (background) |
| `GET`  | `/status` | Poll training status |
| `POST` | `/predict?horizon=N` | Get N-step predictions (1–30) |

## Tech Stack

- **ML Model**: DLinear — trend/seasonal decomposition with shared linear layers
- **Backend**: FastAPI, PyTorch, Pandas, NumPy
- **Frontend**: HTML, Tailwind CSS v3, Chart.js
- **Hyperparameters**: seq_len=24, pred_len=30, 20 epochs, MSE loss

## Sample Data

- `samsung.csv` — Samsung stock prices
- `weather.csv` — Weather time series
