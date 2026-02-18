import logging
import sys

# Configure root logger to always output to stderr (terminal)
# This ensures training logs from background threads are visible
# even when running under uvicorn --reload
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stderr,
    force=True,
)

from fastapi import FastAPI
from app.api.routes import router
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Time Series Forecaster")

# ---- CORS (important for frontend fetch) ----
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all during development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- API routes ----
app.include_router(router)


@app.get("/")
def root():
    return {"status": "Backend is running"}
