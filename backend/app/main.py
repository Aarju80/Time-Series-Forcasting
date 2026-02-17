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
