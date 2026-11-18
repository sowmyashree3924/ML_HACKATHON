"""FastAPI application serving the PathSnap route-matching interface."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.routes.trips import router as trips_router
from backend.services.prediction_service import model_is_available

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_DIR = PROJECT_ROOT / "frontend"

app = FastAPI(
    title="PathSnap API",
    description="Student-test trip browser and GPS-to-road route reconstruction API.",
    version="1.0.0",
)
cors_setting = os.getenv("PATHSNAP_CORS_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000,http://localhost:5173,http://127.0.0.1:5173,http://localhost:5500,http://127.0.0.1:5500")
cors_origins = [origin.strip() for origin in cors_setting.split(",") if origin.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, bool | str]:
    """Report API and model checkpoint readiness."""
    available = model_is_available()
    return {
        "status": "healthy",
        "model_loaded": available,
        "decoder_loaded": available,
    }


@app.get("/api")
def api_info() -> dict[str, str]:
    """Return a short API description and documentation location."""
    return {"project": "PathSnap", "docs": "/docs", "interface": "/"}


app.include_router(trips_router)
if FRONTEND_DIR.is_dir():
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
