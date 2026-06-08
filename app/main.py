"""
app/main.py
============
FastAPI entry point cho VNSP Dashboard.

Chạy:
  uv run fastapi dev app/main.py       ← development (hot-reload)
  uv run fastapi run app/main.py       ← production

Cấu trúc API:
  /                        → serve frontend/index.html
  /static/*                → static files từ frontend/
  /api/data/*              → data management (data.py)
  /api/experiments/*       → experiment control (experiments.py)
  /api/results/*           → results & metrics (results.py)
  /api/viz/*               → visualization figures (viz.py)
  /docs                    → Swagger UI (FastAPI auto-generate)
  /redoc                   → ReDoc UI

CORS: allow_origins=["*"] cho development — restrict lại khi deploy production.
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api import data, experiments, results, viz

# ── Logger ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt = "%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "VNSP — Vietnam Stock Prediction",
    description = (
        "Replication & extension of Li et al. (Engineering Applications of AI, 2026).\n"
        "BiLSTM + Wavelet Decomposition for VCB & VIC stock price forecasting."
    ),
    version     = "1.0.0",
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)

# ── CORS ──────────────────────────────────────────────────────────────────────
# allow_origins=["*"] phù hợp cho development.
# Khi deploy production, thay bằng domain cụ thể.
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# ── Static files (frontend) ────────────────────────────────────────────────────
# Mount /static → frontend/ directory.
# Cần frontend/ tồn tại trước khi chạy server.
_FRONTEND_DIR = Path("frontend")
if _FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_FRONTEND_DIR)), name="static")
    logger.info("Static files mounted: /static → %s", _FRONTEND_DIR.resolve())
else:
    logger.warning(
        "frontend/ directory không tồn tại — /static sẽ không được mount. "
        "Tạo frontend/ trước khi chạy production."
    )

# ── API Routers ───────────────────────────────────────────────────────────────
app.include_router(data.router,        prefix="/api/data",        tags=["data"])
app.include_router(experiments.router, prefix="/api/experiments", tags=["experiments"])
app.include_router(results.router,     prefix="/api/results",     tags=["results"])
app.include_router(viz.router,         prefix="/api/viz",         tags=["visualization"])


# ── Root: serve frontend ──────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
def serve_frontend() -> FileResponse:
    """
    Serve frontend/index.html tại root URL.
    Trả về 404-friendly message nếu file không tồn tại.
    """
    index_path = _FRONTEND_DIR / "index.html"
    if not index_path.exists():
        from fastapi import HTTPException
        raise HTTPException(
            status_code = 404,
            detail      = "frontend/index.html không tồn tại. Chạy Task 8.1 trước.",
        )
    return FileResponse(str(index_path))


# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/health", tags=["system"])
def health_check() -> dict:
    """Kiểm tra server đang chạy."""
    return {"status": "ok", "service": "VNSP API", "version": "1.0.0"}