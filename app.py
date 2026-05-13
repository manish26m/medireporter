"""
=============================================================
MEDIREPORTER — FastAPI Application v2.0
=============================================================
Endpoints:
  GET  /              → Serve SPA
  GET  /api/health    → Model + server status
  GET  /api/version   → Pipeline version info
  POST /api/analyze   → Full pipeline inference
=============================================================
"""

import os
import io
import time
import logging
import logging.handlers

from fastapi import FastAPI, HTTPException, File, UploadFile, Form, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from models.api_pipeline import pipeline

# =============================================================
#  LOGGING SETUP
# =============================================================
log_dir = os.path.join(os.path.dirname(__file__))
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.handlers.RotatingFileHandler(
            os.path.join(log_dir, "server.log"),
            maxBytes=5 * 1024 * 1024,   # 5 MB
            backupCount=3,
            encoding="utf-8",
        ),
    ],
)
logger = logging.getLogger("medireporter.app")

# =============================================================
#  APP FACTORY
# =============================================================
app = FastAPI(
    title="MediReporter API",
    description="Deep Learning + NLP pipeline for automated clinical report summarization.",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Static files
static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# =============================================================
#  MIDDLEWARE — Request Timing
# =============================================================
@app.middleware("http")
async def timing_middleware(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    elapsed = round((time.time() - start) * 1000, 1)
    response.headers["X-Process-Time-Ms"] = str(elapsed)
    logger.info("%s %s → %d  [%.1fms]",
                request.method, request.url.path,
                response.status_code, elapsed)
    return response

# =============================================================
#  STARTUP
# =============================================================
@app.on_event("startup")
async def startup_event():
    logger.info("MediReporter v2.0 starting up…")
    pipeline.load_models()
    logger.info("Server ready.")

# =============================================================
#  ROUTES
# =============================================================
@app.get("/", include_in_schema=False)
async def serve_index():
    return FileResponse(os.path.join(static_dir, "index.html"))


@app.get("/api/health")
async def health_check():
    """Returns model status and server health."""
    status = pipeline.get_status()
    return JSONResponse(content={
        "status": "ok" if status["models_loaded"] else "loading",
        **status
    })


@app.get("/api/version")
async def version_info():
    """Returns pipeline version metadata."""
    return {
        "app_version": "2.0.0",
        "pipeline": {
            "summarizer": "facebook/bart-large-cnn",
            "ner":        "d4data/biomedical-ner-all",
            "baseline":   "LSTM Seq2Seq + Bahdanau Attention (custom trained)"
        },
        "framework": "FastAPI + PyTorch + HuggingFace Transformers"
    }


@app.post("/api/analyze")
async def analyze_report(
    text: str = Form(None),
    file: UploadFile = File(None),
    skip_lstm: str = Form("false"),
):
    """
    Main inference endpoint.
    Accepts a .txt or .pdf file (multipart) or raw text (form field).
    Returns full pipeline output including summary, entities, risk, confidence.
    """
    extracted_text = ""

    # ── PDF / TXT Extraction ───────────────────────────────
    if file and file.filename:
        fname = file.filename.lower()
        if fname.endswith(".pdf"):
            raw_bytes = await file.read()
            # Try pdfplumber first (better quality), fallback to PyPDF2
            try:
                import pdfplumber
                with pdfplumber.open(io.BytesIO(raw_bytes)) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            extracted_text += page_text + " "
            except ImportError:
                try:
                    import PyPDF2
                    pdf_reader = PyPDF2.PdfReader(io.BytesIO(raw_bytes))
                    for page in pdf_reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            extracted_text += page_text + " "
                except Exception as e:
                    raise HTTPException(
                        status_code=400,
                        detail=f"PDF parsing failed: {str(e)}"
                    )
            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"PDF parsing failed: {str(e)}"
                )

        elif fname.endswith(".txt"):
            try:
                extracted_text = (await file.read()).decode("utf-8")
            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to read TXT file: {str(e)}"
                )
        else:
            raise HTTPException(
                status_code=400,
                detail="Only .txt and .pdf files are supported."
            )

    # ── Fallback to form text ─────────────────────────────
    if text and not extracted_text:
        extracted_text = text

    extracted_text = extracted_text.strip()

    if not extracted_text or len(extracted_text) < 20:
        raise HTTPException(
            status_code=400,
            detail=(
                "Report text is too short or could not be extracted. "
                "If uploading a PDF, ensure it contains selectable text (not a scanned image). "
                "Try pasting the text manually."
            )
        )

    if len(extracted_text) > 50_000:
        extracted_text = extracted_text[:50_000]
        logger.warning("Input truncated to 50,000 chars.")

    # ── Run Pipeline ──────────────────────────────────────
    use_lstm = skip_lstm.lower() not in ("true", "1", "yes")
    try:
        logger.info("Processing report (len=%d chars, lstm=%s)…", len(extracted_text), use_lstm)
        results = pipeline.process(extracted_text, use_lstm=use_lstm)
        logger.info(
            "Pipeline complete in %.2fs | risk=%s | entities=%d",
            results["metadata"]["processing_time_s"],
            results["risk"]["level"],
            sum(len(v) for v in results["entities"].values())
        )
        return results
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.exception("Pipeline error for input len=%d", len(extracted_text))
        raise HTTPException(status_code=500, detail=f"Internal pipeline error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False, workers=1)
