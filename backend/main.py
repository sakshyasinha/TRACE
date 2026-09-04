from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ml import Detector

app = FastAPI(title="TRACE API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
detector = Detector(Path(__file__).parent / "weights" / "trace.pt")

class ScanResult(BaseModel):
    label: str
    synthetic_likelihood: float
    spatial_signal: float
    frequency_signal: float
    compression_response: float
    status: str
    note: str

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "trace-api", "model_ready": str(detector.ready).lower()}

@app.post("/scan", response_model=ScanResult)
async def scan(file: UploadFile = File(...)) -> ScanResult:
    if not detector.ready:
        raise HTTPException(status_code=503, detail="Model weights are not available. Run backend/train.py first.")
    image_bytes = await file.read()
    try:
        prediction = detector.predict(image_bytes, file.filename)
    except (OSError, ValueError) as error:
        raise HTTPException(status_code=400, detail=f"Unable to analyze image: {error}") from error
    return ScanResult(
        label=prediction.label,
        synthetic_likelihood=prediction.synthetic_likelihood,
        spatial_signal=prediction.spatial_signal,
        frequency_signal=prediction.frequency_signal,
        compression_response=prediction.compression_response,
        status=prediction.status,
        note=prediction.note,
    )
