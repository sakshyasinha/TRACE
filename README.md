# TRACE

TRACE is an explainable image forensics system for estimating whether an image is AI-generated, manipulated, or authentic. It combines spatial evidence with frequency-domain evidence, localizes suspicious regions with Grad-CAM, tests robustness under JPEG compression, and evaluates generalization to an unseen image generator.

## Current milestone

The repository contains a runnable React workbench, a two-branch PyTorch model, and a FastAPI inference contract. The UI shows a clearly labeled demo result until a trained checkpoint is placed at `backend/weights/trace.pt`.

## Research split

```text
Train: Real + Generator A + Generator B
Test:  Real (held-out) + Generator C (never seen during training)
```

The planned comparison is spatial-only vs frequency-only vs fused, followed by unseen-generator and JPEG robustness evaluation.

## Train and evaluate the model

Put images into the layout described in `backend/data/README.md`, then run:

```bash
cd backend
python train.py --data data/train --output weights/trace.pt --epochs 10
python evaluate.py test_in_distribution --weights weights/trace.pt
python evaluate.py test_unseen --weights weights/trace.pt
```

The scan endpoint also recomputes the fused likelihood after JPEG quality-70 recompression and returns the absolute confidence change as `compression_response`.

## Run the frontend

```bash
cd frontend
npm install
npm run dev
```

## Run the API

```bash
cd backend
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
uvicorn main:app --reload
```

## Scope and limitations

TRACE v1 is image-only. It does not analyze text, video, audio, EXIF metadata, camera fingerprints, or provide legal or absolute proof of authenticity. A prediction is a likelihood based on learned forensic evidence and should be interpreted with appropriate uncertainty.
