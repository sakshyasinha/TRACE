# TRACE

TRACE is an explainable image forensics system for estimating whether an image is AI-generated, manipulated, or authentic. It combines spatial evidence with frequency-domain evidence, localizes suspicious regions with Grad-CAM, tests robustness under JPEG compression, and evaluates generalization to an unseen image generator.

The refined research architecture is documented in [docs/TRACE_RESEARCH_SPEC.md](docs/TRACE_RESEARCH_SPEC.md). TRACE reports measured synthetic evidence, not a literal probability of origin.

## Current milestone

The repository contains a runnable React workbench, a two-branch PyTorch model, and a FastAPI inference contract. The UI shows a clearly labeled demo result until a trained checkpoint is placed at `backend/weights/trace.pt`.

## Research split

```text
Train: Real + Generator A + Generator B
Test:  Real (held-out) + Generator C (never seen during training)
```

The planned comparison is trivial/statistical baselines, spatial-only, deterministic frequency-only, fixed fusion, and learned fusion, followed by cross-generator generalization, calibration, and perturbation robustness evaluation. Perturbations are a robustness harness, not a hand-weighted third proof signal. UI work follows trustworthy research outputs.

## Train and evaluate the model

Put images into the layout described in `backend/data/README.md`, then run:

```bash
cd backend
python train.py --data data/train --output weights/trace.pt --epochs 10
python evaluate.py data/test --train-root data/train --weights weights/trace.pt --output reports/binary_baseline
```

The scan endpoint currently recomputes the fused score after JPEG quality-70 recompression and returns the absolute score change as `compression_response`. This is an early implementation of the perturbation harness, not calibrated probability or provenance evidence.

The evaluator compares spatial-only, frequency-only, fixed 50/50 fusion, and learned fusion. It selects each threshold on a deterministic validation subset of `--train-root`, evaluates the untouched positional split, and writes `reports/binary_baseline.json`, `reports/binary_baseline_records.json`, and `reports/binary_baseline_records.csv`. The current `data/train` and `data/test` folders report `real`/`FAKE` sources only; they do not establish an unseen-generator result.

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

TRACE v1 is image-only. It does not analyze text, video, audio, EXIF metadata, camera fingerprints, or provide legal or absolute proof of authenticity. Scores are model evidence estimates and may fail on unseen generation pipelines, compression, resizing, and post-processing. Calibration, Grad-CAM, frequency-band explanations, learned fusion, and a true generator-held-out evaluation remain research milestones rather than claims of the current smoke-test checkpoint.
