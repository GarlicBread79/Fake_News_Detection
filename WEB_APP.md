# Web application

The application has a React/Vite frontend and a FastAPI backend. The backend
serves predictions from the existing Logistic Regression and BERT classifiers.

## Requirements

- Python dependencies from `requirements.txt`
- Node.js and npm
- Valid `logistic_regression.pkl`, `tfidf_vectorizer.pkl`, and `bert_finetuned/`
  model artifacts

The checked-in Logistic Regression and vectorizer files are currently
incomplete, and the fine-tuned BERT directory is not checked in. The interface
will show a degraded service state until valid model artifacts are available.

## Run locally

From the repository root, install the project and backend requirements, then
start the API:

```bash
python -m pip install -r requirements.txt
python -m pip install -r backend/requirements.txt
python -m uvicorn app.main:app --app-dir backend --reload
```

In another terminal:

```bash
cd frontend
npm install
npm run dev
```

Open the Vite URL shown in the terminal (normally `http://localhost:5173`).
The frontend proxies `/api` requests to `http://127.0.0.1:8000`.

The API health endpoint is `http://127.0.0.1:8000/api/v1/health`. Interactive
API documentation is available at `http://127.0.0.1:8000/docs`.

Model artifact locations can be overridden with `LOGISTIC_MODEL_PATH`,
`VECTORIZER_PATH`, and `BERT_MODEL_PATH`. For a deployed frontend on another
origin, set `CORS_ORIGINS` to a comma-separated list of its trusted origins.
