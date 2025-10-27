# server.py
"""
FastAPI server for catalyst prediction (single-file).
- Uses a joblib artifact saved by src.train (keys expected: 'clf_agent', 'agent_le', optionally 'nn_index','X_fp_for_nn','df_for_nn','nbits')
- POST /predict accepts JSON { smiles, product_smiles, temperature_c, rxn_time_h, yield_percent }
- GET /health and GET /version endpoints provided
- Uses FastAPI lifespan event (not deprecated on_event)
"""
import os
import sys
import time
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.data import reaction_to_fp

# config through env
ARTIFACT_PATH = os.environ.get("ARTIFACT_PATH", "./models/artifacts_balanced2.joblib")
CONFIDENCE_THRESHOLD = float(os.environ.get("CONF_THRESHOLD", "0.30"))
TOP_K_ALTS = int(os.environ.get("TOP_K_ALTS", "5"))
NN_RETURN = int(os.environ.get("NN_RETURN", "5"))

app = FastAPI(title="Catalyst predictor (simple)", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change to frontend origin in prod
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# artifact container
_artifact = None
_artifact_load_time = None

class PredictRequest(BaseModel):
    smiles: Optional[str] = None
    product_smiles: Optional[str] = None
    temperature_c: Optional[float] = None
    rxn_time_h: Optional[float] = None
    yield_percent: Optional[float] = None

# helper utils
def load_artifact(path: str = ARTIFACT_PATH):
    """Load artifact if not loaded. Return artifact dict."""
    global _artifact, _artifact_load_time
    if _artifact is not None:
        return _artifact

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Artifact not found at {path}")
    _artifact = joblib.load(str(p))
    _artifact_load_time = time.time()
    # basic validation
    if "clf_agent" not in _artifact or "agent_le" not in _artifact:
        raise RuntimeError("Artifact missing required keys 'clf_agent' or 'agent_le'")
    # ensure some optional keys exist (set None if missing)
    _artifact.setdefault("nn_index", None)
    _artifact.setdefault("X_fp_for_nn", None)
    _artifact.setdefault("df_for_nn", None)
    _artifact.setdefault("nbits", 128)
    return _artifact

def softmax_from_decision_function(clf, X):
    # return (pred_idx_array, top_conf_array, alts_list_of_lists) using decision_function then stable softmax
    df = clf.decision_function(X)
    if df.ndim == 1:
        df = np.vstack([-df, df]).T
    z = df - np.max(df, axis=1, keepdims=True)
    exp = np.exp(z)
    probs = exp / exp.sum(axis=1, keepdims=True)
    top_idx = np.argmax(probs, axis=1)
    top_conf = probs[np.arange(len(probs)), top_idx]
    alts = []
    for row in probs:
        order = np.argsort(row)[::-1][:TOP_K_ALTS]
        alts.append([(int(i), float(row[i])) for i in order])
    return top_idx, top_conf, alts

# startup and shutdown handler
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # attempt to load artifact on startup but dont fail hard if missing
    try:
        load_artifact()
        print(f"Loaded artifact at startup: {ARTIFACT_PATH}", file=sys.stderr)
    except Exception as e:
        print(f"Startup: failed to load artifact ({ARTIFACT_PATH}): {e}", file=sys.stderr)
    yield
    # shutdown cleanup
    print("Shutting down server...", file=sys.stderr)

app.router.lifespan_context = lifespan

# routes
@app.get("/health")
def health():
    loaded = False
    path = ARTIFACT_PATH
    try:
        _ = load_artifact()
        loaded = True
    except Exception:
        loaded = False
    return {"status": "ok", "model_loaded": loaded, "artifact_path": path}

@app.get("/version")
def version():
    return {"service": "catalyst-predictor", "version": "1.0", "artifact_path": ARTIFACT_PATH}

@app.post("/predict")
def predict(req: PredictRequest):
    # validate smiles presence
    rxn_smi = None
    if req.smiles and str(req.smiles).strip() not in ("", "\\N"):
        rxn_smi = str(req.smiles).strip()
    elif req.product_smiles and str(req.product_smiles).strip() != "":
        rxn_smi = str(req.product_smiles).strip()
    else:
        raise HTTPException(status_code=400, detail="No reaction SMILES or product SMILES provided. Supply 'smiles' (reactants>>products) or 'product_smiles'.")

    # attempt load artifact
    try:
        art = load_artifact()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=f"Model artifact not available on server: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model artifact: {e}")

    clf = art["clf_agent"]
    le = art["agent_le"]
    nbits = int(art.get("nbits", 128))

    # featurize
    X = reaction_to_fp(rxn_smi, nBits=nbits).reshape(1, -1)

    # prediction with confidence and alts
    try:
        preds_idx, confs, alts_idx_score = softmax_from_decision_function(clf, X)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model prediction failed: {e}")

    pred_idx = int(preds_idx[0])
    conf = float(confs[0])
    pred_agent = le.classes_[pred_idx] if pred_idx < len(le.classes_) else str(pred_idx)

    # build alts with readable names
    alternatives = []
    for idx, score in alts_idx_score[0]:
        label = le.classes_[idx] if idx < len(le.classes_) else str(idx)
        alternatives.append({"agent": str(label), "score": float(score)})

    # nearest neighbor similar reactions if available
    similar = []
    try:
        if art.get("nn_index") is not None and art.get("X_fp_for_nn") is not None and art.get("df_for_nn") is not None:
            nn = art["nn_index"]
            df_nn = art["df_for_nn"]
            # guard if nn is not fitted
            if hasattr(nn, "kneighbors"):
                dists, idxs = nn.kneighbors(X, n_neighbors=min(NN_RETURN, len(art["X_fp_for_nn"])))
                for d, i in zip(dists[0], idxs[0]):
                    row = df_nn.iloc[int(i)].to_dict()
                    similar.append({
                        "id": row.get("index", None) or row.get("original_index", None) or str(i),
                        "smiles": row.get("reaction_smiles", row.get("smiles", "")),
                        "agent": row.get("agent_norm", row.get("agent", None)),
                        "solvent": row.get("solvent", None),
                        "temperature": row.get("temperature", None),
                        "distance": float(d)
                    })
    except Exception:
        similar = []

    low_confidence = conf < CONFIDENCE_THRESHOLD

    response = {
        "prediction": {
            "agent": str(pred_agent),
            "confidence": float(conf),
            "low_confidence": bool(low_confidence)
        },
        "alternatives": alternatives,
        "similar_reactions": similar
    }
    if low_confidence:
        response["note"] = "Model confidence below threshold. Check alternatives and similar reactions."

    return response

@app.get("/")
def root():
    try:
        art = load_artifact()
        loaded = True
    except Exception:
        loaded = False
    return {"service": "catalyst-predictor", "model_loaded": loaded, "artifact_path": ARTIFACT_PATH}
