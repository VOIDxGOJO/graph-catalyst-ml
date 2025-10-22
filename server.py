from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import joblib
import numpy as np
from src.data import reaction_to_fp
import traceback
import math
import uvicorn

APP = FastAPI(title="Graph Catalyst ML - API")

MODEL_PATH = "models/artifacts_pruned_safe.joblib"  

class PredictRequest(BaseModel):
    smiles: str
    solvent: Optional[str] = None
    base: Optional[str] = None
    temperature: Optional[float] = None

def load_artifacts(path: str):
    data = joblib.load(path)
    return data

print("Loading model artifacts...")
ART = load_artifacts(MODEL_PATH)
CLF_AGENT = ART.get("clf_agent")
AGENT_LE = ART.get("agent_le")
NN_INDEX = ART.get("nn_index")
X_FP_ALL = ART.get("X_fp_all")
DF = ART.get("df")
CLF_SOLVENT = ART.get("clf_solvent", None)
SOLVENT_LE = ART.get("solvent_le", None)

# json-safe helpers 
def _safe_float(x):
    # return a json-safe float (None if NaN/Inf)
    try:
        if x is None:
            return None
        xf = float(x)
        return xf if math.isfinite(xf) else None
    except Exception:
        return None

def _clean_scores(arr: np.ndarray) -> np.ndarray:
    # replace NaN/Inf in probability arrays and renormalize if possible
    if arr is None:
        return None
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    s = arr.sum()
    if s > 0:
        arr = arr / s
    return arr

def _json_safe(obj):
    # recursively make an object json by cleaning floats
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float):
        return _safe_float(obj)
    if isinstance(obj, np.floating):
        return _safe_float(float(obj))
    if isinstance(obj, np.integer):
        return int(obj)
    if obj is None:
        return None
    return obj


# prediction helpers
def safe_predict_agent(fp_array: np.ndarray, topk: int = 5):
    
    # return list of {catalyst, score, loading_mol_percent} sorted by score desc
    # ensures scores are finite and json-safe
    
    if CLF_AGENT is None or AGENT_LE is None:
        return []
    try:
        proba = CLF_AGENT.predict_proba(fp_array.reshape(1, -1))[0]
        proba = _clean_scores(proba)
        idxs = np.argsort(proba)[::-1][:topk]
        out = []
        for i in idxs:
            label = AGENT_LE.inverse_transform([int(i)])[0]
            out.append({
                "catalyst": label,
                "score": _safe_float(proba[i]),
                "loading_mol_percent": None
            })
        return out
    except Exception:
        # allback to predict() only
        try:
            label_idx = int(CLF_AGENT.predict(fp_array.reshape(1, -1))[0])
            lab = AGENT_LE.inverse_transform([label_idx])[0]
            return [{"catalyst": lab, "score": _safe_float(1.0), "loading_mol_percent": None}]
        except Exception:
            return []

def nearest_similar(fp_array: np.ndarray, k: int = 5):
    
    # use NN index to find nearest examples, make sure output is json safe
    
    if NN_INDEX is None or DF is None:
        return []
    try:
        dists, inds = NN_INDEX.kneighbors(fp_array.reshape(1, -1), n_neighbors=k)
        dists = _clean_scores(dists[0])  # not probs but keep it finite; no renorm necessary
        out = []
        for j, idx in enumerate(inds[0]):
            row = DF.iloc[int(idx)]
            out.append({
                "id": str(row.get("id", idx)),
                "smiles": row.get("smiles", "") or "",
                "catalyst": row.get("agent", "") or "",
                "loading": _safe_float(row.get("yield_percent", None)),
                "distance": _safe_float(dists[j] if dists is not None else None),
            })
        return out
    except Exception:
        return []

def safe_predict_solvent(fp_array: np.ndarray):
    if CLF_SOLVENT is None or SOLVENT_LE is None:
        return None
    try:
        proba = CLF_SOLVENT.predict_proba(fp_array.reshape(1, -1))[0]
        proba = _clean_scores(proba)
        idx = int(np.argmax(proba))
        label = SOLVENT_LE.inverse_transform([idx])[0]
        return {"solvent": label, "score": _safe_float(proba[idx])}
    except Exception:
        try:
            idx = int(CLF_SOLVENT.predict(fp_array.reshape(1, -1))[0])
            label = SOLVENT_LE.inverse_transform([idx])[0]
            return {"solvent": label, "score": _safe_float(1.0)}
        except Exception:
            return None

# api
@APP.post("/predict")
def predict(req: PredictRequest):
    try:
        smiles = req.smiles or ""
        # build fingerprint matching the trained dimensionality
        nbits = int(X_FP_ALL.shape[1]) if X_FP_ALL is not None else 512
        fp = reaction_to_fp(smiles, radius=2, nBits=nbits)

        primary = safe_predict_agent(fp, topk=1)
        alternatives = safe_predict_agent(fp, topk=5)
        similar = nearest_similar(fp, k=5)
        solvent_pred = safe_predict_solvent(fp)

        resp = {
            "prediction": {
                "catalyst": primary[0]["catalyst"] if primary else None,
                "confidence": _safe_float(primary[0]["score"] if primary else 0.0),
                "loading_mol_percent": _safe_float(primary[0].get("loading_mol_percent") if primary else None),
                "protocol_note": ""
            },
            "alternatives": alternatives,
            "similar_reactions": similar,
            "solvent_prediction": solvent_pred
        }
        return _json_safe(resp)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(APP, host="127.0.0.1", port=8000)
