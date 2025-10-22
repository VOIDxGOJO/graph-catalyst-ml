from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import joblib
import numpy as np
from src.data import reaction_to_fp
import traceback
import math
import uvicorn

from fastapi.middleware.cors import CORSMiddleware

APP = FastAPI(title="Catalyst ML - API (frontend-compatible)")

APP.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],               
    allow_credentials=False,          
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    try:
        if x is None:
            return None
        xf = float(x)
        return xf if math.isfinite(xf) else None
    except Exception:
        return None

def _clean_scores(arr: np.ndarray) -> np.ndarray:
    if arr is None:
        return None
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    s = arr.sum()
    if s > 0:
        arr = arr / s
    return arr

def _json_safe(obj):
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
def safe_predict_agent(fp_array: np.ndarray, topk: int = 5, solvent_hint: Optional[str] = None):
    """
    Return list of dicts with keys: catalyst, agent (alias), score, loading_mol_percent, solvent
    """
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
                "agent": label,  # alias for frontend
                "score": _safe_float(proba[i]),
                "loading_mol_percent": None,
                "solvent": solvent_hint
            })
        return out
    except Exception:
        try:
            label_idx = int(CLF_AGENT.predict(fp_array.reshape(1, -1))[0])
            lab = AGENT_LE.inverse_transform([label_idx])[0]
            return [{
                "catalyst": lab,
                "agent": lab,
                "score": _safe_float(1.0),
                "loading_mol_percent": None,
                "solvent": solvent_hint
            }]
        except Exception:
            return []

def nearest_similar(fp_array: np.ndarray, k: int = 5):
    """
    Return list of similar reactions with keys matching frontend: id, smiles, agent, solvent, temperature, yield
    """
    if NN_INDEX is None or DF is None:
        return []
    try:
        dists, inds = NN_INDEX.kneighbors(fp_array.reshape(1, -1), n_neighbors=k)
        # make distances finite
        dists = np.nan_to_num(dists, nan=0.0, posinf=0.0, neginf=0.0)[0]
        out = []
        for j, idx in enumerate(inds[0]):
            row = DF.iloc[int(idx)]
            # attempt to pull fields that may exist in DF; tolerate missing names
            agent_val = row.get("agent") if "agent" in row.index else row.get("catalyst") if "catalyst" in row.index else row.get("agent_000", None)
            solvent_val = row.get("solvent") if "solvent" in row.index else row.get("solvent_000", None) or (row.get("solvent_prediction") if "solvent_prediction" in row.index else None)
            temperature_val = row.get("temperature") if "temperature" in row.index else row.get("temperature_c", None) or row.get("temp", None)
            yield_val = row.get("yield_percent") if "yield_percent" in row.index else row.get("yield_000", None) or row.get("yield", None)
            out.append({
                "id": str(row.get("id", idx)),
                "smiles": row.get("smiles", "") or row.get("rxn_str", "") or "",
                "agent": agent_val or "",
                "catalyst": agent_val or "",  # alias
                "solvent": solvent_val or "",
                "temperature": _safe_float(temperature_val),
                "yield": _safe_float(yield_val),
                "distance": _safe_float(dists[j] if j < len(dists) else None)
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

# API endpoint
@APP.post("/predict")
def predict(req: PredictRequest):
    try:
        smiles = (req.smiles or "").strip()
        nbits = int(X_FP_ALL.shape[1]) if X_FP_ALL is not None else 512
        fp = reaction_to_fp(smiles, radius=2, nBits=nbits)

        # get solvent hint from solvent predictor (so we can populate alternatives' solvent field)
        solvent_pred = safe_predict_solvent(fp)
        solvent_hint = solvent_pred.get("solvent") if solvent_pred else None

        primary_list = safe_predict_agent(fp, topk=1, solvent_hint=solvent_hint)
        alternatives = safe_predict_agent(fp, topk=5, solvent_hint=solvent_hint)
        similar = nearest_similar(fp, k=5)

        primary = primary_list[0] if primary_list else None

        resp = {
            # Keep old names too (catalyst) but expose frontend-friendly names (agent/solvent)
            "prediction": {
                "catalyst": primary.get("catalyst") if primary else None,
                "agent": primary.get("agent") if primary else None,
                "solvent": solvent_hint,
                "confidence": _safe_float(primary.get("score") if primary else 0.0),
                "loading_mol_percent": _safe_float(primary.get("loading_mol_percent") if primary else None),
                "protocol_note": ""
            },
            # alternatives now include agent + solvent keys for frontend
            "alternatives": alternatives,
            # similar reactions use agent, solvent, yield, temperature keys
            "similar_reactions": similar,
            # also provide explicit solvent_prediction object (for advanced UI use)
            "solvent_prediction": solvent_pred
        }
        return _json_safe(resp)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(APP, host="127.0.0.1", port=8000)
