from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import joblib
import numpy as np
from pathlib import Path

from src.data import reaction_to_fp  # reuse same FP routine

ARTIFACT_PATH = Path("models/artifacts.joblib")

app = FastAPI(title="Graph Catalyst ML (ORDerly) - lightweight API")

# load artifacts at startup
@app.on_event("startup")
def load_models():
    global AR
    if not ARTIFACT_PATH.exists():
        raise RuntimeError(f"Artifacts not found at {ARTIFACT_PATH}. Train models first with src/train.py")
    AR = joblib.load(str(ARTIFACT_PATH))
    # AR has- clf_agent, clf_solvent, agent_le, solvent_le, nn_index, X_fp_all, df
    print("Loaded artifacts. Models ready.")

class PredictRequest(BaseModel):
    smiles: Optional[str] = None
    product_smiles: Optional[str] = None
    solvent: Optional[str] = None
    temperature_c: Optional[float] = None
    rxn_time_h: Optional[float] = None
    yield_percent: Optional[float] = None

def _build_reaction_smiles(payload: PredictRequest) -> str:
    if payload.smiles:
        return payload.smiles
    # fallback
    left = ""
    return left + (f">>{payload.product_smiles}" if payload.product_smiles else "")

@app.post("/predict")
def predict(payload: PredictRequest):
    """
    Returns JSON of this form (conservative):
    {
      "prediction": { "agent": "...", "solvent": "...", "confidence": 0.82, "protocol_note": "..." },
      "alternatives": [ { "agent": "...", "solvent": "...", "score": 0.6 }, ... ],
      "similar_reactions": [ { "id": "...", "smiles": "...", "agent": "...", "solvent": "...", "procedure_details": "..." }, ... ]
    }
    """
    try:
        rxn = _build_reaction_smiles(payload)
        fp = reaction_to_fp(rxn, radius=AR['agent_le'].classes_.shape[0] if False else 2, nBits=AR['X_fp_all'].shape[1])
        # ensure correct shape
        if fp.ndim == 1:
            X = fp.reshape(1, -1)
        else:
            X = np.array(fp).reshape(1, -1)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid SMILES or fingerprint error: {e}")

    # agent prediction
    clf_agent = AR.get('clf_agent')
    agent_le = AR.get('agent_le')
    agent_pred = None
    agent_conf = None
    alternatives = []
    if clf_agent is not None:
        probs = clf_agent.predict_proba(X)[0]
        top_idx = np.argsort(probs)[::-1]  # descending
        top0 = top_idx[0]
        agent_pred_label = agent_le.inverse_transform([top0])[0]
        agent_pred = None if agent_pred_label == '<<NULL>>' else agent_pred_label
        agent_conf = float(probs[top0])
        # alt top-5 excluding NULL
        alt_list = []
        for tid in top_idx[:10]:
            label = agent_le.inverse_transform([tid])[0]
            if label == '<<NULL>>':
                continue
            alt_list.append({"agent": label, "score": float(probs[tid])})
            if len(alt_list) >= 5:
                break
        alternatives = alt_list

    # solvent prediction
    clf_solvent = AR.get('clf_solvent')
    sol_pred = None
    if clf_solvent is not None:
        try:
            probs_s = clf_solvent.predict_proba(X)[0]
            topi = np.argsort(probs_s)[::-1][0]
            sol_label = AR['solvent_le'].inverse_transform([topi])[0]
            sol_pred = None if sol_label == '<<NULL>>' else sol_label
        except Exception:
            sol_pred = None

    # similar reactions via NN (exclude exact self-match if found)
    similar = []
    try:
        # AR['nn_index'] fitted on AR['X_fp_all']
        dists, idxs = AR['nn_index'].kneighbors(X, n_neighbors=6, return_distance=True)
        dlist = dists[0].tolist()
        ilist = idxs[0].tolist()
        # iterate skipping first if distance==0 (itself)
        for dist, i in zip(dlist, ilist):
            row = AR['df'].iloc[i]
            similar.append({
                "id": str(row['id']),
                "smiles": row['smiles'],
                "agent": row['agent'] if row['agent'] is not None else None,
                "solvent": row['solvent'] if row['solvent'] is not None else None,
                "procedure_details": row['procedure_details'],
                "temperature": float(row['temperature']) if (row['temperature'] not in (None, 'None', '') and pd_notna(row['temperature'])) else None,
                "rxn_time": float(row['rxn_time']) if (row['rxn_time'] not in (None, 'None', '') and pd_notna(row['rxn_time'])) else None,
                "yield": float(row['yield_percent']) if (row['yield_percent'] not in (None, 'None', '') and pd_notna(row['yield_percent'])) else None,
                "distance": float(dist)
            })
    except Exception:
        similar = []

    result = {
        "prediction": {
            "agent": agent_pred,
            "solvent": sol_pred,
            "confidence": agent_conf,
            "protocol_note": similar[0]['procedure_details'] if similar and similar[0].get('procedure_details') else None
        },
        "alternatives": alternatives,
        "similar_reactions": similar
    }
    return result

def pd_notna(x):
    return x is not None and str(x) != 'nan'
