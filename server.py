from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import joblib
import numpy as np
from pathlib import Path
import pandas as pd

from src.data import reaction_to_fp

ARTIFACT_PATH = Path("models/artifacts.joblib")

app = FastAPI(title="Graph Catalyst ML (ORDerly) - lightweight API")

AR = None  # will hold artifacts dict

@app.on_event("startup")
def load_models():
    global AR
    if not ARTIFACT_PATH.exists():
        raise RuntimeError(f"Artifacts not found at {ARTIFACT_PATH}. Train models first with src/train.py")
    AR = joblib.load(str(ARTIFACT_PATH))
    print("Loaded artifacts. Models ready.")
    required = ['clf_agent', 'agent_le', 'nn_index', 'X_fp_all', 'df']
    for k in required:
        if k not in AR:
            raise RuntimeError(f"Artifact missing required key: {k}")


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
    if payload.product_smiles:
        return f">>{payload.product_smiles}"
    return ""


@app.post("/predict")
def predict(payload: PredictRequest):
    """
    json form:
    {
      "prediction": { "agent": "...", "solvent": "...", "confidence": 0.82, "protocol_note": "..." },
      "alternatives": [ { "agent": "...", "solvent": "...", "score": 0.6 }, ... ],
      "similar_reactions": [ { "id": "...", "smiles": "...", "agent": "...", "solvent": "...", "procedure_details": "..." }, ... ]
    }
    """
    global AR
    if AR is None:
        raise HTTPException(status_code=500, detail="Model artifacts not loaded")

    rxn = _build_reaction_smiles(payload)
    try:
        nBits = AR['X_fp_all'].shape[1]
        fp = reaction_to_fp(rxn, nBits=nBits)
        X = fp.reshape(1, -1)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid SMILES or fingerprint error: {e}")

    # agent prediction
    clf_agent = AR.get('clf_agent')
    agent_le = AR.get('agent_le')
    agent_pred = None
    agent_conf = None
    alternatives = []
    if clf_agent is not None:
        try:
            probs = clf_agent.predict_proba(X)[0]
            class_idxs = clf_agent.classes_
            # get top sorted indices 
            order = np.argsort(probs)[::-1]
            # top class integer index
            top_class_int = class_idxs[order[0]]
            agent_label = agent_le.inverse_transform([top_class_int])[0]
            agent_pred = None if agent_label == '<<NULL>>' else agent_label
            agent_conf = float(probs[order[0]])
            # alts top5
            alt_list = []
            for pos in order[:10]:
                ci = class_idxs[pos]
                label = agent_le.inverse_transform([ci])[0]
                if label == '<<NULL>>':
                    continue
                alt_list.append({"agent": label, "score": float(probs[pos])})
                if len(alt_list) >= 5:
                    break
            alternatives = alt_list
        except Exception:
            agent_pred = None

    # solvent prediction if available
    clf_solvent = AR.get('clf_solvent')
    sol_pred = None
    if clf_solvent is not None:
        try:
            probs_s = clf_solvent.predict_proba(X)[0]
            class_idxs_s = clf_solvent.classes_
            top_pos = np.argsort(probs_s)[::-1][0]
            top_int = class_idxs_s[top_pos]
            sol_label = AR['solvent_le'].inverse_transform([top_int])[0]
            sol_pred = None if sol_label == '<<NULL>>' else sol_label
        except Exception:
            sol_pred = None

    # similar reactions via NN
    similar = []
    try:
        dists, idxs = AR['nn_index'].kneighbors(X, n_neighbors=6, return_distance=True)
        for dist, i in zip(dists[0], idxs[0]):
            row = AR['df'].iloc[int(i)]
            similar.append({
                "id": str(row['id']),
                "smiles": row['smiles'],
                "agent": row['agent'] if row['agent'] is not None else None,
                "solvent": row['solvent'] if row['solvent'] is not None else None,
                "procedure_details": row['procedure_details'] if 'procedure_details' in row else None,
                "temperature": float(row['temperature']) if pd.notna(row['temperature']) and row['temperature'] not in (None, '') else None,
                "rxn_time": float(row['rxn_time']) if pd.notna(row['rxn_time']) and row['rxn_time'] not in (None, '') else None,
                "yield": float(row['yield_percent']) if pd.notna(row['yield_percent']) and row['yield_percent'] not in (None, '') else None,
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
