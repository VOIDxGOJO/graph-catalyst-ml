"""
artifact loading 
label decoding when clf.classes_ can be ints or strings
json safe outputs (no NaN/Inf, no numpy scalars)
src.data.reaction_to_fp for featurization (rdkit)
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import joblib
import numpy as np
import math
import traceback
import sys
from pathlib import Path
import os

proj_root = Path(__file__).resolve().parent
if str(proj_root) not in sys.path:
    sys.path.insert(0, str(proj_root))

from src.data import reaction_to_fp


from rdkit import Chem
from rdkit.Chem import Draw

import base64
from io import BytesIO
from fastapi.middleware.cors import CORSMiddleware

APP = FastAPI(title="Catalyst/Agent ML API (robust)")

APP.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ARTIFACT_PATH = os.environ.get("CATALYST_ARTIFACT", "models/artifacts_partial.joblib")

class PredictRequest(BaseModel):
    smiles: Optional[str] = None
    product_smiles: Optional[str] = None
    solvent: Optional[str] = None
    temperature_c: Optional[float] = None
    rxn_time_h: Optional[float] = None
    yield_percent: Optional[float] = None

# json safety helpers 
def _is_finite_number(x):
    try:
        if x is None:
            return False
        if isinstance(x, (np.floating, float)):
            return math.isfinite(float(x))
        if isinstance(x, (np.integer, int)):
            return True
        return False
    except Exception:
        return False

def _safe_float(v):
    try:
        if v is None:
            return None
        # numpy scalar, then native python
        if isinstance(v, (np.floating, np.integer)):
            v = v.item()
        f = float(v)
        if not math.isfinite(f):
            return None
        return f
    except Exception:
        return None

def _json_safe(obj):
    # recursively convert numpy scalars/arrays and NaN/Inf to json serializable py primitives
    if obj is None:
        return None
    if isinstance(obj, (str, bool, int)):
        return obj
    if isinstance(obj, (np.floating, float)):
        return _safe_float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            out[str(k)] = _json_safe(v)
        return out
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, np.ndarray):
        try:
            lst = obj.tolist()
            return _json_safe(lst)
        except Exception:
            return None
    # fallback try to stringify
    try:
        return str(obj)
    except Exception:
        return None

# RDKit helper
def canonical_smiles(smiles: Optional[str]) -> Optional[str]:
    if not smiles:
        return None
    try:
        m = Chem.MolFromSmiles(smiles)
        if m is None:
            return smiles
        return Chem.MolToSmiles(m, canonical=True)
    except Exception:
        return smiles

def smiles_to_png_base64(smiles: Optional[str], size=(220, 160)) -> Optional[str]:
    if not smiles:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        img = Draw.MolToImage(mol, size=size)
        buf = BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception:
        return None

ART = {}
CLF_AGENT = None
AGENT_LE = None
CLF_SOLVENT = None
SOLVENT_LE = None
NN_INDEX = None
X_FP_ALL = None
DF = None

def load_artifacts(path: str):
    global ART, CLF_AGENT, AGENT_LE, CLF_SOLVENT, SOLVENT_LE, NN_INDEX, X_FP_ALL, DF
    print(f"🔹 Loading artifacts from: {path}")
    ART = joblib.load(path)
    # defensive key fetches
    CLF_AGENT = ART.get("clf_agent") or ART.get("clf") or ART.get("clf_agent_x")
    AGENT_LE = ART.get("agent_le") or ART.get("agent_encoder") or ART.get("agent_labels")
    CLF_SOLVENT = ART.get("clf_solvent")
    SOLVENT_LE = ART.get("solvent_le")
    NN_INDEX = ART.get("nn_index")

    # fingerprints to avoid direct truth checks on arrays
    if "X_fp_all" in ART:
        X_FP_ALL = ART["X_fp_all"]
    elif "X_fp_for_nn" in ART:
        X_FP_ALL = ART["X_fp_for_nn"]
    elif "train_fp" in ART:
        X_FP_ALL = ART["train_fp"]
    else:
        X_FP_ALL = None

    # df retrieval for nearest examples
    if "df" in ART:
        DF = ART["df"]
    elif "df_for_nn" in ART:
        DF = ART["df_for_nn"]
    elif "train_df" in ART:
        DF = ART["train_df"]
    else:
        DF = None

    # log result
    keys = list(ART.keys())
    print("  Keys in artifact:", keys)
    if X_FP_ALL is not None:
        try:
            print("  Using X_fp_all (shape):", np.asarray(X_FP_ALL).shape)
        except Exception:
            pass
    if DF is not None:
        try:
            print("  Using DF (len):", len(DF))
        except Exception:
            pass
    print(" Artifacts loaded SAFEE.")

# load artifacts at import if available
if Path(ARTIFACT_PATH).exists():
    try:
        load_artifacts(ARTIFACT_PATH)
    except Exception as e:
        print(" FAILEDDD to load artifacts:", e)
        traceback.print_exc()
else:
    print(f"Artifact path {ARTIFACT_PATH} not found. Start server after training and writing artifact.")


# MAPPING
def _decode_agent_label_from_clf_class(clf, index_pos, agent_le):
    """
    map classifier.classes_[index_pos] to the final human-readable label string.
    handle cases where clf.classes_ contains ints, numpy ints, or string labels.
    """
    try:
        clf_classes = getattr(clf, "classes_", None)
        if clf_classes is None:
            return None
        enc = clf_classes[int(index_pos)]
        # case 1 enc is numeric index (int or numpy int) then inverse_transform via agent_le
        if isinstance(enc, (int, np.integer)):
            # if agent_le exists and has inverse_transform, use it
            if agent_le is not None:
                try:
                    return agent_le.inverse_transform([int(enc)])[0]
                except Exception:
                    # inverse transform failed (maybe label encoding was different)
                    return str(enc)
            else:
                return str(enc)
        # case 2 enc is string that may equal an encoded label in agent_le.classes_
        if isinstance(enc, (str,)):
            # if agent_le exists and its classes_ contains this enc, return enc directly (most likely)
            try:
                if agent_le is not None and enc in list(agent_le.classes_):
                    return enc
            except Exception:
                pass
            # else try to inverse_transform enc in case agent_le expects that
            try:
                if agent_le is not None:
                    return agent_le.inverse_transform([enc])[0]
            except Exception:
                pass
            return enc
        
        return str(enc)
    except Exception:
        return None

def predict_agent_from_fp(fp_array: np.ndarray, topk: int = 5, solvent_hint: Optional[str] = None):
    if CLF_AGENT is None or AGENT_LE is None:
        return []
    try:
        X = fp_array.reshape(1, -1)
        # probability path
        if hasattr(CLF_AGENT, "predict_proba"):
            probs = CLF_AGENT.predict_proba(X)[0]
            probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
            # handle non finite
            probs = np.array([0.0 if not np.isfinite(x) else x for x in probs], dtype=float)
            idxs = np.argsort(probs)[::-1][:topk]
            out = []
            for i in idxs:
                score = float(probs[i]) if np.isfinite(probs[i]) else 0.0
                label = _decode_agent_label_from_clf_class(CLF_AGENT, i, AGENT_LE)
                if label is None:
                    # lastresort- try to use AGENT_LE.classes_ if numeric mapping matches i
                    try:
                        label = AGENT_LE.inverse_transform([int(i)])[0]
                    except Exception:
                        label = str(i)
                out.append({
                    "agent": label,
                    "score": _safe_float(score),
                    "loading_mol_percent": None,
                    "solvent": solvent_hint
                })
            return out
        # non-proba path
        else:
            p = CLF_AGENT.predict(X)[0]
            # p may be encoded label (int or str)
            try:
                if isinstance(p, (int, np.integer)):
                    label = AGENT_LE.inverse_transform([int(p)])[0]
                else:
                    if isinstance(p, str):
                        # prefer to return p if known label
                        try:
                            if AGENT_LE is not None and p in list(AGENT_LE.classes_):
                                label = p
                            else:
                                # attempt inverse_transform anyway
                                label = AGENT_LE.inverse_transform([p])[0]
                        except Exception:
                            label = p
                    else:
                        label = str(p)
            except Exception:
                label = str(p)
            return [{"agent": label, "score": _safe_float(1.0), "loading_mol_percent": None, "solvent": solvent_hint}]
    except Exception:
        traceback.print_exc()
        return []

def predict_solvent_from_fp(fp_array: np.ndarray):
    if CLF_SOLVENT is None or SOLVENT_LE is None:
        return None
    try:
        X = fp_array.reshape(1, -1)
        if hasattr(CLF_SOLVENT, "predict_proba"):
            probs = CLF_SOLVENT.predict_proba(X)[0]
            probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
            idx = int(np.argmax(probs))
            # decode similarly to agent
            enc = None
            try:
                enc = CLF_SOLVENT.classes_[idx]
            except Exception:
                enc = idx
            # if enc is numeric, inverse_transform, else use as is
            label = None
            try:
                if isinstance(enc, (int, np.integer)):
                    label = SOLVENT_LE.inverse_transform([int(enc)])[0]
                elif isinstance(enc, str):
                    if enc in list(SOLVENT_LE.classes_):
                        label = enc
                    else:
                        try:
                            label = SOLVENT_LE.inverse_transform([enc])[0]
                        except Exception:
                            label = enc
                else:
                    label = str(enc)
            except Exception:
                label = str(enc)
            return {"solvent": label, "score": _safe_float(float(probs[idx]))}
        else:
            p = CLF_SOLVENT.predict(X)[0]
            try:
                if isinstance(p, (int, np.integer)):
                    label = SOLVENT_LE.inverse_transform([int(p)])[0]
                else:
                    label = str(p)
            except Exception:
                label = str(p)
            return {"solvent": label, "score": _safe_float(1.0)}
    except Exception:
        traceback.print_exc()
        return None

def nearest_similar(fp_array: np.ndarray, k: int = 5):
    if NN_INDEX is None or DF is None:
        return []
    try:
        # use NN_INDEX.kneighbors safely
        nnb = min(k, getattr(NN_INDEX, "n_neighbors", k))
        dists, inds = NN_INDEX.kneighbors(fp_array.reshape(1, -1), n_neighbors=nnb)
        dists = np.nan_to_num(dists, nan=0.0, posinf=0.0, neginf=0.0)[0]
        out = []
        for j, idx in enumerate(inds[0]):
            idx = int(idx)
            try:
                row = DF.iloc[idx]
            except Exception:
                continue
            agent_val = row.get("agent") or row.get("agent_000") or row.get("catalyst") or ""
            solvent_val = row.get("solvent") or row.get("solvent_000") or ""
            temperature_val = row.get("temperature") if "temperature" in row.index else row.get("temperature_c") or row.get("temp")
            yield_val = row.get("yield_percent") if "yield_percent" in row.index else row.get("yield_000") or row.get("yield")
            out.append({
                "id": str(row.get("id", idx)),
                "smiles": row.get("smiles", "") or row.get("rxn_str", ""),
                "agent": agent_val,
                "catalyst": agent_val,
                "solvent": solvent_val,
                "temperature": _safe_float(temperature_val),
                "yield": _safe_float(yield_val),
                "distance": _safe_float(dists[j] if j < len(dists) else None)
            })
        return out
    except Exception:
        traceback.print_exc()
        return []


@APP.post("/predict")
def predict(req: PredictRequest):
    try:
        rxn = (req.smiles or "").strip()
        # determine nBits from artifact fingerprints if present, else fallback
        nbits = None
        try:
            if X_FP_ALL is not None:
                nbits = int(np.asarray(X_FP_ALL).shape[1])
        except Exception:
            nbits = None
        if nbits is None:
            nbits = 128

        fp = reaction_to_fp(rxn, radius=2, nBits=nbits)

        solvent_pred = predict_solvent_from_fp(fp)
        solvent_hint = solvent_pred.get("solvent") if solvent_pred else None

        agent_list = predict_agent_from_fp(fp, topk=5, solvent_hint=solvent_hint)
        primary = agent_list[0] if agent_list else None
        alternatives = agent_list[1:5] if len(agent_list) > 1 else []

        similar = nearest_similar(fp, k=5)

        agent_smiles = primary.get("agent") if primary else None
        solvent_smiles = solvent_hint

        agent_cano = canonical_smiles(agent_smiles) if agent_smiles else None
        solvent_cano = canonical_smiles(solvent_smiles) if solvent_smiles else None

        agent_img = smiles_to_png_base64(agent_smiles)
        solvent_img = smiles_to_png_base64(solvent_smiles)

        resp = {
            "prediction": {
                "catalyst": primary.get("agent") if primary else None,
                "agent": agent_smiles,
                "agent_smiles": agent_smiles,
                "agent_smiles_canonical": agent_cano,
                "agent_img": agent_img,
                "solvent": solvent_smiles,
                "solvent_smiles": solvent_smiles,
                "solvent_smiles_canonical": solvent_cano,
                "solvent_img": solvent_img,
                "confidence": _safe_float(primary.get("score") if primary else 0.0),
                "loading_mol_percent": _safe_float(primary.get("loading_mol_percent") if primary else None),
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
    import uvicorn
    uvicorn.run(APP, host="127.0.0.1", port=8000)
