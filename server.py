from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import joblib
import numpy as np
import math
import traceback
from rdkit import Chem
from rdkit.Chem import Draw
from io import BytesIO
import base64
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from src.data import reaction_to_fp

ARTIFACT_PATH = "models/artifacts_quick.joblib"

# globals to be filled at startup
ART = None
CLF_AGENT = None
AGENT_LE = None
CLF_SOLVENT = None
SOLVENT_LE = None
NN_INDEX = None
X_FP_ALL = None
DF = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global ART, CLF_AGENT, AGENT_LE, CLF_SOLVENT, SOLVENT_LE, NN_INDEX, X_FP_ALL, DF
    try:
        print("🔹 Loading artifacts from:", ARTIFACT_PATH)
        ART = joblib.load(ARTIFACT_PATH)
        print("  Keys in artifact:", list(ART.keys()))
        # safe extraction without using truth-value of arrays
        CLF_AGENT = ART.get("clf_agent", None)
        AGENT_LE = ART.get("agent_le", None)
        CLF_SOLVENT = ART.get("clf_solvent", None)
        SOLVENT_LE = ART.get("solvent_le", None)
        NN_INDEX = ART.get("nn_index", None)

        # choose X_fp field
        if "X_fp_all" in ART and ART["X_fp_all"] is not None:
            X_FP_ALL = ART["X_fp_all"]
            print("  Using X_fp_all (shape):", getattr(X_FP_ALL, "shape", "n/a"))
        elif "X_fp_for_nn" in ART and ART["X_fp_for_nn"] is not None:
            X_FP_ALL = ART["X_fp_for_nn"]
            print("  Using X_fp_for_nn (shape):", getattr(X_FP_ALL, "shape", "n/a"))
        else:
            X_FP_ALL = None
            print("  No fingerprint array (X_fp_all / X_fp_for_nn) found in artifact.")

        # choose df
        if "df" in ART and ART["df"] is not None:
            DF = ART["df"]
            print("  Using DF from key 'df' (len):", len(DF) if hasattr(DF, "__len__") else "n/a")
        elif "df_for_nn" in ART and ART["df_for_nn"] is not None:
            DF = ART["df_for_nn"]
            print("  Using DF from key 'df_for_nn' (len):", len(DF) if hasattr(DF, "__len__") else "n/a")
        else:
            DF = None
            print("  No dataframe (df / df_for_nn) found in artifact.")

        print("Artifacts loaded (safe)")
    except Exception as e:
        print("❌ Failed to load artifacts:", e)
        traceback.print_exc()
        raise e
    yield
    print("Server shutting down..")


app = FastAPI(title="Catalyst ML API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictRequest(BaseModel):
    smiles: Optional[str] = None
    product_smiles: Optional[str] = None
    solvent: Optional[str] = None
    temperature_c: Optional[float] = None
    rxn_time_h: Optional[float] = None
    yield_percent: Optional[float] = None

# helpers
def safe_float(v):
    try:
        if v is None: return None
        f = float(v)
        return f if math.isfinite(f) else None
    except Exception:
        return None

def normalize_proba(arr):
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    s = arr.sum()
    return arr / s if s > 0 else arr

def canonical_smiles(smiles):
    if not smiles:
        return None
    try:
        m = Chem.MolFromSmiles(smiles)
        return Chem.MolToSmiles(m, canonical=True) if m else smiles
    except Exception:
        return smiles

def smiles_to_png_base64(smiles, size=(220,160)):
    if not smiles:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return None
        img = Draw.MolToImage(mol, size=size)
        buf = BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception:
        return None

# core prediction helpers
def predict_agent(fp_array, topk=5, solvent_hint=None):
    if CLF_AGENT is None or AGENT_LE is None:
        return []
    try:
        if hasattr(CLF_AGENT, "predict_proba"):
            proba = normalize_proba(CLF_AGENT.predict_proba(fp_array.reshape(1, -1))[0])
            classes = CLF_AGENT.classes_
            top_idx = np.argsort(proba)[::-1][:topk]
            out = []
            for i in top_idx:
                enc = classes[int(i)]
                try:
                    label = AGENT_LE.inverse_transform([int(enc)])[0]
                except Exception:
                    # fallback if mapping fails
                    label = str(enc)
                out.append({"agent": label, "score": float(proba[i]), "loading_mol_percent": None, "solvent": solvent_hint})
            return out
        else:
            idx = int(CLF_AGENT.predict(fp_array.reshape(1, -1))[0])
            label = AGENT_LE.inverse_transform([idx])[0]
            return [{"agent": label, "score": 1.0}]
    except Exception as e:
        print("Agent prediction error:", e)
        traceback.print_exc()
        return []

def predict_solvent(fp_array):
    if CLF_SOLVENT is None or SOLVENT_LE is None:
        return None
    try:
        if hasattr(CLF_SOLVENT, "predict_proba"):
            proba = normalize_proba(CLF_SOLVENT.predict_proba(fp_array.reshape(1, -1))[0])
            classes = CLF_SOLVENT.classes_
            idx = int(np.argmax(proba))
            enc = classes[idx]
            try:
                label = SOLVENT_LE.inverse_transform([int(enc)])[0]
            except Exception:
                label = str(enc)
            return {"solvent": label, "score": float(proba[idx])}
        else:
            idx = int(CLF_SOLVENT.predict(fp_array.reshape(1, -1))[0])
            label = SOLVENT_LE.inverse_transform([idx])[0]
            return {"solvent": label, "score": 1.0}
    except Exception as e:
        print("Solvent prediction error:", e)
        traceback.print_exc()
        return None

def nearest_similar(fp_array, k=5):
    if NN_INDEX is None or DF is None:
        return []
    try:
        dists, inds = NN_INDEX.kneighbors(fp_array.reshape(1, -1), n_neighbors=k)
        out = []
        for j, idx in enumerate(inds[0]):
            r = DF.iloc[int(idx)]
            out.append({
                "id": str(r.get("id", idx)),
                "smiles": r.get("smiles", "") or r.get("rxn_str", ""),
                "agent": r.get("agent", "") or r.get("agent_000", ""),
                "solvent": r.get("solvent", "") or r.get("solvent_000", ""),
                "temperature": safe_float(r.get("temperature")),
                "yield": safe_float(r.get("yield_percent") or r.get("yield_000")),
                "distance": float(dists[0][j]) if (dists is not None and len(dists) and j < len(dists[0])) else None
            })
        return out
    except Exception as e:
        print("Nearest-neighbor error:", e)
        traceback.print_exc()
        return []

# endpoint
@app.post("/predict")
def predict(req: PredictRequest):
    try:
        smiles = (req.smiles or "").strip()
        if not smiles:
            raise HTTPException(status_code=400, detail="Missing SMILES string")

        nbits = int(X_FP_ALL.shape[1]) if (X_FP_ALL is not None and hasattr(X_FP_ALL, "shape")) else 256
        fp = reaction_to_fp(smiles, radius=2, nBits=nbits)

        solvent_pred = predict_solvent(fp)
        solvent_hint = solvent_pred.get("solvent") if solvent_pred else None
        agents = predict_agent(fp, topk=5, solvent_hint=solvent_hint)

        primary = agents[0] if agents else None
        alternatives = agents[1:] if len(agents) > 1 else []
        similar = nearest_similar(fp)

        agent_smiles = primary.get("agent") if primary else None
        solvent_smiles = solvent_hint

        agent_cano = canonical_smiles(agent_smiles)
        solvent_cano = canonical_smiles(solvent_smiles)
        agent_img = smiles_to_png_base64(agent_smiles)
        solvent_img = smiles_to_png_base64(solvent_smiles)

        return {
            "prediction": {
                "agent": agent_smiles,
                "agent_smiles": agent_smiles,
                "agent_smiles_canonical": agent_cano,
                "agent_img": agent_img,
                "solvent": solvent_smiles,
                "solvent_smiles": solvent_smiles,
                "solvent_smiles_canonical": solvent_cano,
                "solvent_img": solvent_img,
                "confidence": safe_float(primary.get("score") if primary else 0.0)
            },
            "alternatives": alternatives,
            "similar_reactions": similar,
            "solvent_prediction": solvent_pred
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
