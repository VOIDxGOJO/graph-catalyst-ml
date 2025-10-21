# api/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import ConvertToNumpyArray
from typing import Optional

# ---- Request / Response schemas ----
class PredictRequest(BaseModel):
    smiles: str
    solvent: Optional[str] = ""
    base: Optional[str] = ""
    temperature: Optional[float] = 0.0

# ---- Load artifacts ----
ARTIFACT_PATH = "model/artifacts.pkl"
with open(ARTIFACT_PATH, "rb") as f:
    artifacts = pickle.load(f)

# Required objects (train script must have saved these)
clf = artifacts.get("clf")
reg = artifacts.get("reg")
catalyst_le = artifacts.get("catalyst_le")
sol_le = artifacts.get("sol_le")      # may be None
base_le = artifacts.get("base_le")    # may be None

train_fp = np.array(artifacts.get("train_fp"))  # shape (N, fp_dim)
train_smiles = artifacts.get("train_smiles", [])
train_catalyst = artifacts.get("train_catalyst", [])
train_loading = artifacts.get("train_loading", [])
catalyst_to_loading = artifacts.get("catalyst_to_loading", {})

# infer fp_dim and feature_dim
if train_fp is None or train_fp.size == 0:
    raise RuntimeError("train_fp missing or empty in artifacts.pkl — cannot run similarity search")

fp_dim = train_fp.shape[1]
# assume training feature vector was fp_dim + 3 (solvent, base, temperature)
# but if the artifacts stored 'feature_dim' prefer that
feature_dim = artifacts.get("feature_dim", fp_dim + 3)

# ---- helper: compute fingerprint the same way as training ----
def compute_reaction_fp(reaction_smiles: str, nBits: int = None, radius: int = 2):
    """
    Create a combined Morgan fingerprint for the reactant side of a reaction SMILES.
    Returns a numpy array of 0/1 ints length nBits (if provided) or train fp_dim by default.
    """
    if nBits is None:
        nBits = fp_dim
    reac_side = reaction_smiles.split(">>")[0]
    parts = [p for p in reac_side.split(".") if p.strip()]
    fp_arr = np.zeros((nBits,), dtype=int)
    for part in parts:
        mol = Chem.MolFromSmiles(part)
        if mol is None:
            # skip invalid substructure (we still allow prediction attempt)
            continue
        rdkit_fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
        tmp = np.zeros((nBits,), dtype=int)
        ConvertToNumpyArray(rdkit_fp, tmp)
        # combine via OR (same as training function)
        fp_arr |= tmp
    return fp_arr

# ---- featurize incoming request to same dims as training ----
def featurize_request(req: PredictRequest):
    if not req.smiles or not isinstance(req.smiles, str):
        raise ValueError("Missing or invalid smiles string")
    fp = compute_reaction_fp(req.smiles, nBits=fp_dim)

    # solvent encoding
    if sol_le is not None and isinstance(req.solvent, str) and req.solvent.strip() != "":
        try:
            sol_val = int(sol_le.transform([req.solvent])[0])
        except Exception:
            # unknown: fallback to 0 (most common / safe)
            sol_val = 0
    else:
        sol_val = 0

    # base encoding
    if base_le is not None and isinstance(req.base, str) and req.base.strip() != "":
        try:
            base_val = int(base_le.transform([req.base])[0])
        except Exception:
            base_val = 0
    else:
        base_val = 0

    try:
        temp_val = float(req.temperature) if req.temperature is not None else 0.0
    except Exception:
        temp_val = 0.0

    misc = np.array([sol_val, base_val, temp_val], dtype=float)

    # assemble final X vector - ensure correct length
    X = np.hstack([fp.astype(float), misc])
    if X.shape[0] != feature_dim:
        # either pad or truncate to expected feature_dim (safer than crashing).
        if X.shape[0] < feature_dim:
            pad = np.zeros((feature_dim - X.shape[0],), dtype=float)
            X = np.hstack([X, pad])
        else:
            X = X[:feature_dim]
    # reshape for sklearn
    return X.reshape(1, -1), fp.astype(float).reshape(1, -1)

# ---- fastapi app ----
app = FastAPI(title="Graph Catalyst ML")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

@app.get("/")
def read_root():
    return {"message": "Graph Catalyst ML API running. POST /predict with JSON."}

@app.post("/predict")
def predict(payload: PredictRequest):
    try:
        X, fp = featurize_request(payload)   # X used for model, fp used for similarity
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    # classifier/regressor predictions
    try:
        pred_idx = int(clf.predict(X)[0])
        catalyst = str(catalyst_le.inverse_transform([pred_idx])[0])
        # probability (confidence)
        proba = clf.predict_proba(X)[0]
        confidence = float(np.max(proba))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"classifier error: {e}")

    try:
        loading = float(reg.predict(X)[0])
    except Exception as e:
        loading = float(catalyst_to_loading.get(catalyst, 0.0))

    # alternatives = top 3 other catalysts by proba (excluding predicted)
    try:
        probs = clf.predict_proba(X)[0]
        order = np.argsort(probs)[::-1]
        alt_idx = [i for i in order if i != pred_idx][:3]
        alternatives = []
        for i in alt_idx:
            cat_name = str(catalyst_le.inverse_transform([int(i)])[0])
            alternatives.append({
                "catalyst": cat_name,
                "loading_mol_percent": float(round(catalyst_to_loading.get(cat_name, loading), 4)),
                "score": float(probs[i])
            })
    except Exception:
        alternatives = []

    # similar reactions: use cosine similarity between fp and train_fp
    try:
        # ensure train_fp is float and same dims
        train_matrix = train_fp.astype(float)
        if train_matrix.shape[1] != fp.shape[1]:
            # try to resize train_fp or fp (rare). We'll do safe broadcast/truncate
            min_dim = min(train_matrix.shape[1], fp.shape[1])
            train_matrix = train_matrix[:, :min_dim]
            fp_sim = fp[:, :min_dim]
        else:
            fp_sim = fp
        sims = cosine_similarity(fp_sim, train_matrix)[0]  # shape (N,)
        top_sim_idx = np.argsort(sims)[::-1][:3]
        similar_reactions = []
        for i in top_sim_idx:
            similar_reactions.append({
                "id": f"RXN-{i}",
                "smiles": train_smiles[i] if i < len(train_smiles) else "",
                "catalyst": train_catalyst[i] if i < len(train_catalyst) else "",
                "loading": float(train_loading[i]) if i < len(train_loading) else 0.0
            })
    except Exception:
        similar_reactions = []

    return {
        "prediction": {
            "catalyst": catalyst,
            "loading_mol_percent": float(round(loading, 4)),
            "confidence": float(round(confidence, 4)),
            "protocol_note": ""
        },
        "alternatives": alternatives,
        "similar_reactions": similar_reactions
    }
