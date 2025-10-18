from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import pickle
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import ConvertToNumpyArray

# Request and response schema
class PredictRequest(BaseModel):
    smiles: str
    solvent: Optional[str] = None
    base: Optional[str] = None
    temperature: Optional[float] = None

class Prediction(BaseModel):
    catalyst: str
    loading_mol_percent: float
    confidence: float
    protocol_note: Optional[str] = None

class Alternative(BaseModel):
    catalyst: str
    loading_mol_percent: float
    score: float

class SimilarReaction(BaseModel):
    id: str
    smiles: str
    catalyst: str
    loading: float

class PredictResponse(BaseModel):
    prediction: Prediction
    alternatives: List[Alternative]
    similar_reactions: List[SimilarReaction]

app = FastAPI()

# Load trained models and artifacts
with open("model/artifacts.pkl", "rb") as f:
    artifacts = pickle.load(f)
clf = artifacts['clf']
reg = artifacts['reg']
catalyst_le = artifacts['catalyst_le']
sol_le = artifacts['sol_le']
base_le = artifacts['base_le']
cat_to_load = artifacts['catalyst_to_loading']
train_ids    = artifacts['train_ids']
train_smiles= artifacts['train_smiles']
train_catalyst = artifacts['train_catalyst']
train_loading  = artifacts['train_loading']
train_fp = np.array(artifacts['train_fp'])

def compute_fingerprint(smiles, radius=2, nBits=2048):
    """
    Compute combined Morgan fingerprint for the reactant side of a reaction SMILES
    """
    reac_side = smiles.split('>>')[0]
    parts = reac_side.split('.')
    fp_arr = np.zeros((nBits,), dtype=int)
    for part in parts:
        mol = Chem.MolFromSmiles(part)
        if mol is None: 
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
        arr = np.zeros((nBits,), dtype=int)
        ConvertToNumpyArray(fp, arr)
        fp_arr |= arr
    return fp_arr

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    if not request.smiles:
        raise HTTPException(status_code=400, detail="Missing SMILES input.")
    # fingerprint features from input
    fp = compute_fingerprint(request.smiles)
    solvent = request.solvent or ""
    base    = request.base or ""
    # Encode solvent/base; if unseen, default to first category (index 0)
    try:
        sol_id  = int(sol_le.transform([solvent])[0])
    except Exception:
        sol_id = 0
    try:
        base_id = int(base_le.transform([base])[0])
    except Exception:
        base_id = 0
    temp = float(request.temperature) if request.temperature is not None else 0.0
    X_input = np.concatenate([fp, [sol_id, base_id, temp]])

    proba = clf.predict_proba([X_input])[0]
    top_idx = int(np.argmax(proba))
    pred_cat = str(catalyst_le.inverse_transform([top_idx])[0])
    confidence = float(proba[top_idx])

    loading_pred = float(reg.predict([X_input])[0])
    prediction = {
        "catalyst": pred_cat,
        "loading_mol_percent": round(loading_pred, 2),
        "confidence": confidence,
        "protocol_note": ""
    }
    # alts, top3 other catalysts by predicted probability
    top_indices = np.argsort(proba)[::-1]
    alt_indices = [i for i in top_indices if i != top_idx][:3]
    alternatives = []
    for i in alt_indices:
        cat_alt = str(catalyst_le.inverse_transform([int(i)])[0])
        score = float(proba[i])
        load_alt = float(cat_to_load.get(cat_alt, 0.0))
        alternatives.append({
            "catalyst": cat_alt,
            "loading_mol_percent": round(load_alt, 2),
            "score": score
        })
    # find nearest training examples by Tanimoto similarity
    dot_prods = np.dot(train_fp, fp)
    sum_a = fp.sum()
    sum_b = np.sum(train_fp, axis=1)
    denom = sum_a + sum_b - dot_prods + 1e-9
    sims = dot_prods / denom
    sim_indices = np.argsort(sims)[::-1][:3]
    similar_list = []
    for idx in sim_indices:
        similar_list.append({
            "id": str(train_ids[idx]),
            "smiles": str(train_smiles[idx]),
            "catalyst": str(train_catalyst[idx]),
            "loading": float(train_loading[idx])
        })
    return {
        "prediction": prediction,
        "alternatives": alternatives,
        "similar_reactions": similar_list
    }
