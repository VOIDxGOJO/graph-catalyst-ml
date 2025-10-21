from fastapi import FastAPI
from pydantic import BaseModel
import pickle
from src.data import reaction_to_fp
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

with open("model/artifacts.pkl", "rb") as f:
    artifacts = pickle.load(f)

clf = artifacts["clf"]
reg = artifacts["reg"]
catalyst_le = artifacts["catalyst_le"]
sol_le = artifacts.get("sol_le")
base_le = artifacts.get("base_le")
train_fp = artifacts["train_fp"]        # shape (N, 512)
train_smiles = artifacts["train_smiles"]
train_catalyst = artifacts["train_catalyst"]
train_loading = artifacts["train_loading"]

app = FastAPI(title="Graph Catalyst ML")

class PredictRequest(BaseModel):
    smiles: str
    solvent: str = ""
    base: str = ""
    temperature: float = 0.0

def featurize_request(req: PredictRequest):
    # fingerprint
    fp = reaction_to_fp(req.smiles)
    if fp.shape[0] != 512:
        fp = np.zeros(512, dtype=int)

    sol_val = 0
    if sol_le and req.solvent and req.solvent in sol_le.classes_:
        sol_val = sol_le.transform([req.solvent])[0]

    base_val = 0
    if base_le and req.base and req.base in base_le.classes_:
        base_val = base_le.transform([req.base])[0]

    temp_val = float(req.temperature)
    misc = np.array([sol_val, base_val, temp_val], dtype=float)

    X = np.hstack([fp, misc]).reshape(1, -1)  # shape = (1, 515)
    return X

@app.post("/predict")
def predict(req: PredictRequest):
    X = featurize_request(req)

    # classifier 
    catalyst_idx = clf.predict(X)[0]
    catalyst = catalyst_le.inverse_transform([catalyst_idx])[0]

    # regressor
    loading = reg.predict(X)[0]

    # confidence score
    conf = clf.predict_proba(X).max()

    # top alt catalysts
    top_idx = np.argsort(clf.predict_proba(X)[0])[::-1][1:4]
    alternatives = []
    for idx in top_idx:
        alternatives.append({
            "catalyst": catalyst_le.inverse_transform([idx])[0],
            "loading_mol_percent": float(loading),
            "score": float(clf.predict_proba(X)[0][idx])
        })

    # similarity search on fingerprints only
    sims = cosine_similarity(X[:, :512], train_fp)[0]
    top_sim_idx = sims.argsort()[::-1][:3]
    similar_reactions = []
    for i in top_sim_idx:
        similar_reactions.append({
            "id": f"RXN-{i}",
            "smiles": train_smiles[i],
            "catalyst": train_catalyst[i],
            "loading": float(train_loading[i])
        })

    return {
        "prediction": {
            "catalyst": catalyst,
            "loading_mol_percent": float(loading),
            "confidence": float(conf),
            "protocol_note": ""
        },
        "alternatives": alternatives,
        "similar_reactions": similar_reactions
    }

@app.get("/")
def read_root():
    return {"message": "Graph Catalyst ML API is running. Use /predict to get predictions."}
