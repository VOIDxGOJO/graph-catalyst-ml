import pandas as pd
import numpy as np
import pickle
import warnings
import os

from rdkit import RDLogger
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.DataStructs import ConvertToNumpyArray
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings("ignore", category=DeprecationWarning)

def load_dataset(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]
    if 'smiles' not in df.columns or 'pic50' not in df.columns:
        raise ValueError("CSV must have 'smiles' and 'pic50' columns")
    df = df.dropna(subset=['smiles', 'pic50']).reset_index(drop=True)
    df['catalyst'] = df.get('catalyst', 'N/A')
    df['loading'] = df.get('loading', 0.0)
    df['solvent'] = df.get('solvent', '')
    df['base'] = df.get('base', '')
    df['temperature'] = df.get('temperature', 0.0)
    df['id'] = df.index.astype(str)
    return df

def mols_to_fp(mols, radius=2, nBits=512):
    fp_arr = np.zeros((nBits,), dtype=int)
    for mol in mols:
        if mol is None:
            continue
        fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
        arr = np.zeros((nBits,), dtype=int)
        ConvertToNumpyArray(fp, arr)
        fp_arr |= arr
    return fp_arr

def reaction_to_fp(reaction_smiles, radius=2, nBits=512):
    mols = [Chem.MolFromSmiles(s) for s in reaction_smiles.split('.') if s.strip()]
    return mols_to_fp(mols, radius=radius, nBits=nBits)

def featurize(df):
    fps = []
    for smi in df['smiles']:
        try:
            fps.append(reaction_to_fp(smi))
        except:
            fps.append(np.zeros(512, dtype=int))
    X_fp = np.array(fps)

    catalyst_le = LabelEncoder()
    y_cat = catalyst_le.fit_transform(df['catalyst'])
    
    sol_le = LabelEncoder()
    base_le = LabelEncoder()
    sol_vals = sol_le.fit_transform(df['solvent'].fillna(''))
    base_vals = base_le.fit_transform(df['base'].fillna(''))
    temps = df['temperature'].fillna(0.0).astype(float).values

    misc = np.vstack([sol_vals, base_vals, temps]).T
    X = np.hstack([X_fp, misc])
    
    catalyst_to_loading = {c: 0.0 for c in catalyst_le.classes_}

    return X, y_cat, X_fp, catalyst_le, catalyst_to_loading, sol_le, base_le

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/SMILES_Big_Data_Set.csv")
    parser.add_argument("--out", default="model/artifacts.pkl")
    args = parser.parse_args()

    print("Loading dataset...")
    df = load_dataset(args.csv)
    X, y_cat, X_fp, catalyst_le, catalyst_to_loading, sol_le, base_le = featurize(df)

    print("Training classifier...")
    clf = RandomForestClassifier(n_estimators=20, random_state=42, n_jobs=-1)
    clf.fit(X, y_cat)

    print("Training regressor...")
    reg = RandomForestRegressor(n_estimators=20, random_state=42, n_jobs=-1)
    reg.fit(X, df['pic50'].values)

    artifacts = {
        "clf": clf,
        "reg": reg,
        "catalyst_le": catalyst_le,
        "sol_le": sol_le,
        "base_le": base_le,
        "train_fp": X_fp,
        "train_smiles": df['smiles'].tolist(),
        "train_catalyst": df['catalyst'].tolist(),
        "train_loading": df['loading'].tolist()
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(artifacts, f)
    print(f"Saved artifacts to {args.out}")
