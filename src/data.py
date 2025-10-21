import pandas as pd
import numpy as np
import pickle

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import ConvertToNumpyArray
from sklearn.preprocessing import LabelEncoder
from joblib import Parallel, delayed

def load_dataset(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]

    required = ['smiles', 'pic50']
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found. Found columns: {list(df.columns)}")

    df = df.dropna(subset=['smiles', 'pic50']).reset_index(drop=True)

    df['catalyst'] = df.get('catalyst', 'N/A')
    df['loading'] = df.get('loading', 0.0)
    df['solvent'] = df.get('solvent', '')
    df['base'] = df.get('base', '')
    df['temperature'] = df.get('temperature', 0.0)

    if 'id' not in df.columns:
        df['id'] = df.index.astype(str)

    return df

def mols_to_fp(mols, radius=2, nBits=2048):
    fp_arr = np.zeros((nBits,), dtype=int)
    for mol in mols:
        if mol is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
        arr = np.zeros((nBits,), dtype=int)
        ConvertToNumpyArray(fp, arr)
        fp_arr |= arr
    return fp_arr

def reaction_to_fp(reaction_smiles, radius=2, nBits=2048):
    reac_side = reaction_smiles.split('>>')[0]
    parts = reac_side.split('.')
    mols = [Chem.MolFromSmiles(part) for part in parts]
    return mols_to_fp(mols, radius=radius, nBits=nBits)

def featurize(df):
    fps = Parallel(n_jobs=-1)(delayed(reaction_to_fp)(smi) for smi in df['smiles'])

    with open("data/fps.pkl", "wb") as f:
        pickle.dump(fps, f)

    X_fp = np.array(fps)
    catalyst_le = LabelEncoder()
    y_cat = catalyst_le.fit_transform(df['catalyst'])

    catalyst_to_loading = {
        c: df.loc[df['catalyst'] == c, 'loading'].astype(float).mean()
        for c in catalyst_le.classes_
    }

    sol_le = LabelEncoder()
    base_le = LabelEncoder()
    solvents = sol_le.fit_transform(df['solvent'].fillna(''))
    bases = base_le.fit_transform(df['base'].fillna(''))
    temps = df['temperature'].fillna(0.0).astype(float).values

    misc = np.vstack([solvents, bases, temps]).T
    X = np.hstack([X_fp, misc])

    return X, y_cat, X_fp, catalyst_le, catalyst_to_loading, sol_le, base_le
