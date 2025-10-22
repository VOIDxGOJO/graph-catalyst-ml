from typing import Tuple, Dict, Any
import pandas as pd
import numpy as np
import joblib

# RDKit imports
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.DataStructs import ConvertToNumpyArray

from sklearn.preprocessing import LabelEncoder

# fingerprint params
FP_RADIUS = 2
FP_NBITS = 1024

def _compose_rxn_str_from_row(r):
    # prefer rxn_str if present and non-null
    if 'rxn_str' in r and pd.notna(r['rxn_str']) and str(r['rxn_str']).strip() != '':
        return r['rxn_str']
    # else try product + reactants into reactant1.reactant2>>product
    reactants = []
    for c in ['reactant_000', 'reactant_001', 'reactant_002']:
        if c in r and pd.notna(r[c]) and str(r[c]).strip() != '':
            reactants.append(str(r[c]).strip())
    product = None
    if 'product_000' in r and pd.notna(r['product_000']) and str(r['product_000']).strip() != '':
        product = str(r['product_000']).strip()
    if reactants:
        left = '.'.join(reactants)
        if product:
            return f"{left}>>{product}"
        else:
            return f"{left}>>"
    # fallback
    return ""

def load_orderly_csv(path: str) -> pd.DataFrame:
    """
    load ORDerly csv and standardize column names and fields needed.
    return df with columns: id, smiles, agent, solvent, temperature, rxn_time, yield, procedure_details
    """
    df = pd.read_csv(path, dtype=str, low_memory=False)
    df.columns = [c.strip().lower() for c in df.columns]

    df['id'] = df.get('index', pd.Series(range(len(df)))).astype(str)
    # compose canonical reaction SMILES string
    df['smiles'] = df.apply(_compose_rxn_str_from_row, axis=1)

    # agents: prefer agent_000
    df['agent'] = df.get('agent_000', pd.NA)
    df['agent'] = df['agent'].replace('\\N', pd.NA)  
    df['agent'] = df['agent'].where(df['agent'].notna(), None)

    # solvent
    df['solvent'] = df.get('solvent_000', pd.NA)
    df['solvent'] = df['solvent'].replace('\\N', pd.NA)
    df['solvent'] = df['solvent'].where(df['solvent'].notna(), None)

    # numeric fields
    def to_float_safe(x):
        try:
            if pd.isna(x): return None
            return float(x)
        except Exception:
            return None

    df['temperature'] = df.get('temperature', None).apply(lambda x: to_float_safe(x) if not pd.isna(x) else None) \
        if 'temperature' in df.columns else None
    df['rxn_time'] = df.get('rxn_time', None).apply(lambda x: to_float_safe(x) if not pd.isna(x) else None) \
        if 'rxn_time' in df.columns else None
    # yield could be yield_000
    if 'yield_000' in df.columns:
        df['yield_percent'] = df['yield_000'].apply(lambda x: to_float_safe(x) if not pd.isna(x) else None)
    else:
        df['yield_percent'] = None

    df['procedure_details'] = df.get('procedure_details', None)
    # minimal
    keep_cols = ['id', 'smiles', 'agent', 'solvent', 'temperature', 'rxn_time', 'yield_percent', 'procedure_details']
    for c in keep_cols:
        if c not in df.columns:
            df[c] = None
    out = df[keep_cols].copy()
    return out

# RDKit fingerprint helpers
def mols_to_fp(mols, radius=FP_RADIUS, nBits=FP_NBITS):
    arr = np.zeros((nBits,), dtype=np.int8)
    for mol in mols:
        if mol is None:
            continue
        try:
            fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
            tmp = np.zeros((nBits,), dtype=np.int8)
            ConvertToNumpyArray(fp, tmp)
            # OR-combine bits so reaction fingerprint encodes presence across components
            arr = np.bitwise_or(arr, tmp)
        except Exception:
            continue
    return arr

def reaction_to_fp(reaction_smiles: str, radius=FP_RADIUS, nBits=FP_NBITS):
    """
    Given a reaction SMILES 'A.B>>C' or similar, compute OR of Morgan fingerprints
    for reactants and product (simple, robust).
    """
    if not reaction_smiles or str(reaction_smiles).strip() == "":
        return np.zeros(nBits, dtype=np.int8)
    try:
        # get left and right around '>>'
        parts = str(reaction_smiles).split('>>')
        left = parts[0] if len(parts) > 0 else ''
        right = parts[1] if len(parts) > 1 else ''
        molecules = []
        for token in (left + '.' + right).split('.'):
            token = token.strip()
            if token == '':
                continue
            try:
                m = Chem.MolFromSmiles(token)
                molecules.append(m)
            except Exception:
                molecules.append(None)
        fp = mols_to_fp(molecules, radius=radius, nBits=nBits)
        return fp
    except Exception:
        return np.zeros(nBits, dtype=np.int8)

def featurize_df(df: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Featurize dataset:
      - compute fingerprints (N x FP_NBITS)
      - create label encoders for agent and solvent (agent may be None)
    Returns:
      X_fp (numpy array), artifacts dict {agent_le, solvent_le, df_index -> metadata}
    """
    fps = []
    for smi in df['smiles'].fillna(''):
        fps.append(reaction_to_fp(smi))
    X_fp = np.vstack(fps).astype(np.int8)

    # label encoders only for non-null targets
    agent_le = LabelEncoder()
    agent_vals = df['agent'].fillna('<<NULL>>').astype(str).values
    agent_le.fit(agent_vals)
    solvent_le = LabelEncoder()
    solvent_vals = df['solvent'].fillna('<<NULL>>').astype(str).values
    solvent_le.fit(solvent_vals)

    artifacts = {
        'agent_le': agent_le,
        'solvent_le': solvent_le,
        'fp_params': {'radius': FP_RADIUS, 'nbits': FP_NBITS},
        'df': df  # keep original standardized df for retrieval
    }
    return X_fp, artifacts

def save_artifacts(path: str, artifacts: dict):
    joblib.dump(artifacts, path)

def load_artifacts(path: str) -> dict:
    return joblib.load(path)
