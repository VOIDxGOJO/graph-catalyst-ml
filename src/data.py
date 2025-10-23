
"""
util for orderly dataset

load_orderly_csv(path)- returns cleaned df with columns:
  ['id','smiles','agent','solvent','temperature','rxn_time','yield_percent','procedure_details']

reaction_to_fp(smiles, radius, nBits)- returns 1D numpy int8 fingerprint
featurize_df(df, radius, nBits, show_progress)- returns (X_fp, meta)
    X_fp- numpy array N x nBits (dtype int8)
    meta- dict with fp params (not fitted LabelEncoders)
"""

from typing import Tuple, Dict, Any
import pandas as pd
import numpy as np
import re
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors
from rdkit.DataStructs import ConvertToNumpyArray
from tqdm import tqdm

RDLogger.DisableLog('rdApp.*')

# defaults
FP_RADIUS = 2
FP_NBITS = 128

_RE_COLON_NUMBER = re.compile(r':\d+')
_RE_BRACKET_MAP = re.compile(r'\[([^\]:]+):\d+\]')

def _strip_atom_map(s: str) -> str:
    if s is None:
        return s
    s = _RE_BRACKET_MAP.sub(r'[\1]', s)
    s = _RE_COLON_NUMBER.sub('', s)
    return s

def _compose_rxn_str_from_row(r):
    if 'rxn_str' in r and pd.notna(r['rxn_str']) and str(r['rxn_str']).strip() != '':
        return str(r['rxn_str'])
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
    return ""

def load_orderly_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, low_memory=False)
    df.columns = [c.strip().lower() for c in df.columns]
    df['id'] = df.get('index', pd.Series(range(len(df)))).astype(str)
    df['smiles'] = df.apply(_compose_rxn_str_from_row, axis=1)

    df['agent'] = df.get('agent_000', pd.NA)
    df['agent'] = df['agent'].replace('\\N', pd.NA)
    df['agent'] = df['agent'].where(df['agent'].notna(), None)

    df['solvent'] = df.get('solvent_000', pd.NA)
    df['solvent'] = df['solvent'].replace('\\N', pd.NA)
    df['solvent'] = df['solvent'].where(df['solvent'].notna(), None)

    def to_float_safe(x):
        try:
            if pd.isna(x): return None
            return float(x)
        except Exception:
            return None

    if 'temperature' in df.columns:
        df['temperature'] = df['temperature'].apply(lambda x: to_float_safe(x))
    else:
        df['temperature'] = None

    if 'rxn_time' in df.columns:
        df['rxn_time'] = df['rxn_time'].apply(lambda x: to_float_safe(x))
    else:
        df['rxn_time'] = None

    if 'yield_000' in df.columns:
        df['yield_percent'] = df['yield_000'].apply(lambda x: to_float_safe(x))
    else:
        df['yield_percent'] = None

    df['procedure_details'] = df.get('procedure_details', None)

    keep_cols = ['id', 'smiles', 'agent', 'solvent', 'temperature', 'rxn_time', 'yield_percent', 'procedure_details']
    for c in keep_cols:
        if c not in df.columns:
            df[c] = None
    return df[keep_cols].copy()

def mols_to_fp(mols, radius=FP_RADIUS, nBits=FP_NBITS):
    arr = np.zeros((nBits,), dtype=np.int8)
    for mol in mols:
        if mol is None:
            continue
        try:
            fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
            tmp = np.zeros((nBits,), dtype=np.int8)
            ConvertToNumpyArray(fp, tmp)
            arr = np.bitwise_or(arr, tmp)
        except Exception:
            continue
    return arr

def reaction_to_fp(reaction_smiles: str, radius=FP_RADIUS, nBits=FP_NBITS):
    if not reaction_smiles or str(reaction_smiles).strip() == "":
        return np.zeros(nBits, dtype=np.int8)
    smi = str(reaction_smiles)
    smi_clean = _strip_atom_map(smi)
    try:
        parts = smi_clean.split('>>')
        left = parts[0] if len(parts) > 0 else ''
        right = parts[1] if len(parts) > 1 else ''
        tokens = (left + '.' + right).split('.')
        mols = []
        for token in tokens:
            token = token.strip()
            if token == '':
                continue
            try:
                m = Chem.MolFromSmiles(token)
                mols.append(m)
            except Exception:
                mols.append(None)
        return mols_to_fp(mols, radius=radius, nBits=nBits)
    except Exception:
        return np.zeros(nBits, dtype=np.int8)

def featurize_df(df: pd.DataFrame, radius=FP_RADIUS, nBits=FP_NBITS, show_progress: bool = True) -> Tuple[np.ndarray, Dict[str, Any]]:
    n = len(df)
    fps = np.zeros((n, nBits), dtype=np.int8)
    bad_count = 0
    it = tqdm(df['smiles'].fillna(''), desc="Featurizing", disable=not show_progress)
    for i, smi in enumerate(it):
        try:
            fp = reaction_to_fp(smi, radius=radius, nBits=nBits)
            fps[i, :] = fp
        except KeyboardInterrupt:
            raise
        except Exception:
            bad_count += 1
            fps[i, :] = np.zeros((nBits,), dtype=np.int8)
    if bad_count > 0:
        print(f"Featurization: {bad_count} rows had parse/feature errors and were zeroed.")
    meta = {
        'fp_params': {'radius': radius, 'nbits': nBits},
        'n_rows': n
    }
    return fps, meta
