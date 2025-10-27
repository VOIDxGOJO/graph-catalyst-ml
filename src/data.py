"""
featurizer utils for reaction smiles


atom mapping removal and light cleaning
fragment-level LRU cache for Morgan fingerprints (huge speed up if many repeated fragments)
skip extremely long fragments cheaply
returns numpy float32 array of bits (shape (nBits,))
"""

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit import RDLogger
import re
import numpy as np
from functools import lru_cache

# disable rdkit warnings
RDLogger.DisableLog('rdApp.*')

# regexes
RE_ATOM_MAP = re.compile(r':\d+')            # [C:1] or :1 mapping
RE_BRACKETED_EMPTY = re.compile(r'\[ *\]')  # empty brackets
RE_NON_PRINT = re.compile(r'[^\x20-\x7E]')  # non printable
RE_KEEP = re.compile(r'[^0-9a-zA-Z\+\-\(\)\.\[\]\=\#@:/,]')  # allowedish chars

# safety limits
MAX_FRAGMENT_LENGTH = 400   # skip
MAX_TOTAL_FRAGMENTS = 40    # if >40 fragments, skip heavy parsing and return zero vector

@lru_cache(maxsize=200000)
def _frag_to_fp_cached(frag_smiles: str, radius: int, nBits: int):
    """
    cached helper:-compute a fingerprint for single fragment smiles
    return numpy uint8 array of length nBits (0/1).
    """
    try:
        frag = frag_smiles.strip()
        if frag == "" or frag == "\\N":
            return None
        # cheap guards
        if len(frag) > MAX_FRAGMENT_LENGTH:
            return None
        # try parse
        m = Chem.MolFromSmiles(frag)
        if m is None:
            # try more permissive cleanup- remove weird chars then parse
            cand = RE_NON_PRINT.sub('', frag)
            cand = RE_ATOM_MAP.sub('', cand)
            cand = RE_KEEP.sub('', cand)
            if len(cand) == 0 or len(cand) > MAX_FRAGMENT_LENGTH:
                return None
            m = Chem.MolFromSmiles(cand)
            if m is None:
                return None
        # compute circular fingerprint bit vector (morgan)
        fp = AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=nBits)
        arr = np.zeros((nBits,), dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr
    except Exception:
        # any rdkit error then return None (treated as no bits)
        return None


def reaction_to_fp(reaction_smiles, radius=2, nBits=128):
    """
    convert reaction smiles (reactants...>>product) into a fingerprint vector
    
    split reactant fragments and product fragments
    compute fragment fingerprints (cached)
    bitwise-or them to build a reaction-level fingerprint
    return float32 numpy vector shape (nBits,)
    return zeros if input is invalid or cannot be parsed.
    """
    if reaction_smiles is None:
        return np.zeros((nBits,), dtype=np.float32)
    s = str(reaction_smiles).strip()
    if s == "" or s == "\\N":
        return np.zeros((nBits,), dtype=np.float32)

    # remove obvious atom-mapping tokens like [C:1] or :1 (keep connectivity)
    try:
        s_clean = RE_ATOM_MAP.sub('', s)

        s_clean = RE_NON_PRINT.sub('', s_clean) # non printable
    except Exception:
        s_clean = s

    # split into reactants and products if possible
    parts = s_clean.split(">>")
    if len(parts) == 0:
        return np.zeros((nBits,), dtype=np.float32)
    reactants_part = parts[0] if parts[0] is not None else ""
    product_part = parts[1] if len(parts) > 1 else ""

    frags = []
    # split reactants and products into fragments by '.' but cap fragments to avoid explosion
    try:
        react_frags = [f.strip() for f in reactants_part.split('.') if f.strip()]
        prod_frags = [f.strip() for f in product_part.split('.') if f.strip()]
        all_frags = react_frags + prod_frags
    except Exception:
        return np.zeros((nBits,), dtype=np.float32)

    if len(all_frags) == 0 or len(all_frags) > MAX_TOTAL_FRAGMENTS:
        # too many fragments (badly tokenized); bail to keep memory/cpu running
        return np.zeros((nBits,), dtype=np.float32)

    fp_accum = np.zeros((nBits,), dtype=np.uint8)
    for frag in all_frags:
        # cheap cleanup per frag
        frag_norm = frag.strip()
        if frag_norm == "" or frag_norm == "\\N":
            continue
        # remove atom mapping tokens and stray punctuation
        frag_norm = RE_ATOM_MAP.sub('', frag_norm)
        frag_norm = RE_NON_PRINT.sub('', frag_norm)
        # if fragment still too large skip
        if len(frag_norm) > MAX_FRAGMENT_LENGTH:
            continue
        # use cached fragment-level fingerprint function
        arr = _frag_to_fp_cached(frag_norm, radius, nBits)
        if arr is None:
            continue
        # OR bits
        fp_accum = np.bitwise_or(fp_accum, arr)

    return fp_accum.astype(np.float32)


def load_orderly_csv(path, label_column="agent_norm"):
    """
    minimal loader used by training code; read CSV and filter rows where
    label_column is non-empty (but does not touch smiles)
    """
    import pandas as pd
    df = pd.read_csv(path, dtype=str, low_memory=False)
    if label_column:
        df = df[df[label_column].notna() & (df[label_column].astype(str).str.strip() != "")]
    return df
