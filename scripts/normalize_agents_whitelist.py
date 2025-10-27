#!/usr/bin/env python3
"""
streaming normalization with catalyst whitelist

saves am output csv with agent_norm column containing only canonical
catalyst/ligand/base names from the WHITELIST
everything else becomes ""

small test-
python scripts/normalize_agents_whitelist.py --input ./data/orderly_condition_with_rare_train.csv \
  --output ./data/orderly_condition_with_rare_train_norm_whitelist.csv \
  --chunksize 5000 --max-rows 5000 --force-overwrite

"""

import argparse
import pandas as pd
import re
from pathlib import Path
from collections import Counter
import time

# canonical map (extendable)
CANONICAL_MAP = {
    # palladium / ligands
    "pd(pph3)4": "Pd(PPh3)4",
    "pd(pph3)2cl2": "Pd(PPh3)2Cl2",
    "pd2(dba)3": "Pd2(dba)3",
    "pd": "Pd",
    "palladium": "Pd",
    # platinum
    "pt": "Pt", "[pt]": "Pt", "platinum": "Pt",
    # copper
    "cui": "CuI", "cu(i)": "CuI", "cubr": "CuBr", "cucl2": "CuCl2", "cu": "Cu",
    # nickel
    "ni": "Ni", "nicl2": "NiCl2",
    # common ligands/bases
    "pph3": "PPh3", "triphenylphosphine": "PPh3",
    "dppf": "dppf", "binap": "BINAP", "nhc": "NHC",
    "potassium tert-butoxide": "KOtBu", "kotbu": "KOtBu",
    "potassium": "K+", "k+": "K+", "[k+]": "K+",
    "sodium": "Na+", "na+": "Na+", "[na+]": "Na+",
    "lithium": "Li+", "li+": "Li+",
    # acids/buffers
    "hcl": "HCl", "h2so4": "H2SO4",
    # metals often used as catalysts
    "ru": "Ru", "rh": "Rh", "ag": "Ag", "fe": "Fe", "co": "Co", "mos": "Mo"
}

# whitelist: tokens we allow as training labels (canonical names)
WHITELIST = set([
    "Pd", "Pd(PPh3)4", "Pd(PPh3)2Cl2", "Pd2(dba)3",
    "Pt", "CuI", "CuBr", "Cu", "Ni", "NiCl2",
    "PPh3", "dppf", "BINAP", "NHC", "KOtBu",
    "K+", "Na+", "Li+", "Ru", "Rh", "Ag", "Fe", "Co",
])

RE_CHARGE = re.compile(r'[\[\]]')
RE_NON_ALPHANUM = re.compile(r'[^0-9a-z\+\-\(\)\.]')
RE_AGENT_COL = re.compile(r'^agent_(\d+)$', flags=re.IGNORECASE)

def canonicalize_agent(raw: str) -> str:
    if raw is None:
        return ""
    s = str(raw).strip()
    if s == "" or s == "\\N":
        return ""
    s_low = s.lower()
    s_low = RE_CHARGE.sub('', s_low)
    s_low = re.sub(r'\s+', ' ', s_low).strip()
    # direct map
    if s_low in CANONICAL_MAP:
        return CANONICAL_MAP[s_low]
    # punctuation-stripped
    s_stripped = RE_NON_ALPHANUM.sub('', s_low)
    if s_stripped in CANONICAL_MAP:
        return CANONICAL_MAP[s_stripped]
    # heuristics- metals
    if 'pd' in s_low:
        if 'pph3' in s_low:
            return "Pd(PPh3)4"
        return "Pd"
    if 'pt' in s_low:
        return "Pt"
    if 'cu' in s_low:
        if 'i' in s_low:
            return "CuI"
        if 'br' in s_low:
            return "CuBr"
        return "Cu"
    if re.search(r'\bk\b', s_low):
        return "K+"
    if re.search(r'\bna\b', s_low):
        return "Na+"
    if re.search(r'\bli\b', s_low):
        return "Li+"
    # fallback- do not return raw reagents, return empty so it is excluded
    return ""

def find_agent_cols(df_columns):
    cols = []
    for c in df_columns:
        m = RE_AGENT_COL.match(c)
        if m:
            cols.append((int(m.group(1)), c))
    cols.sort()
    return [c for _, c in cols]

def process_chunk(df_chunk, agent_cols):
    rows_out = []
    for _, row in df_chunk.iterrows():
        picked = ""
        for c in agent_cols:
            raw = row.get(c, None)
            norm = canonicalize_agent(raw)
            if norm and norm in WHITELIST:
                picked = norm
                break
        # add agent_norm (empty string if nothing valid)
        new_row = row.copy()
        new_row['agent_norm'] = picked
        rows_out.append(new_row)
    if rows_out:
        return pd.DataFrame(rows_out)
    else:
        return pd.DataFrame(columns=list(df_chunk.columns) + ['agent_norm'])

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--chunksize", type=int, default=5000)
    p.add_argument("--max-rows", type=int, default=0, help="If >0, limit input rows for a quick test")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force-overwrite", action="store_true")
    args = p.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    if not in_path.exists():
        print("Input not found:", in_path); return

    if out_path.exists() and not args.force_overwrite and not args.dry_run:
        print("Output exists. Use --force-overwrite to replace:", out_path); return

    reader = pd.read_csv(in_path, dtype=str, low_memory=True, chunksize=args.chunksize)
    first_write = True
    total_in = 0
    total_out = 0
    agent_counts = Counter()
    start = time.time()
    chunk_idx = 0

    for chunk in reader:
        chunk_idx += 1
        if chunk_idx == 1:
            agent_cols = find_agent_cols(chunk.columns)
            if not agent_cols:
                agent_cols = [c for c in ['agent_000','agent_001','agent_002'] if c in chunk.columns]
            print("Detected agent columns:", agent_cols)

        if args.max_rows and total_in >= args.max_rows:
            break
        if args.max_rows:
            remain = args.max_rows - total_in
            if len(chunk) > remain:
                chunk = chunk.iloc[:remain]

        df_out = process_chunk(chunk, agent_cols)
        total_in += len(chunk)
        total_out += len(df_out)
        for v in df_out['agent_norm'].fillna('').tolist():
            if v:
                agent_counts[v] += 1

        if not args.dry_run:
            df_out.to_csv(out_path, mode='a', index=False, header=first_write)
            first_write = False

        elapsed = time.time() - start
        rate = total_in / elapsed if elapsed > 0 else 0.0
        print(f"Chunk {chunk_idx}: in={len(chunk)} out={len(df_out)} totals_in={total_in} totals_out={total_out} rate={rate:.1f} r/s")

    print("Done. Total in:", total_in, "Total out rows:", total_out)
    print("Top whitelist agents observed:")
    for a,c in agent_counts.most_common(50):
        print(f"  {a} : {c}")
    print("Output file:", out_path, "dry_run=", args.dry_run)

if __name__ == "__main__":
    main()
