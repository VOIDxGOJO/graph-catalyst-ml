#!/usr/bin/env python3
"""
if string contains >> and a plausible smiles substring, keep best match
else if single >, convert first single > to >>
strip non-printable / obv non-SMILES text
keep only allowed smiles chars (letters, digits, []()+-=#@/.,%:)
write o/p csv with new column reaction_smiles

small test-
 python scripts/clean_rxn_str.py --input ./data/orderly_condition_with_rare_train.csv --output ./data/orderly_condition_with_rare_train_cleaned.csv --chunksize 5000 --max-rows 5000 --force-overwrite
"""
import argparse
import pandas as pd
import re
from pathlib import Path
import time
from collections import Counter

# allowed chars
RE_ALLOWED = re.compile(r'[^0-9A-Za-z\[\]\+\-\(\)\.=#:\/%@\.>]')
RE_MULTI_GT = re.compile(r'>{2,}')
RE_SINGLE_GT = re.compile(r'(?<!>)>(?!>)')  # match single > not part of '>>'
RE_ATOM_MAP = re.compile(r':\d+')  # remove atom mapping like :1
MAX_LEN = 800

# find plausible reaction pattern "smth>>smth"
RE_RXN_LIKE = re.compile(r'([0-9A-Za-z\[\]\+\-\(\)\.=#:\/%@\.]+>>[0-9A-Za-z\[\]\+\-\(\)\.=#:\/%@\.]+)')

def clean_one(s):
    if s is None:
        return ""
    s0 = str(s).strip()
    if s0 == "" or s0 == "\\N":
        return ""

    s1 = RE_ALLOWED.sub(' ', s0)    # replace suspicious chars with space
    s1 = ' '.join(s1.split())       # collapse whitespace
    # look for rxn like substring first
    m = RE_RXN_LIKE.search(s1)
    if m:
        cand = m.group(1)
        cand = cand.strip()
    else:
        # convert single '>' to '>>' for simple cases like "A>B" -> "A>>B"
        if '>' in s1 and '>>' not in s1:
            # replace first single '>' with '>>'
            cand = RE_SINGLE_GT.sub('>>', s1, count=1)
        else:
            cand = s1
            
    # remove accidental multiple '>' and make exactly '>>'
    cand = RE_MULTI_GT.sub('>>', cand)
    # remove atom mapping like [C:1] -> [C]
    cand = RE_ATOM_MAP.sub('', cand)
    # trim and enforce max length
    cand = cand.strip()
    if len(cand) > MAX_LEN:
        return ""
    # ensure it contains either '>>' or at least '.' (fragment)
    if '>>' not in cand and '.' not in cand:

        # but wrap as reactants>>product where left is empty (single smiles reactant or smth)
        if len(cand) > 0 and any(ch.isalpha() for ch in cand):
            cand = cand
        else:
            return ""
    return cand

def process_stream(in_path, out_path, chunksize=5000, max_rows=0, dry_run=False, force_overwrite=False):
    in_path = Path(in_path)
    out_path = Path(out_path)
    if not in_path.exists():
        print("Input not found:", in_path); return
    if out_path.exists() and not force_overwrite and not dry_run:
        print("Output exists. Use --force-overwrite to overwrite:", out_path); return

    reader = pd.read_csv(in_path, dtype=str, low_memory=True, chunksize=chunksize)
    first_write = True
    total = 0
    kept = 0
    kept_examples = []
    bad_prefix_counter = Counter()
    start = time.time()
    chunk_i = 0

    for df in reader:
        chunk_i += 1
        if max_rows and total >= max_rows:
            break
        if max_rows:
            remain = max_rows - total
            if len(df) > remain:
                df = df.iloc[:remain]

        # determine column to look at- prefer rxn_str, reaction_smiles, smiles etc
        col_candidates = ['reaction_smiles','rxn_str','smiles','rxn_str_canonical','rxn']
        found = None
        for c in col_candidates:
            if c in df.columns:
                found = c
                break
        if found is None:
            # fallback to first column that looks like it contains >
            for c in df.columns:
                if df[c].astype(str).str.contains('>').any():
                    found = c
                    break
        if found is None:
            # take first column (last resort)
            found = df.columns[0]

        df['reaction_smiles'] = df[found].fillna('').astype(str).apply(clean_one)

        # stats
        total += len(df)
        nonempty = df['reaction_smiles'].astype(str).str.strip().replace('\\N','').astype(bool).sum()
        kept += int(nonempty)
        # sample examples
        for ex in df['reaction_smiles'].dropna().astype(str).head(5).tolist():
            if ex:
                kept_examples.append(ex)
        # collect invalid prefixes for diag
        for v in df[found].fillna('').astype(str).tolist():
            v0 = v.strip()
            if v0 and clean_one(v0) == "":
                bad_prefix_counter[v0[:80]] += 1

        # print whats happening
        if not dry_run:
            df.to_csv(out_path, mode='a', index=False, header=first_write)
            first_write = False

        elapsed = time.time() - start
        rate = total / elapsed if elapsed > 0 else 0.0
        print(f"[chunk {chunk_i}] in={len(df)} totals_processed={total} kept_so_far={kept} rate={rate:.1f} rows/s")

    print("Finished cleaning.")
    print("Total rows scanned:", total)
    print("Rows with non-empty reaction_smiles:", kept, f"({kept/total:.2%})")
    print("Sample kept reaction_smiles (up to 10):")
    for s in kept_examples[:10]:
        print("  ", s)
    print("\nTop 20 problematic original prefixes (helpful to tune cleaning):")
    for k,c in bad_prefix_counter.most_common(20):
        print(f"{c:6d}  {k}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--chunksize", type=int, default=5000)
    p.add_argument("--max-rows", type=int, default=0)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force-overwrite", action="store_true")
    args = p.parse_args()
    process_stream(args.input, args.output, chunksize=args.chunksize, max_rows=args.max_rows, dry_run=args.dry_run, force_overwrite=args.force_overwrite)

if __name__ == "__main__":
    main()
