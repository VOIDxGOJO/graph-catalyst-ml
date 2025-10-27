#!/usr/bin/env python3
"""

reads input CSV in chunks (pandas.read_csv with chunksize)
for each chunk, find agent_* columns, canonicalizes agents, and
for each non-empty agent produce one output row (same reaction, diff agent_norm)
write output incrementally with DataFrame.to_csv(mode='a'), to keep memory low
prints progress lines per chunk and summary at the end

run-
python scripts/normalize_and_explode_agents.py --input ./data/orderly_condition_with_rare_train.csv \
                                              --output ./data/orderly_condition_with_rare_train_exploded.csv \
                                              --chunksize 5000 --max-rows 0
"""
import argparse
import pandas as pd
import re
import sys
import time
from pathlib import Path
from collections import Counter

CANONICAL_MAP = {
    "pd(pph3)4": "Pd(PPh3)4",
    "pd(pph3)2cl2": "Pd(PPh3)2Cl2",
    "pd2(dba)3": "Pd2(dba)3",
    "pd2(dba)3·ch": "Pd2(dba)3",
    "pd": "Pd",
    "[pt]": "Pt",
    "pt": "Pt",
    "cui": "CuI",
    "cubr": "CuBr",
    "cucl2": "CuCl2",
    "nicl2": "NiCl2",
    "ni": "Ni",
    "[k+]": "K+",
    "k+": "K+",
    "[na+]": "Na+",
    "na+": "Na+",
    "cl": "Cl",
    "cl-": "Cl",
    "[h-]": "H-",
    "h-": "H-",
    "p(tbu)3": "P(tBu)3",
    "p(p h)3": "PPh3",
    "triphenylphosphine": "PPh3",
    "triethylamine": "Et3N",
}

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
    if s_low in CANONICAL_MAP:
        return CANONICAL_MAP[s_low]
    s_stripped = RE_NON_ALPHANUM.sub('', s_low)
    if s_stripped in CANONICAL_MAP:
        return CANONICAL_MAP[s_stripped]
    if 'pd' in s_low and 'pph3' in s_low:
        return "Pd(PPh3)4"
    if 'pd' in s_low:
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
    return s.strip()

def find_agent_cols(df_columns):
    cols = []
    for c in df_columns:
        m = RE_AGENT_COL.match(c)
        if m:
            cols.append((int(m.group(1)), c))
    cols.sort()
    return [c for _, c in cols]

def process_chunk(df_chunk, agent_cols):
    """Return a DataFrame of exploded rows for this chunk with agent_norm column."""
    rows_out = []
    for _, row in df_chunk.iterrows():
        found = False
        for c in agent_cols:
            raw = row.get(c, None)
            norm = canonicalize_agent(raw)
            if norm:
                new_row = row.copy()
                new_row['agent_norm'] = norm
                rows_out.append(new_row)
                found = True
        if not found:
            # keep single row with empty agent_norm
            new_row = row.copy()
            new_row['agent_norm'] = ""
            rows_out.append(new_row)
    if rows_out:
        df_out = pd.DataFrame(rows_out)
    else:
        df_out = pd.DataFrame(columns=list(df_chunk.columns) + ['agent_norm'])
    return df_out

def main(args):
    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.exists():
        print(f"ERROR: input file not found: {input_path}", file=sys.stderr)
        sys.exit(2)

    if output_path.exists() and not args.force_overwrite and not args.dry_run:
        print(f"ERROR: output file already exists: {output_path}. Use --force-overwrite to overwrite.", file=sys.stderr)
        sys.exit(2)

    chunksize = int(args.chunksize)
    max_rows = int(args.max_rows) if args.max_rows else 0
    dry = args.dry_run

    start = time.time()
    total_in = 0
    total_out = 0
    per_agent_counter = Counter()
    first_write = True

    # streaming read
    print(f"Starting streaming read chunksize={chunksize} max_rows={max_rows} dry_run={dry}")
    reader = pd.read_csv(input_path, dtype=str, low_memory=True, chunksize=chunksize)

    chunk_idx = 0
    for chunk in reader:
        chunk_idx += 1
        if chunk_idx == 1:
            # discover agent cols on first chunk (works if header present)
            agent_cols = find_agent_cols(chunk.columns)
            if not agent_cols:
                # fallback to common names if none detected
                agent_cols = [c for c in ['agent_000','agent_001','agent_002'] if c in chunk.columns]
            print(f"Detected agent columns: {agent_cols}")

        # optional limit by max_rows
        if max_rows and total_in >= max_rows:
            print("Reached max_rows limit, breaking.")
            break

        # if chunk would exceed max_rows, trim it
        if max_rows:
            remain = max_rows - total_in
            if remain <= 0:
                break
            if len(chunk) > remain:
                chunk = chunk.iloc[:remain]

        df_out = process_chunk(chunk, agent_cols)

        # update counters
        total_in += len(chunk)
        total_out += len(df_out)
        for v in df_out['agent_norm'].fillna('').tolist():
            if v:
                per_agent_counter[v] += 1

        # write out
        if not dry:
            # write header on first write, append later
            write_header = first_write
            df_out.to_csv(output_path, mode='a', index=False, header=write_header)
            first_write = False

        # progress print
        elapsed = time.time() - start
        rate = total_in / elapsed if elapsed > 0 else 0.0
        print(f"Chunk {chunk_idx}: in_chunk={len(chunk)} out_chunk={len(df_out)} totals in={total_in} out={total_out} rate={rate:.1f} rows/sec elapsed={elapsed:.1f}s")

    # final summary
    print("Done processing.")
    print(f"Total input rows processed: {total_in}")
    print(f"Total output rows written: {total_out}")
    top_agents = per_agent_counter.most_common(20)
    print("Top agents (sample):")
    for a, c in top_agents:
        print(f"  {a} : {c}")
    print(f"Output file: {output_path} (dry_run={dry})")
    print(f"Elapsed time: {time.time() - start:.1f}s")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--chunksize", type=int, default=5000)
    p.add_argument("--max-rows", type=int, default=0, help="If >0, stop after this many input rows (testing)")
    p.add_argument("--dry-run", action="store_true", help="Do not write output file; only simulate")
    p.add_argument("--force-overwrite", action="store_true", help="Overwrite existing output file")
    args = p.parse_args()
    main(args)
