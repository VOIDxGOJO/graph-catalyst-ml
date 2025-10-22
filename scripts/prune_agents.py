import argparse
import pandas as pd
from pathlib import Path
from collections import Counter

def prune_file(input_csv, output_csv, top_k=500, agent_col='agent_000'):
    df = pd.read_csv(input_csv, dtype=str, low_memory=False)
    vals = df[agent_col].fillna('').astype(str)
    cnt = Counter([v for v in vals if v.strip() != '' and v != '\\N'])
    top = set([a for a,_ in cnt.most_common(top_k)])
    def map_agent(x):
        if pd.isna(x) or str(x).strip() == '' or x == '\\N':
            return ''
        if x in top:
            return x
        return 'OTHER'
    df['agent_pruned'] = df[agent_col].apply(map_agent)

    # also copy agent_pruned back to agent_000 to reuse training pipeline
    df['agent_000'] = df['agent_pruned']
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Wrote pruned CSV: {output_csv}. Kept top-{top_k} agents + OTHER")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--top-k", type=int, default=500)
    args = parser.parse_args()
    prune_file(args.input, args.output, top_k=args.top_k)
