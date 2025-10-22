import argparse
import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors

from src.data import load_orderly_csv, featurize_df

def train(csv_path: str, out_path: str, n_estimators: int = 200):
    print("Loading dataset..")
    df = load_orderly_csv(csv_path)
    print(f" Loaded {len(df)} rows")

    # filter rows that have agent (agent_000), need agent for supervised train
    df_agent = df[df['agent'].notna() & (df['agent'].astype(str).str.strip() != '')].reset_index(drop=True)
    print(f" Rows with agent available: {len(df_agent)}")

    # featurize entire df (need fingerprints for retrieval over full df)
    X_fp_all, artifacts = featurize_df(df)
    agent_le = artifacts['agent_le']
    solvent_le = artifacts['solvent_le']

    # build agent classifier only on rows that have agent labels (non-null)
    idx_agent = df.index[df['agent'].notna() & (df['agent'].astype(str).str.strip() != '')].tolist()
    if len(idx_agent) == 0:
        raise RuntimeError("No agent labels found in dataset (agent_000). Cannot train classifier.")

    X_agent = X_fp_all[idx_agent]
    y_agent = agent_le.transform(df.loc[idx_agent, 'agent'].fillna('<<NULL>>').astype(str).values)

    print("Training agent classifier (RandomForest)...")
    clf_agent = RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1, random_state=42)
    clf_agent.fit(X_agent, y_agent)
    print(" Agent classifier trained.")

    # build solvent classifier if solvents exist
    idx_solvent = df.index[df['solvent'].notna() & (df['solvent'].astype(str).str.strip() != '')].tolist()
    clf_solvent = None
    if len(idx_solvent) > 0:
        X_solvent = X_fp_all[idx_solvent]
        y_solvent = solvent_le.transform(df.loc[idx_solvent, 'solvent'].fillna('<<NULL>>').astype(str).values)
        print("Training solvent classifier (RandomForest)...")
        clf_solvent = RandomForestClassifier(n_estimators=max(50, n_estimators//4), n_jobs=-1, random_state=42)
        clf_solvent.fit(X_solvent, y_solvent)
        print(" Solvent classifier trained.")
    else:
        print("No solvent labels found; skipping solvent classifier.")

    # build NearestNeighbors index for retrieval (use cosine on binary fp)
    print("Building NearestNeighbors index for retrieval...")
    nn = NearestNeighbors(n_neighbors=6, metric='cosine', algorithm='brute')  # 6 includes self at index 0
    nn.fit(X_fp_all)

    # pack artifacts
    packed = {
        'clf_agent': clf_agent,
        'clf_solvent': clf_solvent,
        'agent_le': agent_le,
        'solvent_le': solvent_le,
        'nn_index': nn,
        'X_fp_all': X_fp_all,
        'df': df  # standardized df for retrieval
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    joblib.dump(packed, out_path)
    print(f"Saved artifacts to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to ORDerly-condition CSV")
    parser.add_argument("--out", default="models/artifacts.joblib", help="Output path for artifacts")
    parser.add_argument("--n-est", type=int, default=200, help="n_estimators for random forest")
    args = parser.parse_args()
    train(args.csv, args.out, n_estimators=args.n_est)
