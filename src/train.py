"""

partial_fit with SGDClassifier multiclass logistic to train in chunks
builds sampled NearestNeighbors index for retrieval
saves artifacts needed by server

python -m src.train --train-csv ".\data\orderly_condition_with_rare_train_normalized.csv" --test-csv ".\data\orderly_condition_with_rare_test_normalized.csv" --out ".\models\artifacts_partial.joblib" --nbits 128 --max-rows 20000 --nn-sample 5000
"""
import argparse
import os
import math
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm

# local featurizer
from src.data import featurize_df, reaction_to_fp, load_orderly_csv

# helpers 
def chunked_df(df, chunk_size):
    # successive chunks of dataframe by row
    for start in range(0, len(df), chunk_size):
        yield df.iloc[start:start+chunk_size]

# safe numeric
def safe_float(x):
    try:
        if x is None:
            return None
        f = float(x)
        return f if math.isfinite(f) else None
    except Exception:
        return None

# TRENN
def train_and_save(train_csv, test_csv, out_path,
                   fp_nbits=128,
                   max_rows=None,
                   chunk_size=2000,
                   nn_sample=5000,
                   random_state=42):

    print("Loading CSV (train)...")
    df_train = load_orderly_csv(train_csv)
    if test_csv and os.path.exists(test_csv):
        print("Loading CSV (test)...")
        df_test = load_orderly_csv(test_csv)
    else:
        df_test = None

    if max_rows is not None and len(df_train) > max_rows:
        print(f"Sampling {max_rows} rows from training set (random_state={random_state})")
        df_train = df_train.sample(n=max_rows, random_state=random_state).reset_index(drop=True)

    print(f"Unique agents in train: {df_train['agent'].fillna('<<NULL>>').nunique()}")

    agent_le = LabelEncoder()
    agent_vals = df_train['agent'].fillna('<<NULL>>').astype(str).values
    agent_le.fit(agent_vals)
    classes = agent_le.classes_
    print(f"Number of agent classes: {len(classes)}")

    solvent_le = LabelEncoder()
    solvent_vals = df_train['solvent'].fillna('<<NULL>>').astype(str).values
    solvent_le.fit(solvent_vals)

    clf_agent = SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3, random_state=random_state)
    clf_solvent = SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3, random_state=random_state)

    agent_class_indices = np.arange(len(agent_le.classes_))
    solvent_class_indices = np.arange(len(solvent_le.classes_))

    sampled_fp_rows = []
    sampled_df_rows = []

    print("Training classifiers with partial_fit in chunks...")

    # iterate chunks of training df
    for chunk in tqdm(list(chunked_df(df_train, chunk_size)), desc="train-chunks"):
        X_chunk = np.vstack([reaction_to_fp(smi, radius=2, nBits=fp_nbits) for smi in chunk['smiles'].fillna('')])

        y_agent_chunk = agent_le.transform(chunk['agent'].fillna('<<NULL>>').astype(str).values)
        y_solvent_chunk = solvent_le.transform(chunk['solvent'].fillna('<<NULL>>').astype(str).values)


        if not hasattr(clf_agent, 'coef_') or clf_agent.coef_.size == 0:
            clf_agent.partial_fit(X_chunk, y_agent_chunk, classes=agent_class_indices)
        else:
            clf_agent.partial_fit(X_chunk, y_agent_chunk)

        if not hasattr(clf_solvent, 'coef_') or clf_solvent.coef_.size == 0:
            clf_solvent.partial_fit(X_chunk, y_solvent_chunk, classes=solvent_class_indices)
        else:
            clf_solvent.partial_fit(X_chunk, y_solvent_chunk)

       
        rng = np.random.default_rng(random_state)
        take = min(len(chunk), max(0, int(round(len(chunk) * (nn_sample / max(len(df_train), 1))))))  # proportionally
        if take > 0:
            idx = rng.choice(len(chunk), size=take, replace=False)
            sampled_fp_rows.append(X_chunk[idx])
            sampled_df_rows.append(chunk.iloc[idx])

    print("Finished partial-fit training")

    if sampled_fp_rows:
        X_fp_for_nn = np.vstack(sampled_fp_rows)
        df_for_nn = pd.concat(sampled_df_rows, ignore_index=True).reset_index(drop=True)
    else:
        X_fp_for_nn = None
        df_for_nn = None

    if df_test is not None:
        print("Evaluating on test set (skipping unseen agent labels)...")
        X_test_rows = []
        for chunk in chunked_df(df_test, chunk_size):
            X_test_rows.append(np.vstack([reaction_to_fp(smi, radius=2, nBits=fp_nbits) for smi in chunk['smiles'].fillna('')]))
        X_test = np.vstack(X_test_rows) if X_test_rows else np.zeros((0, fp_nbits))

        test_agent_raw = df_test['agent'].fillna('<<NULL>>').astype(str).values
        mask_in_train = np.array([ (lbl in set(agent_le.classes_)) for lbl in test_agent_raw ])
        if mask_in_train.any():
            y_true = agent_le.transform(test_agent_raw[mask_in_train])
            y_pred = clf_agent.predict(X_test[mask_in_train])
            print("Agent classification report (test subset with known labels):")
            print(classification_report(y_true, y_pred, zero_division=0))
            print("Agent test accuracy (subset):", accuracy_score(y_true, y_pred))
        else:
            print("No test labels matched training classes; skipping agent eval.")

    # build NN index on sampled fps if available
    nn_index = None
    if X_fp_for_nn is not None and len(X_fp_for_nn) > 0:
        try:
            n_neighbors = min(6, len(X_fp_for_nn))
            nn_index = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine', algorithm='brute')
            nn_index.fit(X_fp_for_nn)
            print("Built NearestNeighbors index on sampled fingerprints (size=%d)." % len(X_fp_for_nn))
        except Exception as e:
            print("NearestNeighbors build failed:", e)
            nn_index = None

    artifacts = {
        "clf_agent": clf_agent,
        "agent_le": agent_le,
        "clf_solvent": clf_solvent,
        "solvent_le": solvent_le,
        "nn_index": nn_index,
        "X_fp_for_nn": X_fp_for_nn,
        "df_for_nn": df_for_nn,

        "train_metadata": {
            "train_rows": len(df_train),
            "nbits": fp_nbits
        }
    }

    Path(os.path.dirname(out_path) or ".").mkdir(parents=True, exist_ok=True)
    joblib.dump(artifacts, out_path, compress=3)
    print("Saved artifacts to:", out_path)
    return artifacts


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train-csv", required=True)
    p.add_argument("--test-csv", default=None)
    p.add_argument("--out", default="models/artifacts_partial.joblib")
    p.add_argument("--nbits", type=int, default=128)
    p.add_argument("--max-rows", type=int, default=20000, help="Max train rows to use (None=all)")
    p.add_argument("--chunk-size", type=int, default=2000, help="Rows per partial_fit chunk")
    p.add_argument("--nn-sample", type=int, default=5000, help="Rows to sample for NN index (0 = disabled)")
    args = p.parse_args()

    mt = args.max_rows if args.max_rows is not None else None
    train_and_save(args.train_csv, args.test_csv, args.out, fp_nbits=args.nbits, max_rows=mt, chunk_size=args.chunk_size, nn_sample=args.nn_sample)
