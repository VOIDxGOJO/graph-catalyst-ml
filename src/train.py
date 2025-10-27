
"""
balanced partial-fit trainer for catalyst (agent) classification
does internal per-class downsampling, auto-aggregates agents,
prints a full classification report on test set

run-
python -m src.train --train-csv path/to/train.csv --test-csv path/to/test.csv --out models/artifacts_balanced2.joblib --nbits 128 --cap-per-class 1000 --chunk-size 1000 --random-state 42 --exclude-other

will try to use rxn_str or reaction_smiles as the smiles column
will assemble agent labels using 'agent_norm' if present, otherwise concat agent_000..agent_002
"""

import argparse
import os
from pathlib import Path
import joblib
import math
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from src.data import reaction_to_fp, load_orderly_csv

def safe_read_csv(path):
    # read csv (pandas default engine; caller should ensure file is cleaned)
    return pd.read_csv(path, dtype=str, low_memory=False)

def assemble_agent_norm(df):
    # if agent_norm exists and looks populated, use it; otherwise, combine agent_000..agent_002
    if 'agent_norm' in df.columns:
        nonempty = df['agent_norm'].fillna('').astype(str).str.strip() != ''
        if nonempty.sum() > 0:
            return df['agent_norm'].fillna('').astype(str)
    # fallback- join agent_000..agent_002 first non-empty

    agent_cols = [c for c in ['agent_000','agent_001','agent_002'] if c in df.columns]
    if not agent_cols:
        # no agent columns at all- create empty series
        return pd.Series([''] * len(df), index=df.index)
    def choose_row(r):
        for c in agent_cols:
            v = r.get(c, '')
            if pd.isna(v):
                continue
            s = str(v).strip()
            if s != '' and s != '\\N':
                return s
        return ''
    return df.apply(choose_row, axis=1)

def build_classifier(random_state=42, class_weight=None):
    # try log_loss then log
    try:
        return SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3, random_state=random_state, class_weight=class_weight)
    except Exception:
        return SGDClassifier(loss='log', max_iter=1000, tol=1e-3, random_state=random_state, class_weight=class_weight)

def sample_balanced(df, label_col='agent_norm', cap=2000, random_state=42, exclude_other=False):
    df = df.copy()
    if exclude_other:
        df[label_col] = df[label_col].fillna('').astype(str)
        df = df[df[label_col].str.strip() != '']
    # group and sample up to cap per class
    parts = []
    rng = np.random.RandomState(random_state)
    counts = {}
    for label, g in df.groupby(label_col):
        n = len(g)
        counts[label] = n
        take = min(n, cap)
        if n <= take:
            parts.append(g)
        else:
            parts.append(g.sample(n=take, random_state=random_state))
    if not parts:
        return df.iloc[0:0], counts
    out = pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return out, counts

def fp_stack(smiles_series, nBits):
    # safe vstack of fingerprints
    fps = []
    for s in smiles_series.fillna('').astype(str):
        fps.append(reaction_to_fp(s, nBits=nBits))
    return np.vstack(fps) if fps else np.zeros((0, nBits), dtype=np.float32)

def main(args):
    print("Loading train CSV:", args.train_csv)
    df_train = safe_read_csv(args.train_csv)
    print("Train rows:", len(df_train))
    # assemble label column
    if 'agent_norm' in df_train.columns and df_train['agent_norm'].notna().sum() > 0:
        df_train['agent_norm'] = df_train['agent_norm'].fillna('').astype(str)
    else:
        df_train['agent_norm'] = assemble_agent_norm(df_train)
    # identify smiles column
    smiles_col = None
    for cand in ['rxn_str', 'reaction_smiles', 'rxn_smiles', 'rxn_str_norm']:
        if cand in df_train.columns:
            smiles_col = cand
            break
    if smiles_col is None:
        raise SystemExit("No SMILES column detected in training CSV. Expected one of rxn_str/reaction_smiles.")
    print("Using SMILES column for training:", smiles_col)

    # balance sampling (cap per class)
    print(f"Sampling balanced dataset with cap_per_class={args.cap_per_class}, exclude_other={args.exclude_other}")
    df_bal, orig_counts = sample_balanced(df_train, label_col='agent_norm', cap=args.cap_per_class, random_state=args.random_state, exclude_other=args.exclude_other)
    print("Original per-class counts (sample):")

    # top 20 classes counts
    for k, v in sorted(orig_counts.items(), key=lambda x: -x[1])[:30]:
        print(f"  {k}: {v}")
    print("Balanced dataset rows:", len(df_bal))

    # prepare label encoder
    y_vals = df_bal['agent_norm'].fillna('<<NULL>>').astype(str).values
    agent_le = LabelEncoder()
    agent_le.fit(y_vals)
    n_classes = len(agent_le.classes_)
    print("Label classes:", n_classes)

    # build classifier (not usign class_weight since weve balanced by sampling)
    clf = build_classifier(random_state=args.random_state, class_weight=None)

    # partial_fit training in chunks
    chunk_size = args.chunk_size
    print(f"Starting partial_fit over chunks (chunk_size={chunk_size}) ...")
    for start in range(0, len(df_bal), chunk_size):
        chunk = df_bal.iloc[start:start+chunk_size]
        X_chunk = fp_stack(chunk[smiles_col], nBits=args.nbits)
        y_chunk = agent_le.transform(chunk['agent_norm'].astype(str).values)
        if not hasattr(clf, "classes_") or getattr(clf, "classes_", None) is None or clf.classes_.size == 0:
            clf.partial_fit(X_chunk, y_chunk, classes=np.arange(n_classes))
        else:
            clf.partial_fit(X_chunk, y_chunk)

    print("Finished training.")

    # build small NN index sampled from balanced df
    try:
        from sklearn.neighbors import NearestNeighbors
        nn_sample = min(args.nn_sample, len(df_bal))
        if nn_sample > 0:
            idx = np.random.RandomState(args.random_state).choice(len(df_bal), size=nn_sample, replace=False)
            X_fp_for_nn = fp_stack(df_bal.iloc[idx][smiles_col], nBits=args.nbits)
            df_for_nn = df_bal.iloc[idx].reset_index(drop=True)
            nn_index = NearestNeighbors(n_neighbors=min(6, len(X_fp_for_nn)), metric='cosine', algorithm='brute')
            nn_index.fit(X_fp_for_nn)
            print("Built NN index on sampled balanced fingerprints:", len(X_fp_for_nn))
        else:
            nn_index = None
            X_fp_for_nn = None
            df_for_nn = None
    except Exception as e:
        print("NearestNeighbors failed:", e)
        nn_index = None
        X_fp_for_nn = None
        df_for_nn = None

    # evaluate on test set if provided
    artifacts = {
        "clf_agent": clf,
        "agent_le": agent_le,
        "nn_index": nn_index,
        "X_fp_for_nn": X_fp_for_nn,
        "df_for_nn": df_for_nn,
        "train_metadata": {"train_rows_used": len(df_bal), "nbits": args.nbits, "cap_per_class": args.cap_per_class}
    }

    if args.test_csv and os.path.exists(args.test_csv):
        print("Loading test CSV:", args.test_csv)
        df_test = safe_read_csv(args.test_csv)
        # assemble labels on test too
        if 'agent_norm' not in df_test.columns or df_test['agent_norm'].notna().sum() == 0:
            df_test['agent_norm'] = assemble_agent_norm(df_test)
        # select test rows that have labels present in encoder
        test_mask = df_test['agent_norm'].fillna('').astype(str).str.strip() != ''
        df_test = df_test[test_mask].reset_index(drop=True)
        if len(df_test) == 0:
            print("No labeled rows in test file; skipping test eval.")
        else:
            # only evaluate on labels present in train encoder
            train_classes = set(agent_le.classes_.astype(str))
            test_mask_known = df_test['agent_norm'].astype(str).isin(train_classes)
            if not test_mask_known.any():
                print("No test rows have labels seen during training; skipping eval.")
            else:
                df_test_known = df_test[test_mask_known].reset_index(drop=True)
                X_test = fp_stack(df_test_known[smiles_col], nBits=args.nbits)
                y_test = agent_le.transform(df_test_known['agent_norm'].astype(str).values)
                y_pred = clf.predict(X_test)
                print("Classification report on test subset (labels present in training):")
                print(classification_report(y_test, y_pred, zero_division=0, target_names=agent_le.classes_))
                acc = accuracy_score(y_test, y_pred)
                print("Accuracy (subset):", acc)
                cm = confusion_matrix(y_test, y_pred)
                print("Confusion matrix shape:", cm.shape)
                artifacts['eval'] = {"accuracy": acc, "report": classification_report(y_test, y_pred, zero_division=0), "confusion_matrix": cm}
    else:
        print("No test CSV provided or path does not exist; skipping test eval.")

    # save artifact
    out_dir = os.path.dirname(args.out) or "."
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    joblib.dump(artifacts, args.out)
    print("Saved artifact to:", args.out)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train-csv", required=True)
    p.add_argument("--test-csv", required=False, default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--nbits", type=int, default=128)
    p.add_argument("--chunk-size", type=int, default=1000)
    p.add_argument("--nn-sample", type=int, default=1000)
    p.add_argument("--cap-per-class", type=int, default=2000, help="Max samples to keep per class (downsample majority)")
    p.add_argument("--exclude-other", action="store_true", help="Drop empty/OTHER labels before sampling")
    p.add_argument("--random-state", type=int, default=42)
    args = p.parse_args()
    main(args)
