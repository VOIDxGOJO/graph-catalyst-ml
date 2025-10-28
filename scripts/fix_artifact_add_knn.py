
"""
load artifact, check if clf_agent is a classifier
if missing/invalid, build a lightweight KNeighborsClassifier using X_fp_for_nn and df_for_nn labels,
attach it as clf_agent and save artifact back
"""

import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

def is_classifier(obj):
    # minimal duck-typing: needs predict
    return obj is not None and hasattr(obj, "predict")

def infer_label_column(df, label_values):
    # find a column in df where most values are in label_values
    for col in df.columns:
        vals = df[col].astype(str).values
        # count how many are found in label_values
        in_count = np.isin(vals, label_values).sum()
        ratio = in_count / max(1, len(vals))
        if ratio > 0.7:  # heuristics: >70% matching -> likely label col
            return col
    return None

def main(src_path, out_path=None):
    src = Path(src_path)
    if not src.exists():
        print("Artifact not found:", src)
        return 2
    out = src if out_path is None else Path(out_path)
    print("Loading artifact:", src)
    art = joblib.load(str(src))
    if not isinstance(art, dict):
        print("Artifact not a dict; aborting.")
        return 3

    # quick keys check
    print("Keys:", list(art.keys()))
    clf_agent = art.get("clf_agent")
    agent_le = art.get("agent_le")
    X_fp = art.get("X_fp_for_nn")
    df_for_nn = art.get("df_for_nn")

    if is_classifier(clf_agent):
        print("clf_agent already looks like a classifier (has predict). Nothing to do.")
        return 0

    # confirm we have X + df
    if X_fp is None or df_for_nn is None or agent_le is None:
        print("Missing required pieces: X_fp_for_nn, df_for_nn or agent_le. Cannot build classifier.")
        return 4

    # ensure X is numpy array
    X = np.asarray(X_fp)
    print("X_fp_for_nn shape:", X.shape, "dtype:", X.dtype)

    # find label column in df_for_nn
    label_vals = list(agent_le.classes_.astype(str)) if hasattr(agent_le, "classes_") else []
    print("Agent LE has classes (first 20):", label_vals[:20])
    label_col = infer_label_column(df_for_nn, label_vals)
    if label_col is None:
        # try common names
        for c in ["agent_norm", "agent", "agent_000", "label"]:
            if c in df_for_nn.columns:
                label_col = c
                break
    if label_col is None:
        print("Failed to infer label column in df_for_nn. Columns:", list(df_for_nn.columns))
        return 5

    print("Using label column:", label_col)
    y_raw = df_for_nn[label_col].astype(str).values
    # transform to encoded integers using agent_le
    try:
        y_enc = agent_le.transform(y_raw)
    except Exception as e:
        # fallback: map by creating dict mapping from string to index
        mapping = {str(v): i for i, v in enumerate(agent_le.classes_)}
        y_enc = np.array([mapping.get(str(v), -1) for v in y_raw], dtype=int)
        mask = y_enc >= 0
        X = X[mask]
        y_enc = y_enc[mask]
        print("Some labels were unknown to agent_le; those rows were skipped. Kept rows:", X.shape[0])

    # train a lightweight KNN classifier
    print("Training KNeighborsClassifier (n_neighbors=5, metric='cosine') on sampled fingerprints ...")
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=5, metric='cosine', n_jobs=1)
    knn.fit(X, y_enc)
    print("KNN trained.")

    # attach to artifact
    art["clf_agent"] = knn
    # optional: add a small metadata note
    meta = art.get("train_metadata", {})
    meta = dict(meta)
    meta["_fix_applied"] = "added_knn_from_X_fp_for_nn"
    art["train_metadata"] = meta

    # save out
    print("Saving artifact to:", out)
    joblib.dump(art, str(out), compress=3)
    print("Saved. Done.")
    return 0

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/fix_artifact_add_knn.py <artifact_in.joblib> [artifact_out.joblib]")
        sys.exit(1)
    src = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else None
    sys.exit(main(src, out))
