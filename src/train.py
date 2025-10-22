import argparse
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import classification_report, accuracy_score, top_k_accuracy_score
from src.data import load_orderly_csv, featurize_df
from pathlib import Path
import time

def set_thread_envs():
    # reduce BLAS threads to avoid extra memory
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

def maybe_load_or_compute_fps(df_all, fp_cache_path: str, radius: int, nBits: int, show_progress=True):
    p = Path(fp_cache_path)
    if p.exists():
        print(f"Loading cached fingerprints from {fp_cache_path} ...")
        data = joblib.load(fp_cache_path)
        X_fp = data['X_fp']
        if X_fp.shape[0] != len(df_all):
            print("Cached fingerprint length differs from current df length. Recomputing.")
        else:
            return X_fp
    # compute and save
    print("Computing fingerprints (this may take a while) ...")
    X_fp, artifacts = featurize_df(df_all, radius=radius, nBits=nBits, show_progress=show_progress)
    joblib.dump({'X_fp': X_fp, 'artifacts': artifacts}, fp_cache_path, compress=3)
    print(f"Saved fingerprints to {fp_cache_path}")
    return X_fp

def train(train_csv: str, test_csv: str, out_path: str,
          n_estimators: int = 100, fp_nbits: int = 1024,
          n_jobs: int = 1, max_depth: int = None, max_leaf_nodes: int = None,
          sample_rate: float = 1.0):
    set_thread_envs()

    print("Loading train CSV...")
    df_train = load_orderly_csv(train_csv)
    print(f" Train rows: {len(df_train)}")
    print("Loading test CSV...")
    df_test = load_orderly_csv(test_csv)
    print(f" Test rows: {len(df_test)}")

    df_all = pd.concat([df_train, df_test], ignore_index=True).reset_index(drop=True)
    print(f" Total rows (train+test): {len(df_all)}")

    fp_cache_path = "data/fps_cache.joblib"
    X_fp_all = maybe_load_or_compute_fps(df_all, fp_cache_path, radius=2, nBits=fp_nbits, show_progress=True)

    # indices for train/test
    n_train = len(df_train)
    n_test = len(df_test)
    idx_train = list(range(0, n_train))
    idx_test = list(range(n_train, n_train + n_test))

    # optionally sample train indices for quick runs
    if sample_rate is not None and sample_rate > 0 and sample_rate < 1.0:
        rng = np.random.RandomState(42)
        sampled = rng.choice(idx_train, size=int(len(idx_train) * sample_rate), replace=False)
        idx_train_sampled = sorted(list(sampled))
        print(f"Sampled {len(idx_train_sampled)} / {len(idx_train)} train rows (sample_rate={sample_rate})")
        idx_train_use = idx_train_sampled
    else:
        idx_train_use = idx_train

    # collect training rows that have agent labels
    train_idx_agent = [i for i in idx_train_use if df_all.loc[i, 'agent'] not in (None, '', 'nan') and str(df_all.loc[i, 'agent']).strip() != '']
    if len(train_idx_agent) == 0:
        raise RuntimeError("No agent labels found in train split. Cannot train agent classifier.")
    
    # reconstruct label encoder built in featurize_df artifacts if saved 
    agent_values = df_all['agent'].fillna('<<NULL>>').astype(str).values

    from sklearn.preprocessing import LabelEncoder
    agent_le = LabelEncoder()
    agent_le.fit(agent_values)

    X_agent = X_fp_all[train_idx_agent]
    y_agent = agent_le.transform(df_all.loc[train_idx_agent, 'agent'].fillna('<<NULL>>').astype(str).values)

    print(f"Training agent classifier on {len(train_idx_agent)} rows with {len(agent_le.classes_)} classes...")
    print(f"RF params: n_estimators={n_estimators}, n_jobs={n_jobs}, max_depth={max_depth}, max_leaf_nodes={max_leaf_nodes}")
    clf_agent = RandomForestClassifier(n_estimators=n_estimators, n_jobs=n_jobs, max_depth=max_depth, max_leaf_nodes=max_leaf_nodes, random_state=42)
    t0 = time.time()
    clf_agent.fit(X_agent, y_agent)
    t1 = time.time()
    print(f"Agent classifier trained in {t1-t0:.1f}s")

    # solvent classifier optional
    from sklearn.preprocessing import LabelEncoder as _LE
    solvent_le = _LE()
    solvent_vals = df_all['solvent'].fillna('<<NULL>>').astype(str).values
    solvent_le.fit(solvent_vals)
    train_idx_solvent = [i for i in idx_train_use if df_all.loc[i, 'solvent'] not in (None, '', 'nan') and str(df_all.loc[i, 'solvent']).strip() != '']
    clf_solvent = None
    if len(train_idx_solvent) > 0:
        X_sol = X_fp_all[train_idx_solvent]
        y_sol = solvent_le.transform(df_all.loc[train_idx_solvent, 'solvent'].fillna('<<NULL>>').astype(str).values)
        clf_solvent = RandomForestClassifier(n_estimators=max(20, n_estimators // 4), n_jobs=1, random_state=42, max_depth=max_depth)
        clf_solvent.fit(X_sol, y_sol)
        print("Solvent classifier trained.")
    else:
        print("No solvent labels found in train; skipping solvent classifier.")

    print("Building NearestNeighbors index for retrieval...")
    nn = NearestNeighbors(n_neighbors=6, metric='cosine', algorithm='brute')
    nn.fit(X_fp_all)

    # eval on test labeled agent rows only
    test_idx_agent = [i for i in idx_test if df_all.loc[i, 'agent'] not in (None, '', 'nan') and str(df_all.loc[i, 'agent']).strip() != '']
    if len(test_idx_agent) > 0:
        X_test = X_fp_all[test_idx_agent]
        y_test_true = agent_le.transform(df_all.loc[test_idx_agent, 'agent'].fillna('<<NULL>>').astype(str).values)
        y_test_pred = clf_agent.predict(X_test)
        print("Classification report (agent) on test split:")
        print(classification_report(y_test_true, y_test_pred, zero_division=0))
        acc = accuracy_score(y_test_true, y_test_pred)
        print(f" Test accuracy (agent): {acc:.4f}")
        try:
            proba = clf_agent.predict_proba(X_test)
            top5 = top_k_accuracy_score(y_test_true, proba, k=5)
            print(f" Test top-5 accuracy (agent): {top5:.4f}")
        except Exception:
            print("Top-5 accuracy not available.")
    else:
        print("No labeled agent rows in test split to evaluate.")

    packed = {
        'clf_agent': clf_agent,
        'clf_solvent': clf_solvent,
        'agent_le': agent_le,
        'solvent_le': solvent_le,
        'nn_index': nn,
        'X_fp_all': X_fp_all,
        'df': df_all
    }

    outdir = Path(out_path).parent
    outdir.mkdir(parents=True, exist_ok=True)
    joblib.dump(packed, out_path, compress=3)
    print(f"Saved artifacts to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", required=True, help="Path to train CSV")
    parser.add_argument("--test-csv", required=True, help="Path to test CSV")
    parser.add_argument("--out", default="models/artifacts.joblib", help="Output path for artifacts")
    parser.add_argument("--n-est", type=int, default=100, help="n_estimators for RF")
    parser.add_argument("--nbits", type=int, default=1024, help="Fingerprint nBits")
    parser.add_argument("--n-jobs", type=int, default=1, help="n_jobs for RF (use 1 to reduce memory)")
    parser.add_argument("--max-depth", type=int, default=None, help="max_depth for trees")
    parser.add_argument("--max-leaf-nodes", type=int, default=None, help="max_leaf_nodes for trees")
    parser.add_argument("--sample-rate", type=float, default=1.0, help="Fraction of train rows to sample for quick runs (0-1)")
    args = parser.parse_args()
    train(args.train_csv, args.test_csv, args.out, n_estimators=args.n_est, fp_nbits=args.nbits, n_jobs=args.n_jobs, max_depth=args.max_depth, max_leaf_nodes=args.max_leaf_nodes, sample_rate=args.sample_rate)
