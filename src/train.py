import argparse
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import classification_report, accuracy_score
from src.data import load_orderly_csv, featurize_df
import math
import warnings
warnings.filterwarnings("ignore")


def keep_top_k_agents(df, agent_col='agent', top_k=200):
    ser = df[agent_col].fillna("OTHER").astype(str)
    top = ser.value_counts().nlargest(top_k).index.tolist()
    df = df.copy()
    df[agent_col] = ser.apply(lambda x: x if x in top else "OTHER")
    return df, top


def sample_if_needed(df, max_rows=None, seed=42):
    if max_rows is None:
        return df.reset_index(drop=True)
    if len(df) <= max_rows:
        return df.reset_index(drop=True)
    return df.sample(n=max_rows, random_state=seed).reset_index(drop=True)


def map_test_agents_to_top(df_test_local, agent_le_local, candidate_cols=('agent', 'catalyst', 'agent_000')):

    if df_test_local is None:
        return None, None

    # canonical set of agents from label encoder
    try:
        top_agents_set = set([str(x) for x in agent_le_local.classes_])
    except Exception:
        top_agents_set = set()

    df_test_mapped = df_test_local.copy()

    # find source column
    src_col = None
    for c in candidate_cols:
        if c in df_test_mapped.columns:
            src_col = c
            break

    if src_col is not None:
        df_test_mapped['__agent_raw'] = df_test_mapped[src_col].fillna("").astype(str).str.strip()
        def _map_label(x):
            if x == "" or x.upper() in {"N\\A", "\\N", "NA", "NONE"}:
                return "OTHER"
            return x if (x in top_agents_set) else "OTHER"
        df_test_mapped['agent_mapped'] = df_test_mapped['__agent_raw'].apply(_map_label)
    else:
        # no agent column found, default to OTHER
        df_test_mapped['agent_mapped'] = ["OTHER"] * len(df_test_mapped)

    # safe transform to encoder integers
    try:
        y_true = agent_le_local.transform(df_test_mapped['agent_mapped'].values)
    except Exception as e:
        # fallback- map everything to OTHER index if possible
        print("Warning: agent_le.transform failed on test labels. Falling back to 'OTHER' index. Error:", e)
        try:
            other_idx = int(np.where(agent_le_local.classes_ == "OTHER")[0][0])
            y_true = np.full(len(df_test_mapped), other_idx, dtype=int)
        except Exception:
            y_true = np.zeros(len(df_test_mapped), dtype=int)

    return df_test_mapped, y_true


def train_and_save(train_csv, test_csv, out_path, fp_nbits=256, top_k_agents=200, max_train_sample=None, nn_sample=50000):
    print("Loading CSVs...")
    df_train = load_orderly_csv(train_csv)
    df_test = load_orderly_csv(test_csv) if (test_csv and os.path.exists(test_csv)) else None

    # keep top-K agents to limit classes
    print(f"Filtering to top-{top_k_agents} agents (others -> OTHER)")
    df_train_filtered, top_agents = keep_top_k_agents(df_train, agent_col='agent', top_k=top_k_agents)

    # optionally sample training data for speed/memory
    if max_train_sample is not None:
        print(f"Sampling training data to at most {max_train_sample} rows for faster training.")
        df_train_used = sample_if_needed(df_train_filtered, max_rows=max_train_sample)
    else:
        df_train_used = df_train_filtered.reset_index(drop=True)

    # combine for featurization (so fingerprint space consistent)
    if df_test is not None:
        df_all = pd.concat([df_train_used, df_test], ignore_index=True).reset_index(drop=True)
    else:
        df_all = df_train_used.copy().reset_index(drop=True)

    print(f"Featurizing {len(df_all)} reactions into {fp_nbits}-bit fingerprints (this may take a while)...")
    X_fp_all, fe_art = featurize_df(df_all, radius=2, nBits=fp_nbits, show_progress=True)
    print("Featurization done.")

    # split back, train used rows first
    n_train_used = len(df_train_used)
    X_train = X_fp_all[:n_train_used]
    X_test = X_fp_all[n_train_used:] if df_test is not None else None

    # Prepare labels for agent filtered
    agent_le = LabelEncoder()
    y_train_agent = agent_le.fit_transform(df_train_used['agent'].fillna("OTHER").astype(str).values)

    # train linear SGD multiclass logistic for agent
    print("Training memory-light SGDClassifier (log_loss) for agent prediction ...")
    clf_agent = SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3, random_state=42)
    clf_agent.fit(X_train, y_train_agent)
    print("Agent model trained.")

    # solvent encoder & classifier
    solvent_le = LabelEncoder()
    y_train_solvent = solvent_le.fit_transform(df_train_used['solvent'].fillna("OTHER").astype(str).values)
    print("Training memory-light SGDClassifier (log_loss) for solvent prediction ...")
    clf_solvent = SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3, random_state=42)
    clf_solvent.fit(X_train, y_train_solvent)
    print("Solvent model trained.")

    # optional evaluation on test split (if test has agent labels)
    if df_test is not None and X_test is not None:
        try:
            df_test_mapped, y_true = map_test_agents_to_top(df_test, agent_le)
            if df_test_mapped is not None and y_true is not None and len(y_true) == len(X_test):
                try:
                    y_pred = clf_agent.predict(X_test)
                    print("Agent classification report on test split:")
                    print(classification_report(y_true, y_pred, zero_division=0))
                    print("Agent test accuracy:", accuracy_score(y_true, y_pred))
                except Exception as e:
                    print("Agent evaluation failed (non-fatal):", e)
            else:
                print("Skipping agent evaluation: test mapping or X_test shape mismatch.")
        except Exception as e:
            print("Test evaluation mapping failed (non-fatal):", e)

    # build nearest-neighbor index on a sampled subset to save memory
    nn_idx = None
    X_fp_for_nn = None
    df_for_nn = None
    if nn_sample is not None and nn_sample > 0:
        try:
            print(f"Sampling up to {nn_sample} rows for nearest-neighbor retrieval (saves memory).")
            total = len(df_all)
            nn_sample_actual = min(int(nn_sample), total)
            rng = np.random.RandomState(42)
            sample_idx = rng.choice(total, size=nn_sample_actual, replace=False)
            X_fp_for_nn = X_fp_all[sample_idx]
            df_for_nn = df_all.iloc[sample_idx].reset_index(drop=True)
            print("Building NearestNeighbors (cosine, brute) on sampled fingerprints...")
            nn_idx = NearestNeighbors(n_neighbors=6, metric='cosine', algorithm='brute')
            nn_idx.fit(X_fp_for_nn)
            print("NearestNeighbors index built.")
        except Exception as e:
            print("NearestNeighbors build failed (non-fatal):", e)
            nn_idx = None
    else:
        print("Skipping NN index build (nn_sample <= 0).")

    # prepare artifacts (convert top_agents to list for stable serialization)
    artifacts = {
        "clf_agent": clf_agent,
        "agent_le": agent_le,
        "clf_solvent": clf_solvent,
        "solvent_le": solvent_le,
        "nn_index": nn_idx,
        "X_fp_all": X_fp_all,
        "X_fp_for_nn": X_fp_for_nn,
        "df": df_all,
        "df_for_nn": df_for_nn,
        "top_agents": list(top_agents)
    }

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    print(f"Saving artifacts to {out_path} (compress=3).")
    joblib.dump(artifacts, out_path, compress=3)
    print("Saved artifact. Done.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train-csv", required=True)
    p.add_argument("--test-csv", default=None)
    p.add_argument("--out", default="models/artifacts_light.joblib")
    p.add_argument("--nbits", type=int, default=256)
    p.add_argument("--top-k", type=int, default=200)
    p.add_argument("--max-train-sample", type=int, default=300000, help="Max rows to use for training (None = all)")
    p.add_argument("--nn-sample", type=int, default=50000, help="Rows to sample for NN index (0 to disable)")
    args = p.parse_args()

    mt = args.max_train_sample if args.max_train_sample is not None else None
    train_and_save(args.train_csv, args.test_csv, args.out, fp_nbits=args.nbits, top_k_agents=args.top_k, max_train_sample=mt, nn_sample=args.nn_sample)
