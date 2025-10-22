import argparse
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import classification_report, accuracy_score, top_k_accuracy_score

from src.data import load_orderly_csv, featurize_df

def train(train_csv: str, test_csv: str, out_path: str, n_estimators: int = 200, fp_nbits: int = 1024):
    print("Loading train CSV...")
    df_train = load_orderly_csv(train_csv)
    print(f" Train rows: {len(df_train)}")
    print("Loading test CSV...")
    df_test = load_orderly_csv(test_csv)
    print(f" Test rows: {len(df_test)}")

    # combine for featurization amd retrieval index (train first, test second)
    df_all = pd.concat([df_train, df_test], ignore_index=True).reset_index(drop=True)
    print(f" Total rows (train+test): {len(df_all)}")

    # featurize on the combined df (encoders will see all labels)
    print("Featurizing fingerprints for entire dataset...")
    X_fp_all, artifacts = featurize_df(df_all, radius=2, nBits=fp_nbits)
    agent_le = artifacts['agent_le']
    solvent_le = artifacts['solvent_le']

    # determine indices
    n_train = len(df_train)
    n_test = len(df_test)
    idx_train = list(range(0, n_train))
    idx_test = list(range(n_train, n_train + n_test))

    # filter train rows that actually have an agent label (non-null)
    train_has_agent_mask = [(df_all.loc[i, 'agent'] is not None and str(df_all.loc[i, 'agent']).strip() != '') for i in idx_train]
    train_idx_agent = [idx_train[i] for i, ok in enumerate(train_has_agent_mask) if ok]
    if len(train_idx_agent) == 0:
        raise RuntimeError("No agent labels found in train split. Cannot train agent classifier.")

    # prepare X_train, y_train (use agent_le to transform)
    X_train_agent = X_fp_all[train_idx_agent]
    y_train_agent = agent_le.transform(df_all.loc[train_idx_agent, 'agent'].fillna('<<NULL>>').astype(str).values)

    print(f" Training agent classifier on {len(train_idx_agent)} rows with {len(agent_le.classes_)} agent classes (encoder sees train+test labels).")
    clf_agent = RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1, random_state=42)
    clf_agent.fit(X_train_agent, y_train_agent)
    print(" Agent classifier trained.")

    # solvent classifier (train only on train rows that have solvent)
    train_has_solvent_mask = [(df_all.loc[i, 'solvent'] is not None and str(df_all.loc[i, 'solvent']).strip() != '') for i in idx_train]
    train_idx_solvent = [idx_train[i] for i, ok in enumerate(train_has_solvent_mask) if ok]
    clf_solvent = None
    if len(train_idx_solvent) > 0:
        X_train_sol = X_fp_all[train_idx_solvent]
        y_train_sol = solvent_le.transform(df_all.loc[train_idx_solvent, 'solvent'].fillna('<<NULL>>').astype(str).values)
        print(f" Training solvent classifier on {len(train_idx_solvent)} rows with {len(solvent_le.classes_)} solvent classes.")
        clf_solvent = RandomForestClassifier(n_estimators=max(50, n_estimators // 4), n_jobs=-1, random_state=42)
        clf_solvent.fit(X_train_sol, y_train_sol)
        print(" Solvent classifier trained.")
    else:
        print(" No solvent labels in train; skipping solvent classifier.")

    # build NearestNeighbors index for retrieval (use cosine)
    print("Building NearestNeighbors index for retrieval...")
    nn = NearestNeighbors(n_neighbors=6, metric='cosine', algorithm='brute')
    nn.fit(X_fp_all)

    # eval on test split (only rows that have agent)
    test_has_agent_mask = [(df_all.loc[i, 'agent'] is not None and str(df_all.loc[i, 'agent']).strip() != '') for i in idx_test]
    test_idx_agent = [idx_test[i] for i, ok in enumerate(test_has_agent_mask) if ok]
    if len(test_idx_agent) > 0:
        X_test = X_fp_all[test_idx_agent]
        y_test_true = agent_le.transform(df_all.loc[test_idx_agent, 'agent'].fillna('<<NULL>>').astype(str).values)
        y_test_pred = clf_agent.predict(X_test)
        print("Classification report (agent) on test split:")
        print(classification_report(y_test_true, y_test_pred, zero_division=0))
        acc = accuracy_score(y_test_true, y_test_pred)
        print(f" Test accuracy (agent): {acc:.4f}")
        # top5 accuracy if predict_proba exists
        try:
            proba = clf_agent.predict_proba(X_test)
            top5 = top_k_accuracy_score(y_test_true, proba, k=5)
            print(f" Test top-5 accuracy (agent): {top5:.4f}")
        except Exception:
            print(" Top-5 accuracy not available for this classifier.")
    else:
        print(" No labeled agent rows in test split to evaluate.")

    # save artifacts (include classifier, encoders, nn, fingerprints, df_all)
    packed = {
        'clf_agent': clf_agent,
        'clf_solvent': clf_solvent,
        'agent_le': agent_le,
        'solvent_le': solvent_le,
        'nn_index': nn,
        'X_fp_all': X_fp_all,
        'df': df_all
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    joblib.dump(packed, out_path)
    print(f"Saved artifacts to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", required=True, help="Path to train CSV")
    parser.add_argument("--test-csv", required=True, help="Path to test CSV")
    parser.add_argument("--out", default="models/artifacts.joblib", help="Output path for artifacts")
    parser.add_argument("--n-est", type=int, default=200, help="n_estimators for random forest")
    parser.add_argument("--nbits", type=int, default=1024, help="Fingerprint nBits")
    args = parser.parse_args()
    train(args.train_csv, args.test_csv, args.out, n_estimators=args.n_est, fp_nbits=args.nbits)
