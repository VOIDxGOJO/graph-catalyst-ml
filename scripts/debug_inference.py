#!/usr/bin/env python3
"""
run
python scripts/debug_inference.py --artifact ./models/artifacts_balanced2.joblib --smiles "YOUR_RXN>>PRODUCT"
"""

import argparse
import sys
from pathlib import Path

# ensure repo root is on sys.path so import src works even when running from scripts/
_this_file = Path(__file__).resolve()
REPO_ROOT = _this_file.parent.parent  # assumes scripts/ is at repo_root/scripts
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import joblib
import numpy as np
from src.data import reaction_to_fp

def stable_softmax(logits):
    z = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(z)
    probs = exp / np.sum(exp, axis=1, keepdims=True)
    return probs

def describe_artifact(artifact):
    keys = sorted(list(artifact.keys()))
    print("Artifact keys:", keys)
    if "agent_le" in artifact:
        le = artifact["agent_le"]
        try:
            classes = list(le.classes_)
        except Exception:
            classes = "<unreadable>"
        print("Number of classes in agent_le:", len(classes) if isinstance(classes, list) else classes)
        if isinstance(classes, list):
            print("Sample classes (first 100):", classes[:100])
    else:
        print("Warning: agent_le not found in artifact.")
    if "clf_agent" not in artifact:
        raise RuntimeError("Artifact missing 'clf_agent' key.")
    print("nbits in artifact:", artifact.get("nbits", "(not set)"))
    print()

def predict_single(artifact, smi, topk=10):
    clf = artifact["clf_agent"]
    le = artifact["agent_le"]
    nbits = int(artifact.get("nbits", 128))

    print("Input SMILES:", smi)
    fp = reaction_to_fp(smi, nBits=nbits)
    print("Fingerprint shape:", fp.shape, "bit-count (sum):", int(fp.sum()))
    X = fp.reshape(1, -1)

    # decision_function (if available)
    df = None
    try:
        df = clf.decision_function(X)
    except Exception as e:
        print("decision_function not available or failed:", e)

    probs = None
    if df is not None:
        if df.ndim == 1:
            df_shaped = np.vstack([-df, df]).T
        else:
            df_shaped = df
        probs = stable_softmax(df_shaped)
        top_idx = np.argsort(probs[0])[::-1][:topk]
        print("Top predictions (decision_function -> softmax):")
        for i in top_idx:
            lbl = le.classes_[int(i)] if i < len(le.classes_) else f"IDX_{i}"
            print(f"  - {lbl:30s}  prob={float(probs[0,i]):.4f}  logit={float(df_shaped[0,i]):.4f}")
        top_label = le.classes_[int(top_idx[0])]
        top_prob = float(probs[0,top_idx[0]])
    else:
        if hasattr(clf, "predict_proba"):
            probs = clf.predict_proba(X)
            top_idx = np.argsort(probs[0])[::-1][:topk]
            print("Top predictions (predict_proba):")
            for i in top_idx:
                lbl = le.classes_[int(i)] if i < len(le.classes_) else f"IDX_{i}"
                print(f"  - {lbl:30s}  prob={float(probs[0,i]):.4f}")
            top_label = le.classes_[int(top_idx[0])]
            top_prob = float(probs[0,top_idx[0]])
        else:
            pred = clf.predict(X)[0]
            try:
                pred_label = le.inverse_transform([int(pred)])[0]
            except Exception:
                pred_label = str(pred)
            print("Predict only, no probabilities available. Pred:", pred_label)
            top_label = pred_label
            top_prob = None

    # final predict
    try:
        pred_idx = int(clf.predict(X)[0])
        pred_label = le.classes_[pred_idx] if pred_idx < len(le.classes_) else str(pred_idx)
    except Exception:
        pred_label = "PRED_ERR"

    print("\nFinal predicted label (clf.predict):", pred_label)
    if top_prob is not None:
        print("Top softmax prob (proxy confidence):", f"{top_prob:.4f}")
    else:
        print("No probability available for this classifier.")

    # nearest neighbors (opt)
    if artifact.get("nn_index", None) is not None and artifact.get("X_fp_for_nn", None) is not None:
        try:
            nn = artifact["nn_index"]
            X_fp = artifact["X_fp_for_nn"]
            df_nn = artifact.get("df_for_nn", None)
            nret = min(5, len(X_fp))
            dists, idxs = nn.kneighbors(X.reshape(1, -1), n_neighbors=nret)
            print("\nNearest neighbors (sampled NN index):")
            for d, i in zip(dists[0], idxs[0]):
                if df_nn is not None:
                    r = df_nn.iloc[int(i)]
                    ident = r.get("index", r.get("original_index", str(i)))
                    smi_r = r.get("reaction_smiles", r.get("smiles", ""))
                    agent_r = r.get("agent_norm", r.get("agent", ""))
                    print(f" - idx={i} dist={float(d):.4f} id={ident} agent_norm={agent_r} smiles={smi_r}")
                else:
                    print(f" - idx={i} dist={float(d):.4f}")
        except Exception as e:
            print("NearestNeighbors query failed:", e)

    return

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--artifact", required=True, help="Path to joblib artifact")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--smiles", help="Single reaction SMILES (reactants>>product)")
    g.add_argument("--batch-file", help="Plain text file with one SMILES per line")
    p.add_argument("--topk", type=int, default=10)
    args = p.parse_args()

    art_path = Path(args.artifact)
    if not art_path.exists():
        print("Artifact not found:", art_path)
        sys.exit(2)

    artifact = joblib.load(str(art_path))
    describe_artifact(artifact)

    if args.smiles:
        predict_single(artifact, args.smiles, topk=args.topk)
    else:
        bf = Path(args.batch_file)
        if not bf.exists():
            print("Batch file not found:", bf)
            sys.exit(3)
        with open(bf, "r", encoding="utf-8") as fh:
            for line in fh:
                smi = line.strip()
                if not smi:
                    continue
                print("\n=== NEXT SMILES ===")
                predict_single(artifact, smi, topk=args.topk)

if __name__ == "__main__":
    main()
