# scripts/inspect_artifact.py
# Self-contained inspector that ensures your repo root is on sys.path so `import src` works.
import sys, os
from pathlib import Path
here = Path(__file__).resolve().parent
repo_root = here.parent  # project root
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import joblib, numpy as np
# now we can import project modules
try:
    from src.data import reaction_to_fp
except Exception as e:
    print("Failed to import src.data:", e)
    raise

ART_PATH = "models/artifacts_quick.joblib"  # adjust if you used a different artifact
if not Path(ART_PATH).exists():
    # try other likely names
    alt = "models/artifacts_light.joblib"
    if Path(alt).exists():
        ART_PATH = alt
    else:
        print("Artifact not found at models/artifacts_quick.joblib or models/artifacts_light.joblib")
        raise SystemExit(1)

print("Loading artifact:", ART_PATH)
art = joblib.load(ART_PATH)
print("Artifact keys:", list(art.keys()))

agent_le = art.get("agent_le", None)
train_agent = art.get("train_agent", None) or art.get("train_agent_names", None) or art.get("train_agent_list", None)
clf = art.get("clf_agent", None)

print("\nClassifier object:", type(clf))
print("Has predict_proba?:", hasattr(clf, "predict_proba"))

if agent_le is not None:
    try:
        classes = agent_le.classes_
        print("Number of encoded agent classes in LabelEncoder:", len(classes))
        print("First 40 classes sample:", classes[:40])
    except Exception as e:
        print("Couldn't show agent_le.classes_:", e)

if train_agent is not None:
    try:
        import pandas as pd
        s = pd.Series(train_agent).value_counts()
        print("\nTop 20 agents in training set with counts:")
        print(s.head(20).to_string())
        print("\nCount for 'OTHER' (if present):", int(s.get("OTHER", 0)))
    except Exception as e:
        print("Error summarizing train_agent:", e)

# Test a few SMILES to see classifier probabilities and fingerprint health
SAMPLES = [
    "Nc1ccc(O)cc1F.CC(C)([O-])[K+].Cl>>COc1cc(Cl)ccn1",
    "CC(=O)c1cc(C)cc([N+](=O)[O-])c1O>>CC(=O)c1cc(C)cc(N)c1O",
    "CC(=O)OCC>>CC(=O)OCC"  # trivial
]

nbits = None
if art.get("X_fp_all") is not None:
    try:
        nbits = int(np.asarray(art.get("X_fp_all")).shape[1])
    except Exception:
        pass
if nbits is None and art.get("X_fp_for_nn") is not None:
    try:
        nbits = int(np.asarray(art.get("X_fp_for_nn")).shape[1])
    except Exception:
        pass
if nbits is None:
    nbits = 128

print("\nUsing fingerprint width nbits =", nbits)

for s in SAMPLES:
    print("\n=== SAMPLE ===")
    print("SMILES:", s)
    fp = reaction_to_fp(s, radius=2, nBits=nbits)
    fp = np.asarray(fp, dtype=float)
    print("fp shape:", fp.shape, "nonzero bits:", int((fp != 0).sum()), "L2-norm:", float(np.linalg.norm(fp)))
    if clf is not None:
        try:
            X = fp.reshape(1, -1)
            if hasattr(clf, "predict_proba"):
                probs = clf.predict_proba(X)[0]
                order = np.argsort(probs)[::-1][:10]
                print("Top predicted labels (index -> prob -> label):")
                for i in order:
                    lab_enc = clf.classes_[int(i)]
                    try:
                        lab = agent_le.inverse_transform([int(lab_enc)])[0]
                    except Exception:
                        lab = str(lab_enc)
                    print(f"  {int(i)} -> {probs[i]:.4f} -> {lab}")
            else:
                p = clf.predict(X)[0]
                try:
                    lab = agent_le.inverse_transform([int(p)])[0]
                except Exception:
                    lab = str(p)
                print("predict ->", lab)
        except Exception as e:
            print("Prediction error for this sample:", e)
