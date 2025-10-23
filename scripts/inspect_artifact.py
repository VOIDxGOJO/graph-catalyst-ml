import sys, os
from pathlib import Path
here = Path(__file__).resolve().parent
repo_root = here.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import joblib, numpy as np
from src.data import reaction_to_fp

ART_PATH = "models/artifacts_sgd.joblib"
if not Path(ART_PATH).exists():
    alt = "models/artifacts_forest.joblib"
    if Path(alt).exists():
        ART_PATH = alt
    else:
        print("Artifact not found; adjust path at top of this script.")
        raise SystemExit(1)

print("Loading artifact:", ART_PATH)
art = joblib.load(ART_PATH)
print("Artifact keys:", list(art.keys()))
clf = art.get("clf_agent")
agent_le = art.get("agent_le")

print("Classifier type:", type(clf))
print("Has predict_proba:", hasattr(clf, "predict_proba"))
if agent_le is not None:
    try:
        print("Agent classes:", len(agent_le.classes_))
    except Exception as e:
        print("Could not read agent_le:", e)

SAMPLES = [
    "Nc1ccc(O)cc1F.CC(C)([O-])[K+].Cl>>COc1cc(Cl)ccn1",
    "CC(=O)c1cc(C)cc([N+](=O)[O-])c1O>>CC(=O)c1cc(C)cc(N)c1O"
]
nbits = art.get("meta", {}).get("fp_params", {}).get("nbits", 128)
for s in SAMPLES:
    print("\nSMILES:", s)
    fp = reaction_to_fp(s, radius=2, nBits=nbits)
    print("nonzero bits:", int((np.asarray(fp) != 0).sum()))
    if clf is not None:
        X = np.asarray(fp, dtype=float).reshape(1,-1)
        try:
            if hasattr(clf, "predict_proba"):
                probs = clf.predict_proba(X)[0]
                order = np.argsort(probs)[::-1][:5]
                for i in order:
                    enc = clf.classes_[int(i)]
                    try:
                        lab = agent_le.inverse_transform([int(enc)])[0]
                    except Exception:
                        lab = str(enc)
                    print(f" {lab} -> {probs[i]:.3f}")
            else:
                p = clf.predict(X)[0]
                print("pred:", p)
        except Exception as e:
            print("Prediction error:", e)
