
import sys, joblib, numpy as np, json
from pathlib import Path

def main(p):
    p = Path(p)
    if not p.exists():
        print("File not found:", p)
        return 2
    print("Loading:", p)
    art = joblib.load(str(p))
    print("Loaded artifact type:", type(art))
    keys = list(art.keys()) if isinstance(art, dict) else None
    print("Top-level keys:", keys)
    # common expected keys
    expected = ["clf_agent", "agent_le", "clf_solvent", "solvent_le", "nn_index", "X_fp_for_nn", "df_for_nn", "train_metadata"]
    for k in expected:
        print(f" - {k}: {'PRESENT' if (isinstance(art, dict) and k in art) else 'MISSING'}")
    # agent_le details
    if isinstance(art, dict) and "agent_le" in art:
        le = art["agent_le"]
        try:
            cls = getattr(le, "classes_", None)
            print("agent_le.classes_ count:", len(cls) if cls is not None else "None")
            if cls is not None:
                print("Sample classes (first 50):", list(cls)[:50])
        except Exception as e:
            print("agent_le read error:", e)
    # check nbits
    nbits = art.get("nbits", None) if isinstance(art, dict) else None
    if not nbits:
        nbits = art.get("train_metadata", {}).get("nbits") if isinstance(art, dict) else None
    print("nbits:", nbits)
    # X_fp_for_nn
    if isinstance(art, dict) and art.get("X_fp_for_nn") is not None:
        X = art.get("X_fp_for_nn")
        try:
            print("X_fp_for_nn shape:", getattr(X, "shape", None), "dtype:", getattr(X, "dtype", None))
            print("bit-count stats (first 10 rows):", [int(x.sum()) for x in X[:10]])
        except Exception as e:
            print("X_fp_for_nn inspect error:", e)
    # nn_index
    if isinstance(art, dict) and art.get("nn_index") is not None:
        print("nn_index type:", type(art.get("nn_index")))
    print("Artifact inspection finished.")
    return 0

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/check_artifact.py <path-to-artifact-joblib>")
        sys.exit(1)
    sys.exit(main(sys.argv[1]))
