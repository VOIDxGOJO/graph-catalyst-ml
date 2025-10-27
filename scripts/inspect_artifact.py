"""
inspect saved model artifact and print summary of label classes.
Run- python scripts/inspect_artifact.py {path_to_artifact}
"""

import sys
from pathlib import Path
import joblib

def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("./models/artifacts_balanced2.joblib")
    if not path.exists():
        print("Artifact not found at", path)
        sys.exit(2)
    a = joblib.load(str(path))
    print("Artifact keys:", list(a.keys()))
    le = a.get("agent_le")
    clf = a.get("clf_agent")
    if le is None:
        print("No 'agent_le' found in artifact.")
        sys.exit(0)
    classes = list(le.classes_)
    print("Total agent classes:", len(classes))
    print("\nFirst 80 classes (or fewer):")
    for i, c in enumerate(classes[:80]):
        print(f"{i:03d}: {c}")

    # heuristics to find smiles-like labels
    smile_like = [c for c in classes if ('[' in str(c) or (any(ch in str(c) for ch in 'CNOPS') and '(' in str(c)))]
    if smile_like:
        print("\nSample labels that look like SMILES/molecular strings (heuristic):")
        for s in smile_like[:40]:
            print(" -", s)
    else:
        print("\nNo obviously SMILES-like labels detected by heuristic.")
    print("\nDone.")

if __name__ == "__main__":
    main()
