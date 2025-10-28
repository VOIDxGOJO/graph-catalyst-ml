
import joblib, json, numpy as np
p = "./models/artifacts_balanced_final.joblib"
print("Loading artifact:", p)
art = joblib.load(p)
print("TYPE:", type(art))
print("Keys:", list(art.keys()))

def show(k):
    try:
        v = art[k]
        from collections import Counter
        if hasattr(v, 'classes_'):
            print(f"{k}: LabelEncoder with {len(v.classes_)} classes. Sample:", v.classes_[:20].tolist())
        elif isinstance(v, dict):
            print(f"{k}: dict keys ->", list(v.keys())[:20])
        elif isinstance(v, (list, tuple)) and len(v)>0:
            print(f"{k}: list len {len(v)}; sample type ->", type(v[0]))
        else:
            print(f"{k}: type {type(v)}")
    except Exception as e:
        print(f"{k}: EXCEPTION ->", e)

for k in art:
    show(k)

# check if nbits or train_metadata present
print("nbits in artifact:", art.get("nbits") or art.get("train_metadata",{}).get("nbits"))
print("nn_index present?:", ("nn_index" in art) and (art.get("nn_index") is not None))
# check sample fingerprint shape if exists
if art.get("X_fp_for_nn") is not None:
    import numpy as np
    X = art["X_fp_for_nn"]
    print("X_fp_for_nn shape:", getattr(X, "shape", None), "dtype:", getattr(X, "dtype", None))
print("Done")
