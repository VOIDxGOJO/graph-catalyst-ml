import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, top_k_accuracy_score, confusion_matrix

def load_artifacts(path):
    return joblib.load(path)

def write_text(path: Path, txt: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(txt, encoding='utf-8')
    print(f"Wrote: {path}")

def evaluate(artifacts_path: str, out_dir: str, topk: int = 5):
    AR = load_artifacts(artifacts_path)
    df_all = AR['df']  # combined train+test as produced by src.train
    X_fp_all = AR['X_fp_all']
    clf_agent = AR.get('clf_agent')
    agent_le = AR.get('agent_le')

    if clf_agent is None or agent_le is None:
        raise RuntimeError("Artifacts missing classifier or label encoder for agent.")

    test_mask = None
    if 'is_test' in df_all.columns:
        test_mask = df_all['is_test'].astype(bool).values
    elif 'split' in df_all.columns:
        test_mask = (df_all['split'].astype(str).values == 'test')
    else:
        # fallback- evaluate on rows that have agent label (conservative across whole set)
        print("Warning: no explicit test split column found in artifacts['df']. Evaluating on all labeled rows (may include train).")
        test_mask = df_all['agent'].notna() & (df_all['agent'].astype(str).str.strip() != '')

    idxs = np.where(test_mask)[0].tolist()
    if len(idxs) == 0:
        raise RuntimeError("No test rows found (after mask). Please provide a test split or ensure artifacts['df'] has test rows.")

    X_test = X_fp_all[idxs]
    y_true_labels = df_all.iloc[idxs]['agent'].fillna('<<NULL>>').astype(str).values
    y_true = agent_le.transform(y_true_labels)

    print(f"Evaluating on {len(idxs)} labeled rows.")

    y_pred = clf_agent.predict(X_test)
    # if predict_proba exists, compute top-k accuracy
    has_proba = hasattr(clf_agent, "predict_proba")

    # classification report and accuracy
    report = classification_report(y_true, y_pred, zero_division=0, target_names=agent_le.classes_)
    acc = accuracy_score(y_true, y_pred)

    outdir = Path(out_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    write_text(outdir / "classification_report.txt", report)
    write_text(outdir / "summary.txt", f"n_test={len(idxs)}\naccuracy={acc:.6f}\nhas_proba={has_proba}\n")

    if has_proba:
        proba = clf_agent.predict_proba(X_test)
        try:
            topk_acc = top_k_accuracy_score(y_true, proba, k=topk)
            write_text(outdir / "topk_accuracy.txt", f"top-{topk} accuracy: {topk_acc:.6f}\n")
        except Exception as e:
            write_text(outdir / "topk_accuracy.txt", f"top-k accuracy computation failed: {e}\n")
            topk_acc = None
    else:
        write_text(outdir / "topk_accuracy.txt", "Classifier does not support predict_proba; top-k accuracy not available.\n")
        topk_acc = None

    # per-class top-k: compute for each class label, topk accuracy restricted to that class if proba available
    if has_proba and topk_acc is not None:
        classes = agent_le.classes_
        per_class = []
        proba_all = clf_agent.predict_proba(X_test)
        for ci, cls in enumerate(classes):
            mask = (y_true == ci)
            if mask.sum() == 0:
                continue
            true_idx = np.where(mask)[0]
            try:
                # topk accuracy for subset
                sub_topk = top_k_accuracy_score(y_true[mask], proba_all[mask], k=topk)
            except Exception:
                sub_topk = None
            per_class.append({"agent": cls, "support": int(mask.sum()), "topk": sub_topk})
        df_pc = pd.DataFrame(per_class)
        df_pc.to_csv(outdir / "per_class_topk.csv", index=False)
        print(f"Wrote per-class top-{topk} to {outdir/'per_class_topk.csv'}")

    # confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # convert to df with label names
    labels = agent_le.inverse_transform(range(len(agent_le.classes_)))
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    cm_df.to_csv(outdir / "confusion_matrix.csv")
    print(f"Wrote confusion matrix to {outdir/'confusion_matrix.csv'}")

    print("Evaluation complete. See folder:", outdir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", required=True, help="models/artifacts.joblib path")
    parser.add_argument("--out-dir", default="models/eval", help="output directory")
    parser.add_argument("--topk", type=int, default=5, help="k for top-k accuracy")
    args = parser.parse_args()
    evaluate(args.artifacts, args.out_dir, topk=args.topk)
