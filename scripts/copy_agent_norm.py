import pandas as pd
from pathlib import Path

files = [
    "data/orderly_condition_with_rare_train_normalized.csv",
    "data/orderly_condition_with_rare_test_normalized.csv"
]

for path in files:
    p = Path(path)
    if not p.exists():
        print("Missing:", path)
        continue
    df = pd.read_csv(p, dtype=str, low_memory=False)
    if 'agent_norm' in df.columns:
        df['agent_000'] = df['agent_norm']
        df.to_csv(p, index=False)
        print("Updated agent_000 from agent_norm in", path)
    else:
        print("No agent_norm column in", path)
