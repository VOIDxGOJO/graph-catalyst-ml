import pandas as pd
from pathlib import Path
import numpy as np

IN = Path("data/orderly_condition_with_rare_train_norm_whitelist_skipbad.csv")
OUT = Path("data/orderly_condition_with_rare_train_balanced.csv")
CAP = 2000 
RANDOM_STATE = 42

print("Reading:", IN)
df = pd.read_csv(IN, dtype=str)
print("Rows total:", len(df))
if 'agent_norm' not in df.columns:
    raise SystemExit("agent_norm column not found in input CSV")

# filter out empty/OTHER depending on column convention
mask_valid = df['agent_norm'].fillna('').astype(str).str.strip() != ''
df = df[mask_valid].copy()
print("After dropping empty agent_norm:", len(df))

# group and sample
groups = df.groupby('agent_norm', sort=False)
sampled_parts = []
rng = np.random.RandomState(RANDOM_STATE)
for name, g in groups:
    n = len(g)
    take = min(n, CAP)
    if n <= CAP:
        sampled_parts.append(g)
    else:
        sampled_parts.append(g.sample(n=take, random_state=RANDOM_STATE))
    print(f"Label={name} count={n} kept_for_balanced={min(n,CAP)}")
df_bal = pd.concat(sampled_parts, ignore_index=True).sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)
print("Balanced dataset rows:", len(df_bal))
OUT.parent.mkdir(parents=True, exist_ok=True)
df_bal.to_csv(OUT, index=False)
print("Wrote:", OUT)
