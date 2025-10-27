
import pandas as pd
df = pd.read_csv("data/orderly_condition_with_rare_train_norm_whitelist_skipbad.csv", dtype=str)
print("Total rows:", len(df))
if 'agent_norm' in df.columns:
    vc = df['agent_norm'].fillna('<<EMPTY>>').value_counts()
    print("Top 30 agent_norm counts:")
    print(vc.head(30).to_string())
else:
    print("No column 'agent_norm' found. Columns:", df.columns.tolist()[:40])