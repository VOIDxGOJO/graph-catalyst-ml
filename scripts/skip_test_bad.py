import pandas as pd

inp = "data/orderly_condition_with_rare_test_norm_whitelist.csv"
out = "data/orderly_condition_with_rare_test_norm_whitelist_skipbad.csv"

print("Reading test CSV and skipping malformed lines (engine=python)...")

df = pd.read_csv(inp, dtype=str, on_bad_lines='skip', engine='python')
print("Rows read:", len(df))

df.to_csv(out, index=False)
print("Written:", out)
