from src.data import load_orderly_csv, featurize_df

df = load_orderly_csv('data/orderly_condition_with_rare_train_normalized.csv')
print('Rows in sample df:', len(df))

X, art = featurize_df(df.head(200), radius=2, nBits=512, show_progress=True)
print('X shape:', X.shape)
