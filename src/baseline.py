from sklearn.dummy import DummyClassifier, DummyRegressor
from src.data import load_dataset, featurize

def train_baseline(csv_path):

    df = load_dataset(csv_path)
    X, y_cat, X_fp, catalyst_le, cat_to_load, sol_le, base_le = featurize(df)
    clf = DummyClassifier(strategy="most_frequent")
    reg = DummyRegressor(strategy="mean")
    clf.fit(X, y_cat)
    reg.fit(X, df['loading'].values.astype(float))
    return clf, reg
