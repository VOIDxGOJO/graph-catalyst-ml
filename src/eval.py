from sklearn.metrics import classification_report, mean_absolute_error, r2_score
from src.data import load_dataset, featurize

def evaluate_models(csv_path, clf, reg):

    df = load_dataset(csv_path)
    X, y_cat, X_fp, catalyst_le, cat_to_load, sol_le, base_le = featurize(df)
    y_cat_pred = clf.predict(X)
    y_load_pred = reg.predict(X)

    print("Classification Report:")
    print(classification_report(y_cat, y_cat_pred, zero_division=0))
    
    mae = mean_absolute_error(df['loading'], y_load_pred)
    r2 = r2_score(df['loading'], y_load_pred)
    print(f"Regression (loading) MAE: {mae:.3f}, R2: {r2:.3f}")
