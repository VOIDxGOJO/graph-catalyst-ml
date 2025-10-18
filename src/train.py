import pickle
from src.data import load_dataset, featurize
from src.model import build_models
from sklearn.metrics import accuracy_score, mean_squared_error

def train_and_save(csv_path, model_path):

    df = load_dataset(csv_path)
    X, y_cat, X_fp, catalyst_le, cat_to_load, sol_le, base_le = featurize(df)
    
    clf, reg = build_models()
    clf.fit(X, y_cat)
    reg.fit(X, df['loading'].values.astype(float))
    
    acc = accuracy_score(y_cat, clf.predict(X))
    mse = mean_squared_error(df['loading'], reg.predict(X))
    print(f"Training accuracy: {acc:.3f}, Loading MSE: {mse:.3f}")
    
    artifacts = {
        'clf': clf,
        'reg': reg,
        'catalyst_le': catalyst_le,
        'sol_le': sol_le,
        'base_le': base_le,
        'catalyst_to_loading': cat_to_load,

        'train_ids': df['id'].tolist(),
        'train_smiles': df['smiles'].tolist(),
        'train_catalyst': df['catalyst'].tolist(),
        'train_loading': df['loading'].tolist(),
        'train_fp': X_fp 
    }
    with open(model_path, 'wb') as f:
        pickle.dump(artifacts, f)

if __name__ == "__main__":
    # replace new path 
    train_and_save("data/SMILES_Big_Data_Set.csv", "model/artifacts.pkl")
