import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors

# minimal fake dataset
df = pd.DataFrame({
    'id': [str(i) for i in range(10)],
    'smiles': [
        "CC(=O)c1cc(C)cc([N+](=O)[O-])c1O>>CC(=O)c1cc(C)cc(N)c1O",
        "Nc1ccc(O)cc1F.CC(C)([O-])C.[K+]>>COc1cc(Cl)ccn1",
        "[CH3:1]...>>[CH3:1]...",  # placeholders
        "Oc1ccc(F)cc1>>Oc1ccc(F)cc1",
        "Cc1ccc(...)>>Cc1ccc(...)",
        "CCOC(C)=O>>CCOC(C)=O",
        "C1COCCO1>>C1COCCO1",
        "CCCC>>CCCC",
        "CC(C)OC(=O)...>>CC(C)OC(=O)...",
        "C1CCOC1>>C1CCOC1"
    ],
    'agent': ["[Pt]", "[K+]", "[Pt]", "[H-]", "Cl", None, None, "CN(C)c1ccccn1", "[H-]", None],
    'solvent': ["CCO", "DMA", "CCO", "DMF", "CCOC(C)=O", "C1COCCO1", None, None, "CCO", None],
    'procedure_details': ["proc A","proc B","","","","","","","",""]
})

# dummy fingerprint matrix as random binary vectors
n, nbits = len(df), 1024
rng = np.random.RandomState(42)
X_fp = rng.randint(0, 2, size=(n, nbits)).astype(np.int8)

# dummy classifiers that predict the most frequent class
agent_le = LabelEncoder()
agent_vals = df['agent'].fillna('<<NULL>>').astype(str).values
agent_le.fit(agent_vals)
clf_agent = DummyClassifier(strategy="most_frequent").fit(X_fp, agent_le.transform(agent_vals))

solvent_le = LabelEncoder()
sol_vals = df['solvent'].fillna('<<NULL>>').astype(str).values
solvent_le.fit(sol_vals)
clf_solvent = DummyClassifier(strategy="most_frequent").fit(X_fp, solvent_le.transform(sol_vals))

# NearestNeighbors on the random fingerprint matrix for retrieval
nn = NearestNeighbors(n_neighbors=6, metric='cosine', algorithm='brute')
nn.fit(X_fp)

artifacts = {
    'clf_agent': clf_agent,
    'clf_solvent': clf_solvent,
    'agent_le': agent_le,
    'solvent_le': solvent_le,
    'nn_index': nn,
    'X_fp_all': X_fp,
    'df': df
}

joblib.dump(artifacts, "model/artifacts.joblib")
print("Dummy artifacts written to model/artifacts.joblib")
