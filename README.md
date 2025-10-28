# Reaction Catalyst Prediction

This project predicts the **catalyst** required for a given chemical reaction.
It uses **graph-based ML** (Morgan fingerprints via RDKit) trained on the **ORDerly dataset** and exposes a **FastAPI + Frontend** for easy prediction.

---

## Quick Start

### Clone the repo
```cmd
git clone https://github.com/dshryn/graph-catalyst-ml.git
cd graph-catalyst-ml
```

### Create environment
```cmd
conda create -n catalyst-ml python=3.10 -y
conda activate catalyst-ml
conda install -c conda-forge rdkit tqdm scikit-learn fastapi uvicorn pandas numpy joblib -y
```


## Dataset

Download the ORDerly dataset from Figshare:
https://figshare.com/articles/dataset/ORDerly_chemical_reactions_condition_benchmarks/23298467?file=44413037

Place it like:
```
graph-catalyst-ml/data/
 ├─ orderly_condition_with_rare_train.csv
 └─ orderly_condition_with_rare_test.csv
```

## Preprocessing Pipeline

Clean RXN SMILES
```cmd
python scripts/clean_rxn_str.py --input ./data/orderly_condition_with_rare_train.csv --output ./data/tmp_train_cleaned.csv --chunksize 2000 --max-rows 5000 --force-overwrite
```

Normalize + Explode Agents
```cmd
python scripts/normalize_and_explode_agents.py --input ./data/tmp_train_cleaned.csv --output ./data/orderly_condition_with_rare_train_normalized.csv
````

Skip malformed lines
```cmd
python scripts/skip_bad_lines.py
```

This creates:
```cmd
data/orderly_condition_with_rare_train_norm_whitelist_skipbad.csv
```

Balance dataset
```cmd
python scripts/sample_balance.py --input ./data/orderly_condition_with_rare_train_norm_whitelist_skipbad.csv --output ./data/orderly_condition_with_rare_train_balanced.csv --max-samples 2000
```

## Training

Run:
```cmd
python -m src.train --train-csv "./data/orderly_condition_with_rare_train_balanced.csv" --test-csv "./data/orderly_condition_with_rare_test_norm_whitelist.csv" --out "./models/artifacts_balanced.joblib" --nbits 128 --chunk-size 500 --max-rows 30000 --nn-sample 1000 --exclude-other --class-weight balanced --random-state 42
```

Model artifact:
```cmd
models/artifacts_balanced.joblib
```


## Debug / Local Inference
```cmd
python scripts/debug_inference.py --artifact ./models/artifacts_balanced.joblib --smiles "CC(=O)c1cc(C)cc([N+](=O)[O-])c1O>>CC(=O)c1cc(C)cc(N)c1O"
```
Expected
Fe / Cu / Pd (for nitro reduction reactions)


## Run API + Frontend

Start API
```cmd
uvicorn src.api_server:app --reload --port 8000
```

### Open Frontend

Open web/index.html in a browser and enter required fields to predict the catalyst.