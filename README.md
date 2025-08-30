# CatalystPropNet

**Graph-Based Machine Learning for Catalyst Property Prediction and Design**

This project aims to predict catalyst properties and optimal catalyst usage for chemical reactions using machine learning models.  
It helps reduce the need for repetitive laboratory experiments by providing computational predictions.

## Features
- Data ingestion and preprocessing (structured catalyst datasets)
- ML modeling (baseline linear regression → scalable to advanced models)
- Web interface for reaction input & catalyst prediction
- REST API backend for model serving
- Modular directory structure with reproducibility in mind

## Tech Stack
- **Python** (pandas, scikit-learn, numpy, FastAPI)
- **FastAPI** backend for serving predictions
- **Next.js + Tailwind** for frontend UI
- **Docker + docker-compose** for environment consistency
- **Jupyter Notebooks** for exploratory data analysis

## Getting Started
```bash
# clone repo
git clone https://github.com/VOIDxGOJO/graph-catalyst-ml.git
cd graph-catalyst-ml

# create virtual environment
python -m venv venv
.\venv\Scripts\activate   # Windows
source venv/bin/activate  # Linux/Mac

# install dependencies
pip install -r requirements.txt

# run FastAPI backend
uvicorn api.main:app --reload
