# Catalyst Predictor

**Machine Learning model for Catalyst Prediction**

This project aims to predict catalyst and optimal catalyst usage for chemical reactions using machine learning models.  
It helps reduce the need for repetitive laboratory experiments by providing computational predictions.

## Features
- Data ingestion and preprocessing (structured catalyst dataset)
- Web interface for catalyst prediction
- Flask backend for model serving

## Tech Stack
- **Python** (pandas, scikit-learn, numpy, FastAPI)
- **FastAPI** backend for serving predictions

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
