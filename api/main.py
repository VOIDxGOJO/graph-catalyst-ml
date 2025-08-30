from fastapi import FastAPI

app = FastAPI(title="CatalystPropNet API", version="0.1")

@app.get("/")
def read_root():
    return {"message": "Welcome to CatalystPropNet API 🚀"}

@app.get("/predict")
def predict_placeholder(reaction: str = "H2 + O2 -> H2O"):
    # placeholder
    return {
        "reaction": reaction,
        "predicted_catalyst": "Pt (Platinum)",
        "amount_required": "2.5 g"
    }
