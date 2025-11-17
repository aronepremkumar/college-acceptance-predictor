# predict.py
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import uvicorn

app = FastAPI(title="Stanford Admission Predictor")

# Load model
with open('model_xgboost.bin', 'rb') as f:
    dv, model = pickle.load(f)

class Applicant(BaseModel):
    age: int
    gender_m: int
    in_state: int
    race_asian: int
    race_white: int
    race_hispanic: int
    race_black: int
    race_other: int
    sat_math: int
    sat_ebrw: int
    act_composite: int = 0
    gpa: float
    extracurriculars_count: int
    first_gen: int
    legacy: int
    athlete: int

@app.get("/")
def root():
    return {"message": "College Acceptance Predictor"}

@app.post("/predict")
def predict(applicant: Applicant):
    data = applicant.dict()
    X = dv.transform([data])
    prob = model.predict_proba(X)[0, 1]
    return {"admission_probability": round(float(prob), 4)}

@app.get("/health")
def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))   # EB injects PORT
    uvicorn.run(app, host="0.0.0.0", port=port)