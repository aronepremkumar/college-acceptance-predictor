# College Admission Predictor 

**Predict Stanford admission using Common Data Set (CDS) aggregates + synthetic applicant data.**  
Deployed as a **Dockerized FastAPI service** on **AWS Elastic Beanstalk** (EB).

---

## Problem Description
Stanford admits ~2,000 of 56,000 applicants (~3.6% rate). This binary classifier predicts admission (0=No, 1=Yes) using CDS-derived features: test scores, GPA, demographics, hooks (legacy/athlete).  

**Real-world use**: Admissions offices simulate "what-if" scenarios. Counselors advise students.

**Data Ethics**: Aggregates from public CDS; synthetic generation preserves privacy.

---

## Dataset Creation
- **Source**: Stanford 2023-2024 CDS + CDS template (your link). 
- **How Created** (run `create_dataset.py` once):
  1. Generate synthetic applicants calibrated to CDS stats.
- **File**: `data/stanford_admissions.csv` (committed; 50,000 rows × 16 cols).
- **Target**: `admit` (binary; ~3.6% positive rate).
- **Regenerate**: `python create_dataset.py`

---

## How to Run Locally

```bash
# 1. Clone & enter
git clone https://github.com/aronepremkumar/college-acceptance-predictor
cd college-acceptance-predictor

# 2. Create virtual env
python -m venv venv
source venv/bin/activate  # Linux/Mac
# .\venv\Scripts\activate  # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Recreate dataset
python create_dataset.py

# 5. Train model
python train.py

# 6. Run service
uvicorn predict:app --host 0.0.0.0 --port 8000

# 7. Test (or visit http://localhost:8000/docs)
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 18,
    "gender_m": 1,
    "in_state": 0,
    "race_asian": 1,
    "race_white": 0,
    "race_hispanic": 0,
    "race_black": 0,
    "race_other": 0,
    "sat_math": 780,
    "sat_ebrw": 760,
    "act_composite": 0,
    "gpa": 3.98,
    "extracurriculars_count": 5,
    "first_gen": 0,
    "legacy": 1,
    "athlete": 0
}'

# 8. Building and Running Docker Container
# Build Docker Image
docker build -t stanford-cds-api .

# Run Container 
docker run -p 8000:8000 stanford-cds-api