# NFL Career Longevity Predictor

Final project for DATA 6545 (Spring 2026). Predicts NFL career longevity (Short, Medium, Long, Elite) using draft information and rookie-year performance.

## Live Demo
(https://nfl-career-longevity-gjvg3dtdj49z3bvkzxjdnn.streamlit.app/)

## Project Overview
- **Target:** Career longevity classified into 4 buckets
- **Data:** nflverse draft picks 2005-2018, weekly stats, snap counts
- **Best Model:** Combined features + Random Forest, Macro F1 = 0.359
- **Skill-position subgroup:** Combined features + Logistic Regression, Macro F1 = 0.405

## Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Files
- `app.py` — Streamlit dashboard
- `best_model_full.pkl` — Trained model for all positions
- `best_model_skill.pkl` — Trained model for skill positions only
- `model_metadata.json` — Feature lists and metadata

## Author
Valerie Bien-Aime
