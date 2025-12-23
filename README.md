# Student Behavioral Drift Detection

Predict student performance, identify behavioral patterns, and visualize drift across academic periods. Live demo: https://student-behavioral-drift-detection.onrender.com/

## Overview
- **Goals:**  
  - Classify pass/fail risk  
  - Predict final grade (G3)  
  - Cluster students by behavior  
  - Detect performance drift (G1 → G2)
- **Frontend:** Bootstrap UI with interactive sliders and charts.
- **Backend:** Flask + scikit-learn models loaded from `models/`.

## Key Features
- Pass/Fail risk (Random Forest classifier)
- Final grade prediction (Linear Regression)
- Behavioral clustering (K-Means)
- Drift analysis (grade deltas and cluster drift)
- Input validation and real-time drift preview

## Models
- **Scaler:** StandardScaler (`models/scaler.pkl`)
- **Classifier:** Random Forest (`models/classifier.pkl`)
- **Regressor:** Linear Regression (`models/regressor.pkl`)
- **Clustering:** K-Means (`models/kmeans.pkl`)

## Project Structure
```
app.py                  # Flask app + inference pipeline
models/                 # Saved pkl models (scaler, classifier, regressor, kmeans)
template/               # Flask templates (index.html, result.html)
static/                 # CSS and assets
requirements.txt        # Python deps
Procfile                # Gunicorn entrypoint for Render
```

## Local Setup
```bash
python -m venv .venv && source .venv/bin/activate  # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
flask run  # or: gunicorn app:app
# open http://127.0.0.1:5000
```

## Deployment (Render)
1. Ensure `models/` and all `.pkl` files are in the repo.  
2. Push to GitHub/GitLab.  
3. Create a Render Web Service:  
   - Build: `pip install -r requirements.txt`  
   - Start: `gunicorn app:app`  
   - Python version: 3.10+  
4. Deploy. Example live URL: https://student-behavioral-drift-detection.onrender.com/

## How Prediction Works
1. Collect inputs: study time, absences, failures, G1, G2.  
2. Engineer features: `grade_change_1_2`, `grade_change_2_3` (0 in app), cluster placeholders.  
3. Scale with StandardScaler.  
4. Predict:  
   - Classifier → Pass/Fail  
   - Regressor → Final grade (G3)  
   - K-Means → Behavior cluster  
5. Interpret drift: G2 - G1 thresholds (>2 positive, < -2 negative, else stable).

## Data Notes
- Source: `student-mat.csv` (395 students, 33 features).  
- No missing values; mixed numeric/categorical.  
- Engineered features: grade deltas, pass/fail label, clusters, cluster drift.

## Limitations / Next Steps
- Perfect scores (100% / R²=1.0) indicate likely overfitting or data leakage—add cross-validation and regularization.  
- Cluster features in app should match training exactly; consider recalculating cluster inputs at inference.  
- Clip regression outputs to [0, 20] for realism.  
- Add model versioning and periodic retraining.

## License
MIT (or specify).***

