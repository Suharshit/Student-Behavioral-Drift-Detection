from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__, template_folder='template')

# Load models
scaler = joblib.load("models/scaler.pkl")
classifier = joblib.load("models/classifier.pkl")
regressor = joblib.load("models/regressor.pkl")
kmeans = joblib.load("models/kmeans.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get input values with validation
        studytime = float(request.form.get("studytime", 0))
        absences = float(request.form.get("absences", 0))
        failures = float(request.form.get("failures", 0))
        G1 = float(request.form.get("G1", 0))
        G2 = float(request.form.get("G2", 0))
        
        # Validate input ranges
        if studytime < 0 or studytime > 20:
            return render_template("index.html", error="Study time must be between 0 and 20 hours")
        if absences < 0 or absences > 100:
            return render_template("index.html", error="Absences must be between 0 and 100")
        if failures < 0 or failures > 10:
            return render_template("index.html", error="Failures must be between 0 and 10")
        if G1 < 0 or G1 > 20:
            return render_template("index.html", error="G1 must be between 0 and 20")
        if G2 < 0 or G2 > 20:
            return render_template("index.html", error="G2 must be between 0 and 20")

        # Drift features
        grade_change_1_2 = G2 - G1
        grade_change_2_3 = 0  # unknown, predicted indirectly
        cluster_drift = 0     # initial assumption

        features = np.array([[studytime, absences, failures,
                             G1, G2, grade_change_1_2,
                             grade_change_2_3, 0, cluster_drift]])

        features_scaled = scaler.transform(features)

        # Predictions
        risk = classifier.predict(features_scaled)[0]
        final_grade = regressor.predict(features_scaled)[0]
        cluster = kmeans.predict(features_scaled[:, :5])[0]
        
        # Determine drift status
        if grade_change_1_2 > 2:
            drift_status = "Positive"
            drift_description = "Significant improvement in performance"
        elif grade_change_1_2 < -2:
            drift_status = "Negative"
            drift_description = "Significant decline in performance"
        else:
            drift_status = "Stable"
            drift_description = "Performance remains relatively stable"

        return render_template(
            "result.html",
            risk="Pass" if risk == 1 else "Fail",
            final_grade=round(final_grade, 2),
            cluster=int(cluster),
            drift=round(grade_change_1_2, 2),
            drift_status=drift_status,
            drift_description=drift_description,
            studytime=studytime,
            absences=absences,
            failures=failures,
            G1=G1,
            G2=G2
        )
    except ValueError as e:
        return render_template("index.html", error="Please enter valid numbers for all fields")
    except Exception as e:
        return render_template("index.html", error=f"An error occurred: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)