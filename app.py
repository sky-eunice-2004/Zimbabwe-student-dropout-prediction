from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__)

# ==============================
# LOAD MODEL + SCALER (IMPORTANT)
# ==============================
model = pickle.load(open('model.pkl', 'rb'))

# Load scaler if you used scaling during training
scaler = None
if os.path.exists('scaler.pkl'):
    scaler = pickle.load(open('scaler.pkl', 'rb'))

# ==============================
# HOME PAGE
# ==============================
@app.route('/')
def home():
    return render_template('index.html')

# ==============================
# PREDICTION ROUTE
# ==============================
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # ------------------------------
        # GET INPUTS
        # ------------------------------
        age = float(request.form.get('age', 0))
        attendance = float(request.form.get('attendance_rate', 0))
        study_hours = float(request.form.get('study_hours', 0))
        stress = float(request.form.get('stress_level', 0))

        # ------------------------------
        # INPUT VALIDATION (VERY IMPORTANT)
        # ------------------------------
        if not (15 <= age <= 60):
            return render_template('index.html',
                                   prediction_text="❌ Age must be between 15 and 60")

        if not (0 <= attendance <= 100):
            return render_template('index.html',
                                   prediction_text="❌ Attendance must be between 0 and 100")

        if not (0 <= study_hours <= 24):
            return render_template('index.html',
                                   prediction_text="❌ Study hours must be between 0 and 24")

        if not (0 <= stress <= 10):
            return render_template('index.html',
                                   prediction_text="❌ Stress level must be between 0 and 10")

        # ------------------------------
        # PREPARE FEATURES
        # ------------------------------
        features = np.array([[age, attendance, study_hours, stress]])

        # ------------------------------
        # SCALE DATA (IF USED IN TRAINING)
        # ------------------------------
        if scaler is not None:
            features = scaler.transform(features)

        # ------------------------------
        # PREDICTION
        # ------------------------------
        prediction = model.predict(features)[0]

        # Probability (if available)
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(features)[0][1]
            result = f"{'Student will Dropout ❌' if prediction == 1 else 'Student will Continue ✅'} (Risk: {prob:.2f})"
        else:
            result = "Student will Dropout ❌" if prediction == 1 else "Student will Continue ✅"

        return render_template('index.html', prediction_text=result)

    # ------------------------------
    # ERROR HANDLING
    # ------------------------------
    except ValueError:
        return render_template('index.html',
                               prediction_text="❌ Please enter valid numeric values")

    except Exception as e:
        return render_template('index.html',
                               prediction_text=f"❌ Error: {str(e)}")

# ==============================
# RUN APP
# ==============================
if __name__ == "__main__":
    app.run(debug=True)