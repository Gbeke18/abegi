from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np

# Load model (and scaler if you saved it)
model = joblib.load("injury_model.pkl")
# scaler = joblib.load("scaler.pkl")   # Uncomment if using scaling

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")   # Load the HTML form

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        data = request.form  

        # Extract features (same order as training)
        features = [
            float(data['Training_Hours_Per_Week']),
            float(data['Stress_Level_Score']),
            float(data['Sleep_Hours_Per_Night']),
            float(data['Nutrition_Quality_Score']),
            int(data['Warmup_Routine_Adherence']),
            int(data['Previous_Injury_Count']),
            float(data['Hamstring_Flexibility']),
            float(data['Balance_Test_Score']),
            float(data['ACWR'])
        ]

        # If you used a scaler:
        # features_scaled = scaler.transform([features])
        # prediction = model.predict(features_scaled)[0]
        # prob = model.predict_proba(features_scaled)[0][1]

        # If NO scaler (RandomForest doesn’t require it):
        prediction = model.predict([features])[0]
        prob = model.predict_proba([features])[0][1]

        result = "Injury Risk" if prediction == 1 else "No Injury Risk"

        return render_template("index.html",
                               prediction_text=f"{result} (Probability: {prob:.2f})")

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
