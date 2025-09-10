from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

# Load model and scaler
model = joblib.load("injury_model.pkl")
scaler = joblib.load("scaler.pkl")

app = Flask(__name__)

@app.route('/')
def home():
    return "Injury Prediction API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json  # Expecting JSON input
        
        # Example expected features (must match training order!)
        features = [
            data['Training_Hours_Per_Week'],
            data['Stress_Level_Score'],
            data['Sleep_Hours_Per_Night'],
            data['Nutrition_Quality_Score'],
            data['Warmup_Routine_Adherence'],
            data['Previous_Injury_Count'],
            data['Hamstring_Flexibility'],
            data['Balance_Test_Score'],
            data['ACWR']
        ]

        # Scale input
        features_scaled = scaler.transform([features])

        # Make prediction
        prediction = model.predict(features_scaled)[0]
        prob = model.predict_proba(features_scaled)[0][1]

        return jsonify({
            "injury_prediction": int(prediction),  # 0 = No Injury, 1 = Injury
            "injury_probability": float(prob)
        })

    except Exception as e:
        return jsonify({"error": str(e)})
    
    return render_template("index1.html")

if __name__ == "__main__":
    app.run(debug=True)
