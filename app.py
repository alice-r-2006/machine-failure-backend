from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import os

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# -------------------------
# Load ML model files
# -------------------------
model = joblib.load("rul_model.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "ok",
        "message": "✅ Machine Failure Prediction API Running"
    })


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)

        # ✅ Read timeframe from frontend (Option 2)
        selected_timeframe = str(data.get("timeframe", "24h"))

        # ✅ Read values from Lovable
        type_choice = str(data.get("type", "L")).upper()   # L / M / H

        air_temp = float(data.get("air_temp", 300))
        process_temp = float(data.get("process_temp", 310))
        rot_speed = float(data.get("rot_speed", 1500))
        torque = float(data.get("torque", 40))
        tool_wear = float(data.get("tool_wear", 100))

        # ✅ Convert Type into dummy values
        type_m = 1 if type_choice == "M" else 0
        type_h = 1 if type_choice == "H" else 0

        # ✅ Input dict with same feature names as training
        input_dict = {
            "Air temperature [K]": air_temp,
            "Process temperature [K]": process_temp,
            "Rotational speed [rpm]": rot_speed,
            "Torque [Nm]": torque,
            "Tool wear [min]": tool_wear,
            "Type_M": type_m,
            "Type_H": type_h
        }

        input_df = pd.DataFrame([input_dict])

        # ✅ Ensure correct column order
        input_df = input_df.reindex(columns=columns, fill_value=0)

        # ✅ Scale
        scaled_input = scaler.transform(input_df)

        # ✅ Predict probability + prediction
        prob = float(model.predict_proba(scaled_input)[0][1])   # probability of failure
        pred = int(model.predict(scaled_input)[0])              # 0 or 1

        # -------------------------
        # ✅ Option 3: Risk window mapping
        # -------------------------
        if prob >= 0.80:
            risk_window = "High risk of failure within 24 hours"
        elif prob >= 0.50:
            risk_window = "Moderate risk of failure within 7 days"
        elif prob >= 0.30:
            risk_window = "Low–moderate risk of failure within 30 days"
        else:
            risk_window = "Low risk this month"

        return jsonify({
            "prediction": pred,
            "failure_probability": round(prob * 100, 2),
            "result": "Machine Failure Predicted" if pred == 1 else "No Failure Predicted",
            "risk_window": risk_window,
            "selected_timeframe": selected_timeframe
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


# ✅ Deployment safe (Render uses PORT env)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)