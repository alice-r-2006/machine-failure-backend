from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import os

app = Flask(__name__)

# ✅ Allow frontend (Lovable) to call API
CORS(app, resources={r"/*": {"origins": "*"}})

# ✅ Load model files once (fast)
MODEL_PATH = "rul_model.pkl"
SCALER_PATH = "scaler.pkl"
COLUMNS_PATH = "columns.pkl"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
columns = joblib.load(COLUMNS_PATH)


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

        # ✅ Get values safely
        type_choice = str(data.get("type", "L")).upper()   # L / M / H

        air_temp = float(data.get("air_temp", 300))
        process_temp = float(data.get("process_temp", 310))
        rot_speed = float(data.get("rot_speed", 1500))
        torque = float(data.get("torque", 40))
        tool_wear = float(data.get("tool_wear", 100))

        # ✅ One-hot encoding for Type (same as training)
        type_m = 1 if type_choice == "M" else 0
        type_h = 1 if type_choice == "H" else 0

        # ✅ Create input
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

        # ✅ Match exact training column order
        input_df = input_df.reindex(columns=columns, fill_value=0)

        # ✅ Predict
        scaled_input = scaler.transform(input_df)
        pred = int(model.predict(scaled_input)[0])

        # Probability (if available)
        if hasattr(model, "predict_proba"):
            prob = float(model.predict_proba(scaled_input)[0][1])
        else:
            prob = 0.0

        return jsonify({
            "prediction": pred,   # 0 or 1
            "failure_probability": round(prob * 100, 2),
            "result": "Machine Failure Predicted" if pred == 1 else "No Failure Predicted"
        })

    except Exception as e:
        return jsonify({
            "error": str(e),
            "message": "❌ Something went wrong in prediction"
        }), 400


# ✅ Deployment safe runner (Render/Railway use PORT env)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)