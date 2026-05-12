from dotenv import load_dotenv
import os
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
import random
from chatbot import register_chatbot

# ===============================
# PATH SETUP
# ===============================
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR  = os.path.abspath(os.path.join(BASE_DIR, ".."))
MODEL_DIR    = os.path.join(PROJECT_DIR, "model")
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR   = os.path.join(BASE_DIR, "static")
DATA_DIR     = os.path.join(PROJECT_DIR, "data")

# ===============================
# FLASK APP
# ===============================
app = Flask(
    __name__,
    template_folder=TEMPLATE_DIR,
    static_folder=STATIC_DIR
)
CORS(app)

register_chatbot(app)
print("📋 All routes:", [str(r) for r in app.url_map.iter_rules()])
# ===============================
# LOAD MODELS
# ===============================
try:
    model          = joblib.load(os.path.join(MODEL_DIR, "rf_model.pkl"))
    device_encoder = joblib.load(os.path.join(MODEL_DIR, "device_encoder.pkl"))
    label_encoder  = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
    print("✅ Models loaded successfully")

    if None in [model, device_encoder, label_encoder]:
        raise RuntimeError("One or more model files failed to load")

except Exception as e:
    print(f"❌ Model loading failed: {e}")
    model          = None
    device_encoder = None
    label_encoder  = None

# ===============================
# LOAD DATASET for live feed
# ===============================
dataset    = None
_rpi_idx    = 0
_jetson_idx = 0

def _load_dataset(path):
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
            print(f"✅ Dataset loaded: {path}  ({len(df)} rows)")
            return df.reset_index(drop=True)
        except Exception as e:
            print(f"⚠ Could not load dataset {path}: {e}")
    else:
        print(f"⚠ Dataset not found: {path}")
    return None

# Load single shared dataset (device_data.csv)
for candidate in ["live_data.csv", "device_data.csv", "rpi_data.csv", "dataset.csv"]:
    p = os.path.join(DATA_DIR, candidate)
    if os.path.exists(p):
        dataset = _load_dataset(p)
        break

# If still None, try any CSV in the data folder
if dataset is None and os.path.exists(DATA_DIR):
    csvs = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]
    if csvs:
        dataset = _load_dataset(os.path.join(DATA_DIR, csvs[0]))
        print(f"ℹ Using {csvs[0]} as dataset")

# ===============================
# HOME PAGE
# ===============================
@app.route("/")
def home():
    return render_template("index.html")

# ===============================
# STREAM DATA
# ===============================
@app.route("/stream-data")
def stream_data():
    global _rpi_idx, _jetson_idx

    device = request.args.get("device", "rpi").lower()

    # If no dataset loaded, return simulated data
    if dataset is None or len(dataset) == 0:
        return jsonify(_simulate_row(device))

    # ✅ FIX: Filter rows by device type
    device_map  = {"jetson": "jetson_nano", "rpi": "raspberry_pi"}
    device_name = device_map.get(device, "raspberry_pi")
    filtered_df = dataset[dataset["device_type"] == device_name]

    if len(filtered_df) == 0:
        print(f"⚠ No rows found for device_type='{device_name}', using full dataset")
        filtered_df = dataset

    filtered_df = filtered_df.reset_index(drop=True)

    # Advance index per device
    if device == "jetson":
        idx = _jetson_idx
        _jetson_idx = (_jetson_idx + 1) % len(filtered_df)
    else:
        idx = _rpi_idx
        _rpi_idx = (_rpi_idx + 1) % len(filtered_df)

    row = filtered_df.iloc[idx]

    def col(row, *names, default=0.0):
        for n in names:
            if n in row.index:
                v = row[n]
                try:
                    return float(v)
                except (ValueError, TypeError):
                    pass
        return float(default)

    data = {
        "cpu_temp":          col(row, "cpu_temp", "cpu_temperature", "CPU_Temp"),
        "gpu_temp":          col(row, "gpu_temp", "gpu_temperature", "GPU_Temp"),
        "cpu_usage":         col(row, "cpu_usage", "cpu_use", "CPU_Usage"),
        "gpu_usage":         col(row, "gpu_usage", "gpu_use", "GPU_Usage"),
        "ram_usage":         col(row, "ram_usage", "ram", "RAM_Usage", "memory_usage"),
        "power_consumption": col(row, "power_consumption", "power", "Power_W"),
        "voltage":           col(row, "voltage", "volt", "Voltage_V"),
        "row_index":         int(idx),
        "total_rows":        int(len(filtered_df)),
    }

    return jsonify(data)

def _simulate_row(device):
    if device == "jetson":
        base = dict(cpu_temp=55, gpu_temp=62, cpu_usage=50, gpu_usage=58,
                    ram_usage=64, power_consumption=9.8, voltage=7.6)  # ✅ FIX: was 5.0
    else:
        base = dict(cpu_temp=45, gpu_temp=38, cpu_usage=40, gpu_usage=35,
                    ram_usage=52, power_consumption=4.2, voltage=5.1)

    def jitter(v, pct=0.08):
        return round(v + v * (random.random() - 0.5) * pct, 2)

    return {k: jitter(v) for k, v in base.items()}

# ===============================
# PREDICT API
# ===============================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        if model is None:
            return jsonify({"error": "Model not loaded"}), 500

        data = request.get_json(force=True, silent=True)
        if not data:
            return jsonify({"error": "Invalid or missing JSON body"}), 400

        print("📥 Incoming:", data)

        def get(key, default=0.0):
            val = data.get(key, default)
            try:
                return float(val)
            except (ValueError, TypeError):
                return float(default)

        features = {
            "cpu_temp":          get("cpu_temp"),
            "gpu_temp":          get("gpu_temp"),
            "cpu_usage":         get("cpu_usage"),
            "gpu_usage":         get("gpu_usage"),
            "ram_usage":         get("ram_usage"),
            "power_consumption": get("power_consumption"),
            "voltage":           get("voltage"),
            "temp_diff":         get("temp_diff"),
            "load_index":        get("load_index"),
        }

        df = pd.DataFrame([features])

        # Device encoding
        if device_encoder is not None:
            device_type = data.get("device_type", "unknown")

            # ✅ FIX: map app.js device keys to dataset device names
            device_map  = {"jetson": "jetson_nano", "rpi": "raspberry_pi"}
            device_type = device_map.get(device_type, device_type)

            try:
                if device_type not in device_encoder.classes_:
                    device_type = device_encoder.classes_[0]
                    print(f"⚠ Unknown device type, falling back to: {device_type}")
                device_encoded = device_encoder.transform([device_type])[0]
            except Exception as e:
                print("⚠ Device encoding failed:", e)
                device_encoded = 0
        else:
            device_encoded = 0

        df["device_type"] = device_encoded

        # Feature alignment
        if hasattr(model, "feature_names_in_"):
            for col_name in model.feature_names_in_:
                if col_name not in df.columns:
                    df[col_name] = 0
            df = df[model.feature_names_in_]

        # Prediction
        pred = model.predict(df)[0]

        if label_encoder is not None:
            try:
                pred_label = label_encoder.inverse_transform([pred])[0]
            except Exception as e:
                print("⚠ Label decode failed:", e)
                pred_label = str(pred)
        else:
            pred_label = str(pred)

        if hasattr(model, "predict_proba"):
            probs      = model.predict_proba(df)[0]
            confidence = float(max(probs) * 100)

            if hasattr(model, "classes_"):
                classes  = list(model.classes_)
                risk_idx = None
                for candidate in [1, "failure", "Failure", "FAILURE"]:
                    if candidate in classes:
                        risk_idx = classes.index(candidate)
                        break
                failure_risk = float(probs[risk_idx] * 100) if risk_idx is not None else float(100 - confidence)
            else:
                failure_risk = float(100 - confidence)
        else:
            confidence   = 100.0
            failure_risk = 0.0

        return jsonify({
            "prediction":   pred_label,
            "confidence":   round(confidence, 2),
            "failure_risk": round(failure_risk, 2),
        })

    except Exception as e:
        print("🔥 ERROR:", str(e))
        return jsonify({"error": str(e)}), 500

# ===============================
# RUN SERVER
# ===============================
if __name__ == "__main__":
    print(f"\n📁 BASE_DIR     : {BASE_DIR}")
    print(f"📁 TEMPLATE_DIR : {TEMPLATE_DIR}")
    print(f"📁 STATIC_DIR   : {STATIC_DIR}")
    print(f"📁 MODEL_DIR    : {MODEL_DIR}")
    print(f"📁 DATA_DIR     : {DATA_DIR}\n")

    for name, path in [("Templates", TEMPLATE_DIR), ("Static", STATIC_DIR),
                       ("Models", MODEL_DIR), ("Data", DATA_DIR)]:
        status = "✅" if os.path.exists(path) else "❌ NOT FOUND"
        print(f"{status} {name}: {path}")

    print("\n🚀 Starting SENTINEL on http://127.0.0.1:8080\n")
    app.run(debug=True, host="127.0.0.1", port=8080)