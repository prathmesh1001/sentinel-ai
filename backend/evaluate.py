import os, sys
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))

model      = joblib.load(os.path.join(PROJECT_DIR, "model", "rf_model.pkl"))
device_enc = joblib.load(os.path.join(PROJECT_DIR, "model", "device_encoder.pkl"))
label_enc  = joblib.load(os.path.join(PROJECT_DIR, "model", "label_encoder.pkl"))

df = pd.read_csv(os.path.join(PROJECT_DIR, "data", "live_data.csv"))

# Feature engineering (must match ml_model.py exactly)
df["temp_diff"]        = df["cpu_temp"] - df["gpu_temp"]
df["load_index"]       = (df["cpu_usage"] + df["gpu_usage"]) / 2
df["voltage_stress"]   = df.apply(lambda r:
    1 if (r["device_type"] == "raspberry_pi" and r["voltage"] < 4.9) or
         (r["device_type"] == "jetson_nano"  and r["voltage"] < 7.2)
    else 0, axis=1)
df["thermal_pressure"] = df["cpu_temp"] * 0.55 + df["gpu_temp"] * 0.45

df["device_type"]   = device_enc.transform(df["device_type"])
df["health_status"] = label_enc.transform(df["health_status"])

features = ["device_type","cpu_temp","gpu_temp","cpu_usage","gpu_usage",
            "voltage","power_consumption","ram_usage",
            "temp_diff","load_index","voltage_stress","thermal_pressure"]

X = df[features]
y = df["health_status"]

pred = model.predict(X)

print("\n📊 ACCURACY:", round(accuracy_score(y, pred) * 100, 2), "%")

print("\n📉 CONFUSION MATRIX:")
print(confusion_matrix(y, pred))

print("\n📋 CLASSIFICATION REPORT:")
print(classification_report(y, pred, target_names=label_enc.classes_))