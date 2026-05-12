import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE

# -------------------------
# LOAD DATA
# -------------------------
df = pd.read_csv("data/live_data.csv")

print("\n📊 Class Distribution (Raw):")
print(df["health_status"].value_counts())

# -------------------------
# SENSOR NOISE INJECTION
# Simulates real-world drift — prevents model memorizing clean synthetic patterns
# -------------------------
np.random.seed(None)  # different noise every run

df["cpu_temp"]           += np.random.normal(0, 2.5, len(df))
df["gpu_temp"]           += np.random.normal(0, 2.5, len(df))
df["cpu_usage"]          += np.random.normal(0, 2.0, len(df))
df["gpu_usage"]          += np.random.normal(0, 2.0, len(df))
df["voltage"]            += np.random.normal(0, 0.15, len(df))
df["power_consumption"]  += np.random.normal(0, 0.8, len(df))
df["ram_usage"]          += np.random.normal(0, 2.0, len(df))

# Clip to physical limits
df["cpu_temp"]          = df["cpu_temp"].clip(20, 105)
df["gpu_temp"]          = df["gpu_temp"].clip(20, 105)
df["cpu_usage"]         = df["cpu_usage"].clip(0, 100)
df["gpu_usage"]         = df["gpu_usage"].clip(0, 100)
df["ram_usage"]         = df["ram_usage"].clip(0, 100)
df["power_consumption"] = df["power_consumption"].clip(0, 20)

# -------------------------
# FEATURE ENGINEERING
# -------------------------
df["temp_diff"]     = df["cpu_temp"] - df["gpu_temp"]
df["load_index"]    = (df["cpu_usage"] + df["gpu_usage"]) / 2

# New: voltage stress flag — gives model explicit voltage signal
df["voltage_stress"] = (
    df.apply(lambda r:
        1 if (r["device_type"] == "raspberry_pi" and r["voltage"] < 4.9) or
             (r["device_type"] == "jetson_nano"  and r["voltage"] < 7.2)
        else 0, axis=1)
)

# New: thermal pressure — combined heat signal
df["thermal_pressure"] = (df["cpu_temp"] * 0.55) + (df["gpu_temp"] * 0.45)

# -------------------------
# ENCODING
# -------------------------
device_enc = LabelEncoder()
df["device_type"] = device_enc.fit_transform(df["device_type"])

label_enc = LabelEncoder()
df["health_status"] = label_enc.fit_transform(df["health_status"])

print("\n🔎 Label Mapping:", list(label_enc.classes_))

failure_class = list(label_enc.classes_).index("FAILURE")
warning_class = list(label_enc.classes_).index("WARNING")
normal_class  = list(label_enc.classes_).index("NORMAL")

print(f"🚨 FAILURE: {failure_class} | ⚠️ WARNING: {warning_class} | ✅ NORMAL: {normal_class}")

# -------------------------
# FEATURES
# -------------------------
features = [
    "device_type",
    "cpu_temp",
    "gpu_temp",
    "cpu_usage",
    "gpu_usage",
    "voltage",
    "power_consumption",
    "ram_usage",
    "temp_diff",
    "load_index",
    "voltage_stress",     # new
    "thermal_pressure"    # new
]

X = df[features]
y = df["health_status"]

# -------------------------
# HOLD-OUT SPLIT
# 70% train | 15% val | 15% test
# Prevents test leakage into model selection
# -------------------------
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp
)
# 0.176 of 85% ≈ 15% of total

print(f"\n📦 Split — Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

# -------------------------
# SMOTE — ONLY ON TRAIN
# Do NOT balance perfectly: keep slight natural imbalance
# -------------------------
minority_count = y_train.value_counts().min()
majority_count = y_train.value_counts().max()

# Target: bring minority up to 70% of majority — not 100%
smote_target = int(majority_count * 0.70)
smote_target = max(smote_target, minority_count)

sampling_strategy = {
    cls: smote_target
    for cls in y_train.unique()
    if y_train.value_counts()[cls] < smote_target
}

sm = SMOTE(random_state=42, sampling_strategy=sampling_strategy, k_neighbors=4)
X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)

print("\n📊 After SMOTE (Train Only):")
print(pd.Series(y_train_sm).value_counts())

# -------------------------
# MODEL — Regularized to reduce overfit
# max_depth reduced, min_samples_leaf increased
# -------------------------
class_weights = {
    failure_class: 3,    # heavily penalize missing failures
    warning_class: 2,
    normal_class:  1
}

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,           # was 16 — reduced to limit overfit
    min_samples_leaf=4,     # was 1 — forces generalization
    min_samples_split=8,    # new — prevents tiny splits memorizing noise
    max_features="sqrt",    # each tree sees random feature subset
    class_weight=class_weights,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_sm, y_train_sm)

# -------------------------
# VALIDATION CHECK (before test)
# -------------------------
val_pred = model.predict(X_val)
val_acc  = accuracy_score(y_val, val_pred)
print(f"\n🔍 Validation Accuracy: {val_acc:.4f}")

if val_acc > 0.997:
    print("⚠️  Val accuracy suspiciously high — possible overfit. Consider more noise or depth reduction.")
elif val_acc < 0.88:
    print("⚠️  Val accuracy low — model may be underfitting. Check data quality.")
else:
    print("✅  Validation looks healthy.")

# -------------------------
# TEST EVALUATION
# -------------------------
pred = model.predict(X_test)

print(f"\n✅ Test Accuracy: {accuracy_score(y_test, pred):.4f}")

cm = confusion_matrix(y_test, pred)
print("\n📉 Confusion Matrix:")
print(cm)

print("\n📊 Classification Report:")
print(classification_report(y_test, pred, zero_division=0))

# -------------------------
# FAILURE / WARNING DETECTION
# -------------------------
failure_total   = cm[failure_class].sum()
failure_correct = cm[failure_class][failure_class]
warning_total   = cm[warning_class].sum()
warning_correct = cm[warning_class][warning_class]

print(f"\n🚨 FAILURE Detection Rate: {(failure_correct/failure_total)*100:.2f}%")
print(f"⚠️  WARNING Detection Rate: {(warning_correct/warning_total)*100:.2f}%")

# Overfit gap check
train_acc = accuracy_score(y_train_sm, model.predict(X_train_sm))
test_acc  = accuracy_score(y_test, pred)
gap       = train_acc - test_acc

print(f"\n📐 Overfit Gap (train - test): {gap:.4f}")
if gap > 0.05:
    print("🚨 Overfit detected — gap too large. Reduce max_depth or increase min_samples_leaf.")
elif gap > 0.02:
    print("⚠️  Mild overfit — acceptable for production but monitor.")
else:
    print("✅  No significant overfit detected.")

# -------------------------
# CROSS VALIDATION — on noisy data
# -------------------------
print("\n🔁 Running Stratified Cross Validation...")
skf    = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=skf, n_jobs=1)

print(f"CV Scores: {scores}")
print(f"Mean CV:   {scores.mean():.4f} ± {scores.std():.4f}")

if scores.std() > 0.02:
    print("⚠️  High variance across folds — model is unstable.")
else:
    print("✅  Low variance — model is stable across folds.")

# -------------------------
# FEATURE IMPORTANCE
# -------------------------
importance = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
print("\n🔥 Feature Importance:")
print(importance)

if importance["voltage"] < 0.02:
    print("⚠️  Voltage still underweighted — voltage_stress feature may need more samples.")

# -------------------------
# SAVE
# -------------------------
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/rf_model.pkl")
joblib.dump(device_enc, "model/device_encoder.pkl")
joblib.dump(label_enc, "model/label_encoder.pkl")

print("\n💾 Model saved successfully!")