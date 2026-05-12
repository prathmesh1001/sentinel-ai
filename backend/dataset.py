import random
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import os

os.makedirs("data", exist_ok=True)

THRESHOLDS = {
    "raspberry_pi": {
        "cpu_temp":  {"warn": 70, "fail": 82},
        "gpu_temp":  {"warn": 72, "fail": 84},
        "cpu_usage": {"warn": 75, "fail": 92},
        "gpu_usage": {"warn": 80, "fail": 95},
        "ram_usage": {"warn": 80, "fail": 93},
        "power":     {"warn": 8,  "fail": 12},
        "voltage":   {"normal": (4.7, 5.3), "warn_low": 4.9, "fail_low": 4.7}
    },
    "jetson_nano": {
        "cpu_temp":  {"warn": 70, "fail": 82},
        "gpu_temp":  {"warn": 72, "fail": 84},
        "cpu_usage": {"warn": 75, "fail": 92},
        "gpu_usage": {"warn": 80, "fail": 95},
        "ram_usage": {"warn": 80, "fail": 93},
        "power":     {"warn": 8,  "fail": 12},
        "voltage":   {"normal": (7.0, 9.0), "warn_low": 7.2, "fail_low": 7.0}
    }
}

COLUMNS = [
    "timestamp", "device_type",
    "cpu_temp", "gpu_temp",
    "cpu_usage", "gpu_usage",
    "voltage", "power_consumption",
    "ram_usage", "health_status"
]

def get_health(row, device):
    t = THRESHOLDS[device]
    has_failure = (
        row["cpu_temp"]  > t["cpu_temp"]["fail"]  or
        row["gpu_temp"]  > t["gpu_temp"]["fail"]  or
        row["cpu_usage"] > t["cpu_usage"]["fail"] or
        row["gpu_usage"] > t["gpu_usage"]["fail"] or
        row["ram_usage"] > t["ram_usage"]["fail"] or
        row["power"]     > t["power"]["fail"]     or
        row["voltage"]   < t["voltage"]["fail_low"]
    )
    has_warning = (
        row["cpu_temp"]  > t["cpu_temp"]["warn"]  or
        row["gpu_temp"]  > t["gpu_temp"]["warn"]  or
        row["cpu_usage"] > t["cpu_usage"]["warn"] or
        row["gpu_usage"] > t["gpu_usage"]["warn"] or
        row["ram_usage"] > t["ram_usage"]["warn"] or
        row["power"]     > t["power"]["warn"]     or
        row["voltage"]   < t["voltage"]["warn_low"]
    )
    if has_failure:
        return "FAILURE"
    elif has_warning:
        return "WARNING"
    return "NORMAL"

def add_noise(val, std, low, high):
    return float(np.clip(val + np.random.normal(0, std), low, high))

def generate_metrics(device, scenario):
    t = THRESHOLDS[device]
    v_min, v_max = t["voltage"]["normal"]

    if scenario == "normal":
        cpu_usage = random.uniform(10, 74)
        gpu_usage = random.uniform(10, 79)
        cpu_temp  = random.uniform(38, 69)
        gpu_temp  = random.uniform(36, 71)
        ram_usage = random.uniform(20, 79)
        power     = random.uniform(2, 7.9)
        voltage   = random.uniform(t["voltage"]["warn_low"] + 0.05, v_max)

    elif scenario == "warning":
        cpu_usage = random.uniform(75, 91)
        gpu_usage = random.uniform(60, 94)
        cpu_temp  = random.uniform(70, 81)
        gpu_temp  = random.uniform(65, 83)
        ram_usage = random.uniform(80, 92)
        power     = random.uniform(8, 11.9)
        voltage   = random.uniform(
            t["voltage"]["fail_low"] + 0.01,
            t["voltage"]["warn_low"] - 0.01
        )

    else:  # failure
        cpu_usage = random.uniform(92, 100)
        gpu_usage = random.uniform(85, 100)
        cpu_temp  = random.uniform(82, 95)
        gpu_temp  = random.uniform(84, 96)
        ram_usage = random.uniform(93, 100)
        power     = random.uniform(12, 15)
        voltage   = random.uniform(v_min - 0.3, t["voltage"]["fail_low"] - 0.01)

    return {
        "cpu_temp":  add_noise(cpu_temp,  1.5, 20, 105),
        "gpu_temp":  add_noise(gpu_temp,  1.5, 20, 105),
        "cpu_usage": add_noise(cpu_usage, 1.0, 0, 100),
        "gpu_usage": add_noise(gpu_usage, 1.0, 0, 100),
        "ram_usage": add_noise(ram_usage, 1.0, 0, 100),
        "power":     add_noise(power,     0.3, 0, 20),
        "voltage":   voltage
    }

def generate_voltage_only_failures(n=400):
    """
    Voltage is the ONLY metric out of range.
    All other metrics perfectly normal.
    Forces model to learn voltage as an independent failure signal.
    """
    data = []
    base_time = datetime.now() - timedelta(days=3)

    for i in range(n):
        device = random.choice(["raspberry_pi", "jetson_nano"])
        t = THRESHOLDS[device]

        cpu_usage = random.uniform(15, 60)
        gpu_usage = random.uniform(15, 60)
        cpu_temp  = random.uniform(38, 62)
        gpu_temp  = random.uniform(36, 60)
        ram_usage = random.uniform(20, 65)
        power     = random.uniform(2.5, 6.5)

        # Voltage below failure threshold ONLY
        if device == "raspberry_pi":
            voltage = random.uniform(4.40, t["voltage"]["fail_low"] - 0.01)
        else:
            voltage = random.uniform(6.50, t["voltage"]["fail_low"] - 0.01)

        timestamp = base_time + timedelta(seconds=i * 15)

        data.append([
            timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            device,
            round(add_noise(cpu_temp,  1.0, 20, 105), 4),
            round(add_noise(gpu_temp,  1.0, 20, 105), 4),
            round(add_noise(cpu_usage, 0.8, 0, 100),  4),
            round(add_noise(gpu_usage, 0.8, 0, 100),  4),
            round(voltage, 4),
            round(add_noise(power, 0.2, 0, 20), 4),
            round(add_noise(ram_usage, 0.8, 0, 100), 4),
            "FAILURE"
        ])
    return data

def generate_voltage_only_warnings(n=300):
    """
    Voltage is in warning range ONLY.
    All other metrics normal.
    """
    data = []
    base_time = datetime.now() - timedelta(days=2)

    for i in range(n):
        device = random.choice(["raspberry_pi", "jetson_nano"])
        t = THRESHOLDS[device]

        cpu_usage = random.uniform(15, 60)
        gpu_usage = random.uniform(15, 60)
        cpu_temp  = random.uniform(38, 62)
        gpu_temp  = random.uniform(36, 60)
        ram_usage = random.uniform(20, 65)
        power     = random.uniform(2.5, 6.5)

        if device == "raspberry_pi":
            voltage = random.uniform(
                t["voltage"]["fail_low"] + 0.01,
                t["voltage"]["warn_low"] - 0.01
            )
        else:
            voltage = random.uniform(
                t["voltage"]["fail_low"] + 0.01,
                t["voltage"]["warn_low"] - 0.01
            )

        timestamp = base_time + timedelta(seconds=i * 15)

        data.append([
            timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            device,
            round(add_noise(cpu_temp,  1.0, 20, 105), 4),
            round(add_noise(gpu_temp,  1.0, 20, 105), 4),
            round(add_noise(cpu_usage, 0.8, 0, 100),  4),
            round(add_noise(gpu_usage, 0.8, 0, 100),  4),
            round(voltage, 4),
            round(add_noise(power, 0.2, 0, 20), 4),
            round(add_noise(ram_usage, 0.8, 0, 100), 4),
            "WARNING"
        ])
    return data

def generate(n=5000):
    data = []

    # Realistic distribution: 60% normal, 25% warning, 15% failure
    scenarios = random.choices(
        ["normal", "warning", "failure"],
        weights=[0.60, 0.25, 0.15],
        k=n
    )

    # Timestamps spread over last 7 days
    base_time = datetime.now() - timedelta(days=7)

    for i, scenario in enumerate(scenarios):
        device  = random.choice(["raspberry_pi", "jetson_nano"])
        metrics = generate_metrics(device, scenario)
        health  = get_health(metrics, device)
        ts      = base_time + timedelta(seconds=i * 10)

        data.append([
            ts.strftime("%Y-%m-%d %H:%M:%S"),
            device,
            round(metrics["cpu_temp"],  4),
            round(metrics["gpu_temp"],  4),
            round(metrics["cpu_usage"], 4),
            round(metrics["gpu_usage"], 4),
            round(metrics["voltage"],   4),
            round(metrics["power"],     4),
            round(metrics["ram_usage"], 4),
            health
        ])

    df = pd.DataFrame(data, columns=COLUMNS)

    # Append voltage-specific scenarios
    df_vf = pd.DataFrame(generate_voltage_only_failures(400), columns=COLUMNS)
    df_vw = pd.DataFrame(generate_voltage_only_warnings(300), columns=COLUMNS)

    df = pd.concat([df, df_vf, df_vw], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    df.to_csv("data/device_data.csv", index=False)

    print("\n✅ Dataset generated successfully!")
    print(f"📁 Saved to: data/device_data.csv")
    print(f"📊 Total rows: {len(df)}")
    print("\n📊 Class distribution:")
    print(df["health_status"].value_counts())
    print("\n📊 Device distribution:")
    print(df["device_type"].value_counts())

if __name__ == "__main__":
    generate(5000)