"""
========================================================================================
Synthetic Manufacturing WIP & Bottleneck Dataset Generator (Zero Company Data Leakage)
========================================================================================
Generates realistic, anonymized multi-stage factory time-series data for testing
and demonstrating the TimesNet Forecasting and Bottleneck Detection models.
========================================================================================
"""

import os
import sys
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def generate_synthetic_factory_data(
    num_hours: int = 1500,
    output_path: str = "data/mock_sample_timeseries.csv",
    seed: int = 42
):
    np.random.seed(seed)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 1. Timeline Generation (Hourly)
    start_dt = pd.Timestamp("2026-06-01 00:00:00")
    timestamps = [start_dt + pd.Timedelta(hours=i) for i in range(num_hours)]
    df = pd.DataFrame({"Timestamp": timestamps})

    # 2. Exogenous Temporal Features
    df["Hour"] = df["Timestamp"].dt.hour
    df["DayOfWeek"] = df["Timestamp"].dt.dayofweek
    df["DayOfMonth"] = df["Timestamp"].dt.day

    df["Hour_Sin"] = np.sin(2 * np.pi * df["Hour"] / 24.0)
    df["Hour_Cos"] = np.cos(2 * np.pi * df["Hour"] / 24.0)
    df["DOW_Sin"] = np.sin(2 * np.pi * df["DayOfWeek"] / 7.0)
    df["DOW_Cos"] = np.cos(2 * np.pi * df["DayOfWeek"] / 7.0)
    df["DOM_Sin"] = np.sin(2 * np.pi * (df["DayOfMonth"] - 1) / 31.0)
    df["DOM_Cos"] = np.cos(2 * np.pi * (df["DayOfMonth"] - 1) / 31.0)

    # Shift A (07:00 - 19:00), Shift B (19:00 - 07:00)
    df["Shift_A_Flag"] = ((df["Hour"] >= 7) & (df["Hour"] < 19)).astype(float)
    df["Is_Month_End"] = (df["DayOfMonth"] >= 25).astype(float)
    df["Week_Of_Month"] = ((df["DayOfMonth"] - 1) // 7 + 1).astype(float)

    # Synthetic Customer Distribution Ratios
    df["Cust_A_Ratio"] = 0.55 + 0.08 * np.sin(np.linspace(0, 10, num_hours))
    df["Cust_B_Ratio"] = 0.30 + 0.05 * np.cos(np.linspace(0, 8, num_hours))
    df["Cust_C_Ratio"] = np.clip(1.0 - df["Cust_A_Ratio"] - df["Cust_B_Ratio"], 0.05, 0.40)

    # 3. Flow Dynamics (Inflow & Outflow with Daily Cyclicity)
    base_flow = 800.0 + 400.0 * df["Shift_A_Flag"] + 250.0 * np.sin(2 * np.pi * df["Hour"] / 24.0)
    noise = np.random.normal(0, 80, num_hours)
    df["Inflow_Qty"] = np.clip(base_flow + noise, 100.0, None)
    df["Outflow_Qty"] = np.clip(df["Inflow_Qty"].shift(12).fillna(df["Inflow_Qty"].mean()) + np.random.normal(0, 70, num_hours), 80.0, None)
    df["Net_Line_Flow_Roll6"] = (df["Inflow_Qty"] - df["Outflow_Qty"]).rolling(6, min_periods=1).mean()
    df["Avg_Inflow_HPU"] = 0.045 + 0.012 * np.random.uniform(0.8, 1.2, num_hours)

    # 4. Multi-Stage Anonymized Station WIP (Process A to Process E)
    stations = ["WIP_Process_A", "WIP_Process_B", "WIP_Process_C", "WIP_Process_D", "WIP_Process_E"]
    station_baselines = {
        "WIP_Process_A": 2200.0,  # Kitting
        "WIP_Process_B": 1800.0,  # Washing
        "WIP_Process_C": 1500.0,  # Bonding
        "WIP_Process_D": 1400.0,  # Damper
        "WIP_Process_E": 1100.0   # OQA
    }

    # Simulate realistic production lag and bottleneck congestion waves
    cum_wip = np.zeros(num_hours)
    for idx, st in enumerate(stations):
        base = station_baselines[st]
        drift = 300.0 * np.sin(np.linspace(idx * 2, (idx + 1) * 15, num_hours))
        st_noise = np.random.normal(0, 100, num_hours)
        congestion_surge = np.where((df["Hour"] >= 14) & (df["Hour"] <= 20) & (df["DayOfWeek"] <= 4), 350.0, 0.0)
        df[st] = np.clip(base + drift + st_noise + congestion_surge, 150.0, None)
        cum_wip += df[st].values

        # Physics & Capacity Engineered Features
        df[f"Eff_UPH_{st}"] = 120.0 + 15.0 * np.random.randn(num_hours)
        df[f"Cap_DOS_{st}"] = df[st] / (df[f"Eff_UPH_{st}"] * 21.0 + 1e-4)
        df[f"Mature_Ratio_{st}"] = np.clip(0.40 + 0.25 * np.sin(np.linspace(0, 20, num_hours)) + np.random.normal(0, 0.05, num_hours), 0.05, 0.95)
        df[f"Mature_WIP_{st}"] = df[st] * df[f"Mature_Ratio_{st}"]
        df[f"Fresh_WIP_{st}"] = df[st] - df[f"Mature_WIP_{st}"]
        df[f"Expected_Cohort_{st}"] = df[f"Fresh_WIP_{st}"] + df[st].shift(4).fillna(df[st].mean()) * 0.4

        if idx > 0:
            prev_st = stations[idx - 1]
            df[f"Net_UPH_Gap_{st}"] = df[f"Eff_UPH_{prev_st}"] - df[f"Eff_UPH_{st}"]
            df[f"Upstream_MC_Ratio_{st}"] = (df[st] + 1.0) / (df[prev_st] + 1.0)

    df["Total_WIP"] = cum_wip
    df["Total_Eff_UPH"] = 150.0 + 10.0 * np.random.randn(num_hours)
    df["Total_Cap_DOS"] = df["Total_WIP"] / (df["Total_Eff_UPH"] * 21.0 + 1e-4)
    df["Total_Plan_DOS"] = df["Total_Cap_DOS"] * np.random.uniform(0.9, 1.1, num_hours)

    # Save Cleaned Synthetic Demonstration Dataset
    df.to_csv(output_path, index=False)
    print(f" Synthetic Demonstration Dataset generated: {output_path}")
    print(f" Timestamps: {len(df):,} hourly records | Columns: {len(df.columns)}")


if __name__ == "__main__":
    generate_synthetic_factory_data()
