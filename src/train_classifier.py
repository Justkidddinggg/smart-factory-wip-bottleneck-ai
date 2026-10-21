"""
========================================================================================
SMART FACTORY AI: 6-HOUR BOTTLENECK EARLY-WARNING CLASSIFIER
========================================================================================
Architecture: Balanced Random Forest with Physics-Informed Capacity Features
Evaluates 6-hour predictive line congestion on imbalanced production queues.
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
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, fbeta_score,
    roc_auc_score, average_precision_score, confusion_matrix
)


def prepare_bottleneck_dataset(csv_path: str, horizon_hours: int = 6):
    df = pd.read_csv(csv_path)
    df["Timestamp_dt"] = pd.to_datetime(df["Timestamp"])

    stations = ["Total_WIP", "WIP_Process_A", "WIP_Process_B", "WIP_Process_C", "WIP_Process_D", "WIP_Process_E"]

    # 1. 72-Hour Causal Rolling Z-Scores
    for st in stations:
        s_val = df[st].astype(float)
        rm72 = s_val.rolling(72, min_periods=1).mean()
        rs72 = s_val.rolling(72, min_periods=1).std().fillna(1.0) + 1.0

        df[f"{st}_RollMean72"] = rm72
        df[f"{st}_RollStd72"] = rs72
        df[f"{st}_Local_Z_t"] = (s_val - rm72) / rs72
        df[f"{st}_Roll6_Z_t"] = (s_val.rolling(6, min_periods=1).mean() - rm72) / rs72
        df[f"{st}_Vel6_Z"] = s_val.diff(6).fillna(0.0) / rs72

    # 2. Ground-Truth Bottleneck Labeling (Congestion in future horizon >= 75th percentile)
    n_total = len(df)
    train_cutoff = int((n_total - horizon_hours) * 0.70)
    label_cols = {}

    for st in stations:
        fut_mean = sum(df[st].shift(-h) for h in range(1, horizon_hours + 1)) / float(horizon_hours)
        fut_z = (fut_mean - df[f"{st}_RollMean72"]) / df[f"{st}_RollStd72"]
        crit_thresh = float(np.nanpercentile(fut_z.iloc[:train_cutoff], 75.0))
        df[f"Target_Bottleneck_{st}"] = (fut_z >= crit_thresh).astype(int)
        label_cols[st] = f"Target_Bottleneck_{st}"

    df_valid = df.iloc[:-horizon_hours].copy().reset_index(drop=True)
    return df_valid, label_cols, stations


def compute_metrics(y_true, probs, threshold=0.50):
    preds = (probs >= threshold).astype(int)
    return {
        "Accuracy": accuracy_score(y_true, preds) * 100.0,
        "Precision": precision_score(y_true, preds, zero_division=0) * 100.0,
        "Recall": recall_score(y_true, preds, zero_division=0) * 100.0,
        "F1_Score": f1_score(y_true, preds, zero_division=0) * 100.0,
        "F2_Score": fbeta_score(y_true, preds, beta=2.0, zero_division=0) * 100.0,
        "ROC_AUC": roc_auc_score(y_true, probs) * 100.0 if len(np.unique(y_true)) > 1 else 50.0,
        "PR_AUC": average_precision_score(y_true, probs) * 100.0 if np.sum(y_true) > 0 else 0.0
    }


def run_bottleneck_classification():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if base_dir not in sys.path:
        sys.path.insert(0, base_dir)
    csv_path = os.path.join(base_dir, "data", "mock_sample_timeseries.csv")

    if not os.path.exists(csv_path):
        from data.generate_mock_data import generate_synthetic_factory_data
        generate_synthetic_factory_data(output_path=csv_path)

    df, label_cols, stations = prepare_bottleneck_dataset(csv_path, horizon_hours=6)
    print(f" Loaded dataset: {len(df):,} hourly records across {len(stations)} production stages.")

    common_features = [
        "Net_Line_Flow_Roll6", "Avg_Inflow_HPU",
        "Hour_Sin", "Hour_Cos", "DOW_Sin", "DOW_Cos", "Shift_A_Flag",
        "Is_Month_End", "Week_Of_Month", "DOM_Sin", "DOM_Cos"
    ]

    results = []

    for st in stations:
        target = label_cols[st]
        st_features = [
            f"{st}_Local_Z_t", f"{st}_Roll6_Z_t", f"{st}_Vel6_Z"
        ]
        # Include capacity & physics features if available
        for extra in [f"Eff_UPH_{st}", f"Cap_DOS_{st}", f"Mature_Ratio_{st}", f"Expected_Cohort_{st}", f"Net_UPH_Gap_{st}"]:
            if extra in df.columns:
                st_features.append(extra)

        features = [f for f in (st_features + common_features) if f in df.columns]

        X = df[features].replace([np.inf, -np.inf], np.nan).fillna(0.0).values
        y = df[target].values

        # 70% Train, 15% Validation, 15% Test Split
        n = len(df)
        train_end = int(n * 0.70)
        val_end = int(n * 0.85)

        X_train, y_train = X[:train_end], y[:train_end]
        X_test, y_test = X[val_end:], y[val_end:]

        scaler = RobustScaler()
        X_train_s = np.clip(scaler.fit_transform(X_train), -6.0, 6.0)
        X_test_s = np.clip(scaler.transform(X_test), -6.0, 6.0)

        # 1. Balanced Random Forest Classifier
        rf = RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=4,
            class_weight="balanced_subsample", random_state=42, n_jobs=-1
        )
        rf.fit(X_train_s, y_train)
        rf_probs = rf.predict_proba(X_test_s)[:, 1]
        m_rf = compute_metrics(y_test, rf_probs)

        # 2. Gradient Boosting
        gb = HistGradientBoostingClassifier(
            max_iter=120, max_depth=4, learning_rate=0.05,
            class_weight="balanced", random_state=42
        )
        gb.fit(X_train_s, y_train)
        gb_probs = gb.predict_proba(X_test_s)[:, 1]
        m_gb = compute_metrics(y_test, gb_probs)

        results.append({
            "Station": st.replace("WIP_", ""),
            "RF_Accuracy": m_rf["Accuracy"],
            "RF_Precision": m_rf["Precision"],
            "RF_Recall": m_rf["Recall"],
            "RF_F1": m_rf["F1_Score"],
            "RF_ROC_AUC": m_rf["ROC_AUC"],
            "GB_F1": m_gb["F1_Score"],
            "GB_ROC_AUC": m_gb["ROC_AUC"]
        })

    res_df = pd.DataFrame(results)
    out_dir = os.path.join(base_dir, "output")
    os.makedirs(out_dir, exist_ok=True)
    res_df.to_csv(os.path.join(out_dir, "bottleneck_benchmark_results.csv"), index=False)

    print("\n" + "=" * 90)
    print("6-HOUR BOTTLENECK CLASSIFICATION BENCHMARK ON TEST SET (RANDOM FOREST)")
    print("=" * 90)
    print(f"{'Station':<16} | {'Accuracy':>9} | {'Precision':>10} | {'Recall':>9} | {'F1-Score':>9} | {'ROC-AUC':>9}")
    print("-" * 90)
    for _, r in res_df.iterrows():
        print(f"{r['Station']:<16} | {r['RF_Accuracy']:>8.2f}% | {r['RF_Precision']:>9.2f}% | {r['RF_Recall']:>8.2f}% | {r['RF_F1']:>8.2f}% | {r['RF_ROC_AUC']:>8.2f}%")
    print("=" * 90)
    print(f"Results exported to: {os.path.join(out_dir, 'bottleneck_benchmark_results.csv')}\n")


if __name__ == "__main__":
    run_bottleneck_classification()
