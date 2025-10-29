import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from src.summary_utils import print_comparison_table

OUTPUT_DIR = "output"
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

def train_and_compare_models(df):
    print("\n🚀 Training and comparing multiple models for DC & AC power prediction...\n")

    features = ["Solar_Irradiance(kWh/m2)", "Temperature(C)", "Humidity(%)", "Wind_Speed(m/s)"]
    target_dc, target_ac = "Actual_DC_Power(kW)", "Actual_AC_Power(kW)"

    X = df[features]
    y_dc, y_ac = df[target_dc], df[target_ac]

    X_train, X_test, y_train_dc, y_test_dc, y_train_ac, y_test_ac = train_test_split(
        X, y_dc, y_ac, test_size=0.2, random_state=42
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42)
    }

    results = []

    for name, model in models.items():
        print(f"\n🔹 Training model: {name}")

        model.fit(X_train, y_train_dc)
        pred_dc = model.predict(X_test)

        model.fit(X_train, y_train_ac)
        pred_ac = model.predict(X_test)

        metrics = {
            "Model": name,
            "MAE_DC": mean_absolute_error(y_test_dc, pred_dc),
            "RMSE_DC": np.sqrt(mean_squared_error(y_test_dc, pred_dc)),
            "R2_DC": r2_score(y_test_dc, pred_dc),
            "MAE_AC": mean_absolute_error(y_test_ac, pred_ac),
            "RMSE_AC": np.sqrt(mean_squared_error(y_test_ac, pred_ac)),
            "R2_AC": r2_score(y_test_ac, pred_ac),
        }
        metrics["Avg_R2"] = (metrics["R2_DC"] + metrics["R2_AC"]) / 2
        results.append(metrics)

        # Save comparison tables
        dc_comp = print_comparison_table(y_test_dc, pred_dc, label=f"{name}_DC")
        ac_comp = print_comparison_table(y_test_ac, pred_ac, label=f"{name}_AC")
        dc_comp.to_csv(f"{OUTPUT_DIR}/{name}_DC_comparison.csv", index=False)
        ac_comp.to_csv(f"{OUTPUT_DIR}/{name}_AC_comparison.csv", index=False)

    results_df = pd.DataFrame(results).sort_values("Avg_R2", ascending=False)
    results_df.to_csv(f"{OUTPUT_DIR}/model_performance_summary.csv", index=False)
    print(f"\n✅ Results saved to {OUTPUT_DIR}/model_performance_summary.csv")

    return results_df, models

if __name__ == "__main__":
    df = pd.read_csv("data/combined_data.csv")
    train_and_compare_models(df)
