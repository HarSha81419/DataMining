import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

OUTPUT_DIR = "output"
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


def forecast_next_7_days(models, df):
    print("🛰️ Generating 7-day solar power forecast...")

    # 🔹 Automatically pick the best model (prefer Random Forest or XGBoost)
    best_model_name = max(
        models.keys(), key=lambda name: "Random Forest" in name or "XGBoost" in name
    )
    best_model = models[best_model_name]

    print(f"🔮 Using {best_model_name} for next 7 days DC & AC power forecast...")

    cities = ["Hyderabad", "Mumbai", "Delhi", "Chennai", "Bengaluru", "Kolkata", "Ahmedabad", "Pune", "Jaipur"]
    date_rng = pd.date_range(start=datetime.now(), periods=7, freq="D")
    all_forecasts = []

    for city in cities:
        # Simulate future NASA-style weather data (could be replaced with live API)
        df_future = pd.DataFrame({
            "DATE": date_rng,
            "CITY": city,
            "Solar_Irradiance(kWh/m2)": np.random.uniform(4, 7, len(date_rng)),
            "Temperature(C)": np.random.uniform(25, 38, len(date_rng)),
            "Humidity(%)": np.random.uniform(40, 75, len(date_rng)),
            "Wind_Speed(m/s)": np.random.uniform(1, 5, len(date_rng)),
        })

        X_future = df_future[["Solar_Irradiance(kWh/m2)", "Temperature(C)", "Humidity(%)", "Wind_Speed(m/s)"]]

        # Predict using best model
        df_future["Predicted_DC_Power(kW)"] = best_model.predict(X_future)
        df_future["Predicted_AC_Power(kW)"] = df_future["Predicted_DC_Power(kW)"] * np.random.uniform(0.85, 0.95)

        all_forecasts.append(df_future)

    forecast_df = pd.concat(all_forecasts, ignore_index=True)
    forecast_path = os.path.join(OUTPUT_DIR, "predicted_next_7_days.csv")
    forecast_df.to_csv(forecast_path, index=False)

    print(f"✅ Saved 7-day forecast to {forecast_path}")

    # 📊 Plot for each city
    for city in cities:
        city_data = forecast_df[forecast_df["CITY"] == city]
        plt.figure(figsize=(8, 5))
        plt.plot(city_data["DATE"], city_data["Predicted_DC_Power(kW)"], label="Predicted DC Power", marker='o')
        plt.plot(city_data["DATE"], city_data["Predicted_AC_Power(kW)"], label="Predicted AC Power", marker='x')
        plt.title(f"{city} - Next 7 Days Solar Power Forecast")
        plt.xlabel("Date")
        plt.ylabel("Power (kW)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{PLOTS_DIR}/{city}_Next7DaysForecast.png")
        plt.close()

    print(f"📊 Saved forecast plots to {PLOTS_DIR}/")

    return forecast_df


if __name__ == "__main__":
    # For standalone testing
    df = pd.read_csv("data/combined_data.csv")
    from joblib import load
    models = {"Random Forest": load("output/random_forest_model.joblib")}
    forecast_next_7_days(models, df)
