import os
import pandas as pd
import numpy as np

DATA_DIR = "data"
OUTPUT_DIR = "data/cleaned"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def preprocess_data():
    raw_path = os.path.join(DATA_DIR, "combined_data.csv")
    cleaned_path = os.path.join(OUTPUT_DIR, "cleaned_data.csv")

    if not os.path.exists(raw_path):
        print("⚠️ No raw data found. Run data_collector.py first.")
        return

    print("🔍 Reading raw NASA data...")
    df = pd.read_csv(raw_path)

    # --- Column renaming ---
    rename_map = {
        "ALLSKY_SFC_SW_DWN": "Solar_Irradiance(kWh/m2)",
        "T2M": "Temperature(C)",
        "RH2M": "Humidity(%)",
        "WS2M": "Wind_Speed(m/s)"
    }
    df = df.rename(columns=rename_map)

    # --- Date parsing ---
    if "DATE" in df.columns:
        df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    else:
        print("⚠️ DATE column missing, adding today’s date.")
        df["DATE"] = pd.Timestamp.today()

    # --- Remove invalid rows ---
    df = df.dropna(subset=["Solar_Irradiance(kWh/m2)", "Temperature(C)"])
    df = df[df["Solar_Irradiance(kWh/m2)"] > 0]

    # --- Compute DC & AC Power ---
    efficiency = 0.20      # PV efficiency (20%)
    temp_coeff = -0.005    # -0.5% per °C above 25°C
    inverter_eff = 0.9     # inverter efficiency

    df["Actual_DC_Power(kW)"] = (
        df["Solar_Irradiance(kWh/m2)"] * efficiency *
        (1 + temp_coeff * (df["Temperature(C)"] - 25))
    ).clip(lower=0)

    df["Actual_AC_Power(kW)"] = df["Actual_DC_Power(kW)"] * inverter_eff

    # --- Round values for readability ---
    df = df.round(3)

    # --- Merge with existing cleaned dataset ---
    if os.path.exists(cleaned_path):
        print("📈 Merging with existing cleaned data...")
        existing = pd.read_csv(cleaned_path)
        existing["DATE"] = pd.to_datetime(existing["DATE"], errors="coerce")
        combined = pd.concat([existing, df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["DATE", "CITY"], keep="last")
    else:
        combined = df

    # --- Save final cleaned dataset ---
    combined.to_csv(cleaned_path, index=False)

    print(f"✅ Cleaned & merged dataset saved to {cleaned_path}")
    print(f"Records: {len(combined)} | Cities: {combined['CITY'].nunique()}")
    print(f"Avg Irradiance: {combined['Solar_Irradiance(kWh/m2)'].mean():.2f}")
    return combined

if __name__ == "__main__":
    preprocess_data()
