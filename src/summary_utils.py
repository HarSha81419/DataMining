import pandas as pd

def show_summary(df, metrics):
    print("\n=== ⚡ SOLAR PERFORMANCE SUMMARY ⚡ ===")
    print(f"Records: {len(df)}")
    print(f"Cities: {df['CITY'].nunique() if 'CITY' in df.columns else 'N/A'}")
    print(f"Avg Irradiance: {df['Solar_Irradiance(kWh/m2)'].mean():.2f}")
    print(f"Avg DC Power: {df['Actual_DC_Power(kW)'].mean():.2f}")
    print("==============================\n")

def print_comparison_table(y_true, y_pred, label="Power"):
    comp = pd.DataFrame({
        "Actual": y_true.values,
        "Predicted": y_pred,
    })
    comp["Error"] = comp["Predicted"] - comp["Actual"]
    comp["Error_%"] = abs(comp["Error"] / comp["Actual"]) * 100
    print(f"\n--- {label} Comparison Table (sample) ---")
    print(comp.head(10).to_string(index=False))
    return comp
