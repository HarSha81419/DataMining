import os
import pandas as pd
import matplotlib.pyplot as plt

OUTPUT_DIR = "output"
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

def visualize_results():
    summary_path = os.path.join(OUTPUT_DIR, "model_performance_summary.csv")
    if not os.path.exists(summary_path):
        print("⚠️ No model summary found. Run model_trainer.py first.")
        return

    df = pd.read_csv(summary_path)
    plt.figure(figsize=(8, 5))
    plt.bar(df["Model"], df["Avg_R2"], color="skyblue")
    plt.ylabel("Average R² Score")
    plt.title("Model Comparison for Solar Power Prediction")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/Model_R2_Comparison.png")
    plt.close()
    print(f"📊 Saved model comparison plot to {PLOTS_DIR}/Model_R2_Comparison.png")

if __name__ == "__main__":
    visualize_results()
