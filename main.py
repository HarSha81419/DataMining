from src.data_collector import collect_all
from src.preprocess import preprocess_data
from src.model_trainer import train_and_compare_models
from src.visualizer import visualize_results
from src.forecast_next7days import forecast_next_7_days
from src.summary_utils import show_summary
from src.visualize_actual_vs_predicted import visualize_actual_vs_predicted  # 👈 NEW import

print("📡 Collecting data...")
collect_all()

print("\n🧹 Preprocessing data...")
df = preprocess_data()

print("\n🤖 Training models...")
results, trained_models = train_and_compare_models(df)

print("\n📈 Generating Summary...")
show_summary(df, results)

print("\n🔮 Forecasting next 7 days of solar power...")
forecast_next_7_days(trained_models, df)

print("\n📊 Visualizing model performance summary...")
visualize_results()

print("\n📉 Plotting Actual vs Predicted results for each city and model...")
visualize_actual_vs_predicted()  # 👈 Added here

print("\n✅ All tasks completed successfully! Check the 'output/' folder for results and plots.")
