import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# --- Config ---
input_path = "v3/results_v3/comparison_summary_all_models.csv"  # update if needed

# --- Load data ---
df = pd.read_csv(input_path)

# --- Melt class columns into long format ---
melted_df = df.melt(id_vars=["Model"], 
                    value_vars=[col for col in df.columns if "class" in col],
                    var_name="Metric", value_name="Accuracy (%)")

# Extract Dataset (train/test) and Class index
melted_df[["Dataset", "ClassIndex", "Drop"]] = melted_df["Metric"].str.extract(r"(train|test)_class_(\d+)(_acc)?")
melted_df.drop(columns=["Metric", "Drop"], inplace=True)
melted_df["Class"] = melted_df["ClassIndex"].map({
    "0": "A", "1": "B", "2": "C", "3": "M", "4": "X"
})

# Pivot to get class accuracy table
pivot_df = melted_df.pivot_table(index=["Model", "Dataset"], columns="Class", values="Accuracy (%)")

# --- Heatmap for Test Set ---
test_only = pivot_df.loc[pivot_df.index.get_level_values("Dataset") == "test"]

plt.figure(figsize=(10, 6))
sns.heatmap(test_only, annot=True, fmt=".1f", cmap="YlGnBu", cbar_kws={'label': 'Accuracy (%)'})
plt.title("Per-Class Accuracy (Test Set)")
plt.ylabel("Model")
plt.xlabel("Flare Class")
plt.tight_layout()
plt.savefig("per_class_accuracy_heatmap_testset.png")
plt.show()

# --- Most Balanced Models (Lowest Std Dev across class accuracy in test) ---
test_std = test_only.std(axis=1).sort_values()
balanced_models = test_std.head(5).reset_index()
balanced_models.columns = ["Model", "Dataset", "Std Deviation (%)"]
balanced_models.to_csv("balanced_models_by_std.csv", index=False)

# --- Models with <10% Generalization Gap per class ---
generalization_gap = []
models = df["Model"].unique()

for model in models:
    try:
        train_row = pivot_df.loc[(model, "train")]
        test_row = pivot_df.loc[(model, "test")]
        gap = abs(train_row - test_row)
        if all(gap < 10):
            avg_test_acc = test_row.mean()
            generalization_gap.append({"Model": model, "Avg Test Accuracy": round(avg_test_acc, 2)})
    except:
        continue

gap_df = pd.DataFrame(generalization_gap)
gap_df = gap_df.sort_values(by="Avg Test Accuracy", ascending=False)
gap_df.to_csv("models_with_small_generalization_gap.csv", index=False)

# --- Summary Output ---
print("\n✅ Analysis complete.")
print("- Heatmap saved to: per_class_accuracy_heatmap_testset.png")
print("- Balanced models saved to: balanced_models_by_std.csv")
print("- Generalization gap summary saved to: models_with_small_generalization_gap.csv")

if gap_df.empty:
    print("\n⚠️ No models found with <10% generalization gap per class.")
