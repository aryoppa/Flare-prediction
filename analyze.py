import os
import pandas as pd
from glob import glob
from collections import defaultdict
from sklearn.metrics import accuracy_score

# --- Config ---
comparison_dir = "v3/results_v3"  # Folder where comparison_*.csv files are saved
output_path = os.path.join(comparison_dir, "comparison_summary_all_models.csv")

# --- Scan All Comparison Files ---
test_files = sorted(glob(os.path.join(comparison_dir, "comparison_test_*.csv")))
train_files = sorted(glob(os.path.join(comparison_dir, "comparison_train_*.csv")))

# --- Utility: Analyze single comparison file ---
def analyze_comparison(file_path, label_col, pred_col, dataset_type):
    df = pd.read_csv(file_path)
    df[label_col] = df[label_col].astype(int)
    df[pred_col] = df[pred_col].astype(int)

    accuracy = accuracy_score(df[label_col], df[pred_col])
    model_name = os.path.basename(file_path).replace(f"comparison_{dataset_type}_", "").replace(".csv", "")

    per_class_result = defaultdict(float)
    for class_id in sorted(df[label_col].unique()):
        mask = df[label_col] == class_id
        correct = (df[label_col] == df[pred_col]) & mask
        acc = correct.sum() / mask.sum() if mask.sum() > 0 else 0.0
        per_class_result[f"{dataset_type}_class_{class_id}_acc"] = round(acc * 100, 2)

    return {
        "Model": model_name,
        f"{dataset_type}_overall_acc": round(accuracy * 100, 2),
        **per_class_result
    }

# --- Analyze All Models ---
results = []

for test_file, train_file in zip(test_files, train_files):
    model_test = analyze_comparison(test_file, "True_Test", "Pred_Test", "test")
    model_train = analyze_comparison(train_file, "True_Train", "Pred_Train", "train")

    if model_test["Model"] != model_train["Model"]:
        print(f"⚠️ Model mismatch: {model_test['Model']} vs {model_train['Model']}")
        continue

    combined = {**model_test, **model_train}
    results.append(combined)

# --- Create DataFrame and Export ---
df_summary = pd.DataFrame(results)
df_summary = df_summary.sort_values(by="test_overall_acc", ascending=False)
df_summary.to_csv(output_path, index=False)

# --- Save and print summary ---
df_summary = pd.DataFrame(results)
df_summary = df_summary.sort_values(by="test_overall_acc", ascending=False)

# Save to CSV
output_path = os.path.join(comparison_dir, "comparison_summary_all_models.csv")
df_summary.to_csv(output_path, index=False)

# Print top 5 results
print("\n✅ Comparison summary saved to:", output_path)
print("\n🔝 Top 5 models based on test accuracy:\n")
print(df_summary.head(5).to_string(index=False))
