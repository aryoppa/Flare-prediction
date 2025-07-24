import pandas as pd

# Load your comparison summary file
summary_path = "results/comparison_summary_all_models.csv"  # Update with full path if needed
df_summary = pd.read_csv(summary_path)

# Top 5 by test accuracy
top5 = df_summary.sort_values(by="test_overall_acc", ascending=False).head(5)

# Per-class accuracy for test
test_cols = [col for col in df_summary.columns if col.startswith("test_class_")]
per_class_test = top5[["Model"] + test_cols]

class_names = ['A', 'B', 'C', 'M', 'X']
tidy_per_class = per_class_test.copy()
for i, cname in enumerate(class_names):
    old_col = f"test_class_{i}_acc"
    if old_col in tidy_per_class.columns:
        tidy_per_class[cname] = tidy_per_class[old_col]
tidy_per_class = tidy_per_class[["Model"] + class_names]

print("=== Top 5 Models (by Test Accuracy) ===")
print(top5[["Model", "test_overall_acc", "train_overall_acc"]])

print("\n=== Per-Class Test Accuracy for Top 5 Models ===")
print(tidy_per_class)

# Suggestions
low_acc_classes = tidy_per_class[class_names].mean().sort_values()
improvement_suggestions = []
if any(tidy_per_class[class_names].mean() < 5):
    improvement_suggestions.append(
        "Some classes still have very low test accuracy (<5%). Consider:\n"
        "- Increasing precursor window (steps_before=2 or 3)\n"
        "- Using even higher class weighting/focal loss alpha for A/X\n"
        "- Blending precursor labeling with aggressive oversampling\n"
        "- Using ensemble predictions"
    )
else:
    improvement_suggestions.append(
        "Per-class accuracy is more balanced. You may tune the precursor window, or optimize the model hyperparameters for further improvement."
    )

print("\n=== Suggestions for Further Improvement ===")
for suggestion in improvement_suggestions:
    print("-", suggestion)
