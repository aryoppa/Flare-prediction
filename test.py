# import pandas as pd
# import matplotlib.pyplot as plt

# csv_path = "/Users/aryobagaspamungkas/Flare-prediction/data/datasets/model_eval_per_class.csv"  # Update with your file path
# df = pd.read_csv(csv_path)
# flare_classes = ['A', 'B', 'C', 'M', 'X']

# # Find best model per class by accuracy
# best_per_class = {}
# for c in flare_classes:
#     idx = df[f"{c}_Accuracy"].idxmax()
#     best_per_class[c] = {
#         "Model": df.loc[idx, "Model"],
#         "Accuracy": df.loc[idx, f"{c}_Accuracy"],
#         "TestCount": df.loc[idx, f"{c}_TestCount"],
#         "Correct": df.loc[idx, f"{c}_Correct"],
#     }

# best_models_df = pd.DataFrame.from_dict(best_per_class, orient="index")
# best_models_df.reset_index(inplace=True)
# best_models_df.rename(columns={'index':'Class'}, inplace=True)
# # display(best_models_df)
# print(best_models_df)
#   # Or print(best_models_df) if not in Jupyter

# # Plot bar chart of per-class best accuracy
# plt.figure(figsize=(8,5))
# plt.bar(best_models_df["Class"], best_models_df["Accuracy"], color="orange")
# plt.title("Best Accuracy by Flare Class (Across All Models)")
# plt.xlabel("Flare Class")
# plt.ylabel("Best Per-Class Accuracy (%)")
# plt.ylim(0, 110)
# plt.show()

# # Average per-class accuracy for each model
# df["Mean_PerClass_Accuracy"] = df[[f"{c}_Accuracy" for c in flare_classes]].mean(axis=1)
# sorted_df = df.sort_values("Mean_PerClass_Accuracy", ascending=False)
# plt.figure(figsize=(10,6))
# plt.barh(sorted_df["Model"], sorted_df["Mean_PerClass_Accuracy"], color='skyblue')
# plt.xlabel("Mean Per-Class Accuracy (%)")
# plt.title("Model Ranking by Average Per-Class Accuracy")
# plt.tight_layout()
# plt.show()

import pandas as pd

csv_path = "/Users/aryobagaspamungkas/Flare-prediction/data/datasets/model_eval_per_class.csv"
df = pd.read_csv(csv_path)
flare_classes = ['A', 'B', 'C', 'M', 'X']

# Melt the DataFrame for each class and model
all_results = []
for idx, row in df.iterrows():
    for c in flare_classes:
        all_results.append({
            "Model": row["Model"],
            "Class": c,
            "Accuracy": row[f"{c}_Accuracy"],
            "TestCount": row[f"{c}_TestCount"],
            "Correct": row[f"{c}_Correct"],
        })

full_results_df = pd.DataFrame(all_results)
print(full_results_df)

# Optional: Save to CSV for analysis in Excel/Sheets
full_results_df.to_csv("model_eval_per_class_long.csv", index=False)
