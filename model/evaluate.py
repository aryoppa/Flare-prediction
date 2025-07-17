# model/evaluate.py
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from utils.helpers import index_to_class_map


def run_evaluation(model, X_train, y_train_cat, X_test, y_test_cat,
                   split_index, valid_data, mode="plain"):
    # --- Prediksi ---
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test_cat, axis=1)

    # --- Confusion matrix ---
    conf_matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix ({mode})")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()

    # --- Classification report ---
    print(f"\nClassification Report ({mode.upper()}):")
    target_names = [index_to_class_map[i] for i in sorted(np.unique(y_true))]
    print(classification_report(y_true, y_pred, target_names=target_names))

    # --- Manual Accuracy Check ---
    df_test = valid_data.iloc[split_index:].copy().reset_index(drop=True)
    df_test['True Label Index'] = y_true
    df_test['Predicted Label Index'] = y_pred
    df_test['Predicted Label'] = [index_to_class_map.get(i, 'Unknown') for i in y_pred]
    df_test['Checking'] = df_test.apply(
        lambda row: "Yes" if row['True Label Index'] == row['Predicted Label Index'] else "No",
        axis=1
    )

    correct = (df_test['Checking'] == "Yes").sum()
    total = len(df_test)
    print(f"\nManual Accuracy: {correct}/{total} = {correct / total * 100:.2f}%")

    # --- Simpan hasil ke CSV ---
    out_path = f"data/datasets/hasil_prediksi_test_{mode}.csv"
    df_test.to_csv(out_path, index=False)
    print(f"Hasil prediksi disimpan ke: {out_path}")
