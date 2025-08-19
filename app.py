import os
import numpy as np
import pickle
import pandas as pd
from collections import Counter

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    f1_score,
    precision_score,
    recall_score,
    balanced_accuracy_score
)

from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from imblearn.over_sampling import RandomOverSampler

# (opsional) untuk simpan gambar confusion matrix
import matplotlib.pyplot as plt

from data.preprocessing import load_and_preprocess_data
from model.tcn_model import build_manual_tcn_model
from utils.helpers import index_to_class_map

# ========== Utility Functions ==========
def compute_class_weight(y, num_classes):
    counts = Counter(y)
    total = sum(counts.values())
    class_weight = {cls: total / (num_classes * count) for cls, count in counts.items()}
    log_scaled_weight = {cls: np.log1p(total / (count + 1)) for cls, count in counts.items()}
    return class_weight, log_scaled_weight

def build_windowed_dataset(X, y, window_size):
    X_windowed, y_windowed = [], []
    for i in range(window_size, len(X)):
        X_windowed.append(X[i - window_size:i])
        y_windowed.append(y[i])
    return np.array(X_windowed), np.array(y_windowed)

def oversample_windows(X_windowed, y_windowed, random_state=42):
    n_samples, seq_len, n_features = X_windowed.shape
    X_flat = X_windowed.reshape((n_samples, seq_len * n_features))
    ros = RandomOverSampler(random_state=random_state)
    X_resampled, y_resampled = ros.fit_resample(X_flat, y_windowed)
    X_resampled = X_resampled.reshape((-1, seq_len, n_features))
    print("Class balance after oversampling:", Counter(y_resampled))
    return X_resampled, y_resampled

def per_class_metrics(y_true, y_pred, label_map):
    result = {}
    for idx, cname in label_map.items():
        n_true = np.sum(y_true == idx)
        n_correct = np.sum((y_true == idx) & (y_pred == idx))
        acc = (n_correct / n_true) * 100 if n_true > 0 else 0.0
        result[f'{cname}_TestCount'] = n_true
        result[f'{cname}_Correct'] = n_correct
        result[f'{cname}_Accuracy'] = acc  # ini secara teknis recall per kelas
    return result

def save_confusion_matrix_png(cm, class_names, title, out_png_path):
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_png_path, dpi=150)
    plt.close(fig)

# ========== Main Experiment Pipeline ==========
def train_and_evaluate(MODEL_DIR, OUTDIR, window_sizes=[2,4,5]):
    split_ratio = 70
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(OUTDIR, exist_ok=True)
    summary_results = []
    metrics_rows = []  # ringkasan metrik agregat per model

    # Load and preprocess data
    X_train_raw, X_test_raw, y_train_cat, y_test_cat, y_train, y_test, split_index, valid_data, _ = load_and_preprocess_data(split_ratio)

    # Squeeze if needed
    X_train_raw = np.squeeze(X_train_raw)
    X_test_raw = np.squeeze(X_test_raw)
    if X_train_raw.ndim == 1: X_train_raw = X_train_raw[:, None]
    if X_test_raw.ndim == 1: X_test_raw = X_test_raw[:, None]

    # Kunci kelas & nama label
    # Pastikan index_to_class_map berbentuk {0:'A',1:'B',2:'C',3:'M',4:'X'}
    label_map = index_to_class_map
    all_class_indices = sorted(label_map.keys())
    num_classes = len(all_class_indices)
    target_names = [label_map[i] for i in all_class_indices]

    print(f"Split index: {split_index}")
    print(f"Train shape: {X_train_raw.shape}, Test shape: {X_test_raw.shape}")

    for window_size in window_sizes:
        print(f"\n\n🔄 Trying window_size = {window_size}")

        # Windowed dataset
        X_train_w, y_train_w = build_windowed_dataset(X_train_raw, y_train, window_size)
        X_test_w, y_test_w = build_windowed_dataset(X_test_raw, y_test, window_size)

        print("Class balance before oversampling:", Counter(y_train_w))
        class_weight, log_scaled_weight = compute_class_weight(y_train_w, num_classes)
        modes = [
            ("oversampled", None),
            ("oversampled_classweight", class_weight),
            ("oversampled_logscaled", log_scaled_weight),
        ]

        # Oversample (TRAIN only)
        X_train_bal, y_train_bal = oversample_windows(X_train_w, y_train_w)
        y_train_cat_bal = to_categorical(y_train_bal, num_classes=num_classes)
        y_test_cat = to_categorical(y_test_w, num_classes=num_classes)

        for mode_name, cweight in modes:
            print(f"\n== Training ({mode_name}) for window_size={window_size} ==")
            model = build_manual_tcn_model(input_shape=X_train_bal.shape[1:], num_classes=num_classes)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

            history = model.fit(
                X_train_bal, y_train_cat_bal,
                validation_data=(X_test_w, y_test_cat),
                epochs=20,
                batch_size=32,
                callbacks=[reduce_lr],
                class_weight=cweight,
                verbose=1
            )

            # --- Prediction ---
            y_test_pred = np.argmax(model.predict(X_test_w), axis=1)
            y_test_true = y_test_w

            # --- Save raw comparison ---
            comp_path = os.path.join(OUTDIR, f'comparison_test_{mode_name}_ws{window_size}.csv')
            pd.DataFrame({"True_Test": y_test_true, "Pred_Test": y_test_pred}).to_csv(comp_path, index=False)

            # --- Confusion Matrix (CSV + PNG) ---
            cm = confusion_matrix(y_test_true, y_test_pred, labels=all_class_indices)
            cm_csv_path = os.path.join(OUTDIR, f'confusion_matrix_{mode_name}_ws{window_size}.csv')
            pd.DataFrame(cm, index=target_names, columns=target_names).to_csv(cm_csv_path)
            
            # opsional PNG
            cm_png_path = os.path.join(OUTDIR, f'confusion_matrix_{mode_name}_ws{window_size}.png')
            save_confusion_matrix_png(cm, target_names,
                                      title=f"Confusion Matrix - {mode_name} (ws={window_size})",
                                      out_png_path=cm_png_path)

            # --- Classification report per kelas (CSV) ---
            # output_dict=True agar bisa disimpan sebagai DataFrame
            cls_report = classification_report(
                y_test_true, y_test_pred,
                labels=all_class_indices,
                target_names=target_names,
                zero_division=0,
                output_dict=True
            )
            cls_report_df = pd.DataFrame(cls_report).transpose()
            rep_csv_path = os.path.join(OUTDIR, f'classification_report_{mode_name}_ws{window_size}.csv')
            cls_report_df.to_csv(rep_csv_path)

            # --- Aggregate metrics ---
            test_acc = np.mean(y_test_pred == y_test_true)
            macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
                y_test_true, y_test_pred, labels=all_class_indices, zero_division=0, average='macro'
            )
            micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(
                y_test_true, y_test_pred, labels=all_class_indices, zero_division=0, average='micro'
            )
            weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
                y_test_true, y_test_pred, labels=all_class_indices, zero_division=0, average='weighted'
            )
            bal_acc = balanced_accuracy_score(y_test_true, y_test_pred)

            metrics_rows.append({
                "Model": f"{mode_name}_ws{window_size}",
                "Window Size": window_size,
                "Test Samples": X_test_w.shape[0],
                "Test Accuracy": test_acc,
                "F1 Macro": macro_f1,
                "Precision Macro": macro_p,
                "Recall Macro": macro_r,
                "F1 Micro": micro_f1,
                "Precision Micro": micro_p,
                "Recall Micro": micro_r,
                "F1 Weighted": weighted_f1,
                "Precision Weighted": weighted_p,
                "Recall Weighted": weighted_r,
                "Balanced Accuracy": bal_acc
            })

            # --- Per-class analysis (recall per kelas, dsb) untuk ringkasan utama ---
            per_class_summary = per_class_metrics(y_test_true, y_test_pred, label_map)
            row = {
                'Model': f"{mode_name}_ws{window_size}",
                'Window Size': window_size,
                'Train Samples': X_train_bal.shape[0],
                'Test Samples': X_test_w.shape[0],
                'Train Accuracy': history.history['accuracy'][-1],
                'Val Accuracy (Last Epoch)': history.history['val_accuracy'][-1],
                'Val Loss (Last Epoch)': history.history['val_loss'][-1],
                'Test Accuracy': test_acc,
                'F1 Macro': macro_f1,
                'Balanced Accuracy': bal_acc
            }
            row.update(per_class_summary)
            summary_results.append(row)

            # --- Save model & history ---
            model_path = os.path.join(MODEL_DIR, f'solarflare_tcn_{mode_name}_ws{window_size}.h5')
            model.save(model_path)
            with open(os.path.join(MODEL_DIR, f'train_history_manual_{mode_name}_ws{window_size}.pkl'), 'wb') as f:
                pickle.dump(history.history, f)

    # Save summaries
    summary_df = pd.DataFrame(summary_results)
    csv_path = os.path.join(OUTDIR, "model_eval_per_class_summary.csv")
    summary_df.to_csv(csv_path, index=False)

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_path = os.path.join(OUTDIR, "model_eval_metrics_summary.csv")
    metrics_df.to_csv(metrics_path, index=False)

    print(f"\n✅ Saved: {csv_path}")
    print(f"✅ Saved: {metrics_path}")

if __name__ == '__main__':
    MODEL_DIR = "final/trained_model"
    OUTDIR = "final/results"
    train_and_evaluate(MODEL_DIR, OUTDIR, window_sizes=[2, 4, 5])
    print("Training and evaluation completed.")
