import os
import numpy as np
import pickle
import pandas as pd
from collections import Counter
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
from model.tcn_manual import build_manual_tcn_model
from data.preprocessing import load_and_preprocess_data
import tensorflow.keras.backend as K

# --- Parameters ---
split_ratio = 70
MODEL_DIR = 'trained_model_v3'
os.makedirs(MODEL_DIR, exist_ok=True)
OUTDIR = "results"
os.makedirs(OUTDIR, exist_ok=True)

# --- Precursor relabeling function ---
def relabel_precursors(y, classes_of_interest=(0, 4), steps_before=1):
    y_new = y.copy()
    for i in range(steps_before, len(y)):
        if y[i] in classes_of_interest:
            y_new[i - steps_before] = y[i]
    return y_new

# --- Load Data ---
X_train_raw, X_test_raw, y_train_cat, y_test_cat, y_train, y_test, split_index, valid_data, class_weight_dict = load_and_preprocess_data(split_ratio)

# --- Apply precursor relabeling ---
y_train_precursor = relabel_precursors(y_train, classes_of_interest=(0, 4), steps_before=1)
y_test_precursor = relabel_precursors(y_test, classes_of_interest=(0, 4), steps_before=1)

# --- Prepare data: remove dummy dimension ---
X_train_raw = np.squeeze(X_train_raw)
X_test_raw = np.squeeze(X_test_raw)

# --- Class Weights ---
label_counts = Counter(y_train_precursor)
total_samples = sum(label_counts.values())
num_classes = len(label_counts)

adjusted_class_weight = {cls: total_samples / (num_classes * count) for cls, count in label_counts.items()}
log_scaled_class_weight = {cls: np.log1p(total_samples / (count + 1)) for cls, count in label_counts.items()}

# --- Focal Loss ---
def focal_loss(gamma=2.0, alpha=0.25):
    def loss(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * K.log(y_pred)
        weight = alpha * K.pow(1 - y_pred, gamma)
        return K.sum(weight * cross_entropy, axis=1)
    return loss

# --- Build windowed dataset from previous events ---
def build_windowed_dataset(X, y, window_size):
    X_windowed = []
    y_windowed = []
    for i in range(window_size, len(X)):
        X_windowed.append(X[i - window_size:i])
        y_windowed.append(y[i])
    return np.array(X_windowed), np.array(y_windowed)

# --- Train & Evaluate ---
def train_and_eval_tcn(X_train, y_train, X_test, y_test, num_classes, class_weight=None, mode_name="plain", use_focal=False):
    print(f"\n======= Training ({mode_name}) =======")
    model = build_manual_tcn_model(input_shape=X_train.shape[1:], num_classes=num_classes)
    loss_fn = focal_loss() if use_focal else 'categorical_crossentropy'
    model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=20,
        batch_size=32,
        class_weight=class_weight,
        callbacks=[reduce_lr],
        verbose=1
    )

    # === Prediction: Test ===
    y_test_true = np.argmax(y_test, axis=1)
    y_pred = np.argmax(model.predict(X_test), axis=1)

    # === Prediction: Train ===
    y_train_true = np.argmax(y_train, axis=1)
    y_pred_train = np.argmax(model.predict(X_train), axis=1)

    # === Accuracy Comparison ===
    train_accuracy = np.mean(y_train_true == y_pred_train)
    test_accuracy = np.mean(y_test_true == y_pred)

    label_map = {0:'A', 1:'B', 2:'C', 3:'M', 4:'X'}
    target_names = [label_map[i] for i in sorted(label_map.keys())]

    print(f"\n--- Classification Report ({mode_name}) ---")
    print(classification_report(y_test_true, y_pred, target_names=target_names))
    print(f"\n--- Confusion Matrix ({mode_name}) ---")
    print(confusion_matrix(y_test_true, y_pred))

    # === Save Model & History ===
    history_dict = history.history
    history_dict['y_test'] = y_test_true
    history_dict['y_pred'] = y_pred
    history_dict['y_train_true'] = y_train_true
    history_dict['y_train_pred'] = y_pred_train
    history_dict['train_accuracy'] = train_accuracy
    history_dict['test_accuracy'] = test_accuracy

    model_path = os.path.join(MODEL_DIR, f'solarflare_tcn_manual_{mode_name}.h5')
    history_path = os.path.join(MODEL_DIR, f'train_history_manual_{mode_name}.pkl')
    model.save(model_path)
    with open(history_path, 'wb') as f:
        pickle.dump(history_dict, f)

    print(f"Model saved as {model_path}")
    print(f"History saved as {history_path}")

    # === Save Comparison CSV ===
    df_train = pd.DataFrame({'True_Train': y_train_true, 'Pred_Train': y_pred_train})
    df_test = pd.DataFrame({'True_Test': y_test_true, 'Pred_Test': y_pred})
    df_train.to_csv(os.path.join(OUTDIR, f'comparison_train_{mode_name}.csv'), index=False)
    df_test.to_csv(os.path.join(OUTDIR, f'comparison_test_{mode_name}.csv'), index=False)

    return model, history, y_pred, y_test_true, y_pred_train, y_train_true, train_accuracy, test_accuracy

# --- Main Evaluation and Summary ---
window_sizes = [2, 4, 5]
flare_classes = ['A', 'B', 'C', 'M', 'X']
summary_results = []

for w in window_sizes:
    try:
        print(f"\n\n🔄 Trying window_size = {w}")

        # Build windowed dataset for both train and test, using precursor-labeled y
        X_train_windowed, y_train_windowed = build_windowed_dataset(X_train_raw, y_train_precursor, window_size=w)
        X_test_windowed, y_test_windowed = build_windowed_dataset(X_test_raw, y_test_precursor, window_size=w)

        # One-hot encode new windowed y
        y_train_cat_windowed = np.eye(num_classes)[y_train_windowed]
        y_test_cat_windowed = np.eye(num_classes)[y_test_windowed]

        for mode_name, class_weight, use_focal in [
            (f"plain_ws{w}", None, False),
            (f"weighted_ws{w}", class_weight_dict, False),
            (f"adjusted_weight_ws{w}", adjusted_class_weight, False),
            (f"log_scaled_weight_ws{w}", log_scaled_class_weight, False),
            (f"focal_ws{w}", None, True)
        ]:
            model, history, y_pred, y_test_true, y_pred_train, y_train_true, train_acc, test_acc = train_and_eval_tcn(
                X_train_windowed, y_train_cat_windowed, X_test_windowed, y_test_cat_windowed,
                num_classes, class_weight=class_weight, mode_name=mode_name, use_focal=use_focal
            )

            per_class_summary = {}
            for idx, cname in enumerate(flare_classes):
                n_true = np.sum(y_test_true == idx)
                n_correct = np.sum((y_test_true == idx) & (y_pred == idx))
                acc = (n_correct / n_true) * 100 if n_true > 0 else 0.0
                per_class_summary[f'{cname}_TestCount'] = n_true
                per_class_summary[f'{cname}_Correct'] = n_correct
                per_class_summary[f'{cname}_Accuracy'] = acc

            row = {
                'Model': mode_name,
                'Window Size': w,
                'Train Accuracy': train_acc,
                'Test Accuracy': test_acc,
                'Val Accuracy (Last Epoch)': history.history['val_accuracy'][-1],
                'Val Loss (Last Epoch)': history.history['val_loss'][-1],
            }
            row.update(per_class_summary)
            summary_results.append(row)

    except Exception as e:
        print(f"⚠️ Skipping window_size={w}: {e}")

# --- Save results to CSV ---
summary_df = pd.DataFrame(summary_results)
csv_path = os.path.join(OUTDIR, "model_eval_per_class_v3.csv")
summary_df.to_csv(csv_path, index=False)
print(f"\n✅ Model per-class evaluation summary saved to {csv_path}")
