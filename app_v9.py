import os
import numpy as np
import pickle
import pandas as pd
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from imblearn.over_sampling import RandomOverSampler
from tensorflow.keras import layers, regularizers, models
import tensorflow.keras.backend as K

# --- Parameters ---
window_size = 5      # Choose as needed: 2, 4, 5
test_split = 0.3     # 70% train, 30% test
MODEL_DIR = 'trained_model_v2'
OUTDIR = "data/datasets"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTDIR, exist_ok=True)

# --- Load Data ---
X = np.load("data/datasets/sdo_image_features_2015-2025.npy")
y = np.load("data/datasets/sdo_labels_2015-2025.npy")

num_classes = len(np.unique(y))

# --- Train-Test Split (time-based) ---
split_idx = int(len(X) * (1 - test_split))
X_train_raw, X_test_raw = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# --- Windowed Dataset ---
def build_windowed_dataset(X, y, window_size):
    X_windowed, y_windowed = [], []
    for i in range(window_size, len(X)):
        X_windowed.append(X[i - window_size:i])
        y_windowed.append(y[i])
    return np.array(X_windowed), np.array(y_windowed)

X_train_w, y_train_w = build_windowed_dataset(X_train_raw, y_train, window_size)
X_test_w, y_test_w = build_windowed_dataset(X_test_raw, y_test, window_size)

# --- Oversample Train Data Only ---
def oversample_windows(X_windowed, y_windowed, random_state=42):
    n_samples, seq_len, n_features = X_windowed.shape
    X_flat = X_windowed.reshape((n_samples, seq_len * n_features))
    ros = RandomOverSampler(random_state=random_state)
    X_resampled, y_resampled = ros.fit_resample(X_flat, y_windowed)
    X_resampled = X_resampled.reshape((-1, seq_len, n_features))
    print("Class balance after oversampling:", Counter(y_resampled))
    return X_resampled, y_resampled

print("Class balance before oversampling:", Counter(y_train_w))
X_train_bal, y_train_bal = oversample_windows(X_train_w, y_train_w)
print("Balanced train shape:", X_train_bal.shape, y_train_bal.shape)

# --- One-hot encode labels ---
y_train_cat_bal = to_categorical(y_train_bal, num_classes=num_classes)
y_test_cat = to_categorical(y_test_w, num_classes=num_classes)

# --- Build TCN Model with Dropout and L2 ---
def build_manual_tcn_model(input_shape, num_classes, dropout_rate=0.3, l2_val=1e-4):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv1D(64, kernel_size=3, dilation_rate=1, padding="causal",
                      kernel_regularizer=regularizers.l2(l2_val))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Conv1D(64, kernel_size=3, dilation_rate=2, padding="causal",
                      kernel_regularizer=regularizers.l2(l2_val))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(l2_val))(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(inputs, outputs)
    return model

# --- Callbacks ---
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

# --- Compile and Train Model ---
model = build_manual_tcn_model(input_shape=X_train_bal.shape[1:], num_classes=num_classes)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(
    X_train_bal, y_train_cat_bal,
    validation_data=(X_test_w, y_test_cat),
    epochs=20,
    batch_size=32,
    callbacks=[reduce_lr],
    verbose=1
)

# --- Evaluation ---
y_test_pred = np.argmax(model.predict(X_test_w), axis=1)
y_test_true = y_test_w
label_map = {0: 'A', 1: 'B', 2: 'C', 3: 'M', 4: 'X'}
target_names = [label_map[i] for i in range(num_classes)]

print("\n--- Test Classification Report ---")
print(classification_report(y_test_true, y_test_pred, target_names=target_names))
print("\n--- Test Confusion Matrix ---")
print(confusion_matrix(y_test_true, y_test_pred))

# --- Save Model & History ---
model_path = os.path.join(MODEL_DIR, f'solarflare_tcn_oversampled_dropout_l2_ws{window_size}.h5')
model.save(model_path)
with open(os.path.join(MODEL_DIR, f'train_history_manual_oversampled_dropout_l2_ws{window_size}.pkl'), 'wb') as f:
    pickle.dump(history.history, f)

# --- Save predictions for further analysis ---
pd.DataFrame({
    "True_Test": y_test_true,
    "Pred_Test": y_test_pred
}).to_csv(os.path.join(OUTDIR, f'comparison_test_oversampled_dropout_l2_ws{window_size}.csv'), index=False)

print(f"\n✅ Model & predictions saved. Model: {model_path}")
