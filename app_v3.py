import os
import numpy as np
import pickle
from collections import Counter
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
from model.tcn_manual import build_manual_tcn_model
from data.preprocessing import load_and_preprocess_data

# --- Parameters ---
split_ratio = 70
MODEL_DIR = 'trained_model'
os.makedirs(MODEL_DIR, exist_ok=True)

# --- Load Data ---
X_train_tcn, X_test_tcn, y_train_cat, y_test_cat, y_train, y_test, split_index, valid_data, class_weight_dict = load_and_preprocess_data(split_ratio)

# --- Adjusted class weights ---
label_counts = Counter(y_train)
total_samples = sum(label_counts.values())
num_classes = len(label_counts)
adjusted_class_weight = {
    cls: total_samples / (num_classes * count)
    for cls, count in label_counts.items()
}

# --- Train & Evaluate Function ---
def train_and_eval_tcn(X_train, y_train, X_test, y_test, num_classes, class_weight=None, mode_name="plain"):
    print(f"\n======= Training ({mode_name}) =======")
    
    model = build_manual_tcn_model(input_shape=X_train.shape[1:], num_classes=num_classes)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
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
    
    y_test_true = np.argmax(y_test, axis=1)
    y_pred = np.argmax(model.predict(X_test), axis=1)
    
    label_map = {0:'A', 1:'B', 2:'C', 3:'M', 4:'X'}
    target_names = [label_map[i] for i in sorted(label_map.keys())]
    
    print(f"\n--- Classification Report ({mode_name}) ---")
    print(classification_report(y_test_true, y_pred, target_names=target_names))
    print(f"\n--- Confusion Matrix ({mode_name}) ---")
    print(confusion_matrix(y_test_true, y_pred))

    model_path = os.path.join(MODEL_DIR, f'solarflare_tcn_manual_{mode_name}.h5')
    history_path = os.path.join(MODEL_DIR, f'train_history_manual_{mode_name}.pkl')
    model.save(model_path)
    
    with open(history_path, 'wb') as f:
        pickle.dump(history.history, f)
    
    print(f"Model saved as {model_path}")
    print(f"History saved as {history_path}")
    
    return model, history, y_pred

# --- Training: Plain ---
num_classes = y_train_cat.shape[1]
model_plain, history_plain, y_pred_plain = train_and_eval_tcn(
    X_train_tcn, y_train_cat, X_test_tcn, y_test_cat,
    num_classes, class_weight=None, mode_name="plain"
)

# --- Training: Weighted ---
model_weighted, history_weighted, y_pred_weighted = train_and_eval_tcn(
    X_train_tcn, y_train_cat, X_test_tcn, y_test_cat,
    num_classes, class_weight=class_weight_dict, mode_name="weighted"
)

# --- Training: Adjusted Weight ---
model_adjusted, history_adjusted, y_pred_adjusted = train_and_eval_tcn(
    X_train_tcn, y_train_cat, X_test_tcn, y_test_cat,
    num_classes, class_weight=adjusted_class_weight, mode_name="adjusted_weight"
)
