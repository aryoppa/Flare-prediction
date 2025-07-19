# streamlit_app.py
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from model.tcn_model import build_model
from data.preprocessing import load_and_preprocess_data
from utils.helpers import index_to_class_map
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf

st.set_page_config(page_title="Solar Flare Prediction", layout="centered")
st.title("☀️ Solar Flare Class Prediction App")

# --- Sidebar ---
split_ratio = st.sidebar.slider("Split Ratio", 1, 100, 70)
epochs = st.sidebar.slider("Epochs", 1, 20, 5)
batch_size = st.sidebar.selectbox("Batch Size", [16, 32, 64], index=1)

with st.spinner("Loading and preprocessing data..."):
    X_train_tcn, X_test_tcn, y_train_cat, y_test_cat, y_train, y_test, split_index, data_fix, class_weight_dict = load_and_preprocess_data(split_ratio)
    num_classes = y_train_cat.shape[1]
    st.write(f"Train shape: {X_train_tcn.shape}, Test shape: {X_test_tcn.shape}")

# ...existing code...
st.success("Data loaded!")

# --- Data validation ---
if np.isnan(X_test_tcn).any() or np.isinf(X_test_tcn).any():
    st.error("X_test_tcn contains NaN or infinite values. Please check your preprocessing.")
    st.stop()
if X_test_tcn.shape[0] == 0:
    st.error("X_test_tcn is empty. Adjust your split ratio or check your data.")
    st.stop()
# ...existing code...

# --- Train model ---
st.subheader("Train and Compare Models")
if st.button("Train Both Models"):
    with st.spinner("Training weighted model..."):
        model_weighted = build_model(input_shape=(1, X_train_tcn.shape[2]), num_classes=num_classes)
        model_weighted.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model_weighted.fit(
            X_train_tcn, y_train_cat,
            validation_data=(X_test_tcn, y_test_cat),
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weight_dict,
            verbose=0
        )

    with st.spinner("Training plain model..."):
        model_plain = build_model(input_shape=(1, X_train_tcn.shape[2]), num_classes=num_classes)
        model_plain.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model_plain.fit(
            X_train_tcn, y_train_cat,
            validation_data=(X_test_tcn, y_test_cat),
            epochs=epochs,
            batch_size=batch_size,
            verbose=0
        )

    y_pred_w = np.argmax(model_weighted.predict(X_test_tcn), axis=1)
    y_pred_p = np.argmax(model_plain.predict(X_test_tcn), axis=1)
    y_true = np.argmax(y_test_cat, axis=1)

    st.subheader("📊 Confusion Matrix Comparison")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    cm_w = confusion_matrix(y_true, y_pred_w)
    cm_p = confusion_matrix(y_true, y_pred_p)
    sns.heatmap(cm_w, annot=True, fmt='d', cmap='Blues', xticklabels=index_to_class_map.values(), yticklabels=index_to_class_map.values(), ax=axes[0])
    axes[0].set_title("Weighted")
    sns.heatmap(cm_p, annot=True, fmt='d', cmap='Oranges', xticklabels=index_to_class_map.values(), yticklabels=index_to_class_map.values(), ax=axes[1])
    axes[1].set_title("Plain")
    st.pyplot(fig)

    st.subheader("📈 Accuracy Comparison")
    acc_w = np.mean(y_pred_w == y_true) * 100
    acc_p = np.mean(y_pred_p == y_true) * 100
    st.metric("Accuracy (Weighted)", f"{acc_w:.2f}%")
    st.metric("Accuracy (Plain)", f"{acc_p:.2f}%")

    st.subheader("📃 Classification Report")
    st.text("Weighted Model")
    st.text(classification_report(y_true, y_pred_w, target_names=[index_to_class_map[i] for i in sorted(np.unique(y_true))]))
    st.text("Plain Model")
    st.text(classification_report(y_true, y_pred_p, target_names=[index_to_class_map[i] for i in sorted(np.unique(y_true))]))

# --- Manual Prediction ---
# st.subheader("Manual Feature Vector Prediction")
# with st.form("manual_form"):
#     st.write("Masukkan 1280 nilai CNN feature vector (dipisahkan koma):")
#     input_str = st.text_area("Vector Input", height=150)
#     submitted = st.form_submit_button("Predict")

#     if submitted:
#         try:
#             vec = np.array([float(x.strip()) for x in input_str.strip().split(',')])
#             if vec.shape[0] != X_train_tcn.shape[2]:
#                 st.error(f"Vector harus memiliki panjang {X_train_tcn.shape[2]}")
#             else:
#                 model = build_model(input_shape=(1, vec.shape[0]), num_classes=num_classes)
#                 model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#                 model.fit(X_train_tcn, y_train_cat, epochs=3, verbose=0)

#                 input_vec = vec.reshape((1, 1, -1))
#                 pred = model.predict(input_vec)
#                 pred_class = np.argmax(pred)
#                 st.success(f"Predicted Class: {index_to_class_map[pred_class]} ({pred[0][pred_class]*100:.2f}%)")
#         except Exception as e:
#             st.error(f"Error: {str(e)}")
