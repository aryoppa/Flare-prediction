import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from data.preprocessing import load_and_preprocess_data
from sklearn.preprocessing import LabelEncoder
import os
from tcn import TCN 

st.set_page_config(page_title="Solar Flare Model Comparison", layout="centered")
st.title("☀️ Solar Flare Prediction: Keras-TCN vs Manual TCN")

# --- Path model disimpan di folder trained_model/ ---
MODEL_DIR = 'trained_model'
MANUAL_MODEL_PATH = os.path.join(MODEL_DIR, 'solarflare_tcn_manual_plain.h5')
MANUAL_MODEL_WEIGHTED_PATH = os.path.join(MODEL_DIR, 'solarflare_tcn_manual_weighted.h5')

# --- Load data dengan fungsi preprocessing ---
with st.spinner("Loading and preprocessing data..."):
    X_train, X_test, y_train_cat, y_test_cat, y_train, y_test, split_index, valid_data, class_weight_dict = load_and_preprocess_data(split_ratio=70)

if 'class' in valid_data.columns:
    labels = valid_data['flare_class'].values
else:
    labels = y_test  # fallback

le = LabelEncoder()
le.fit(labels)
class_labels = le.classes_

y_test_true = np.argmax(y_test_cat, axis=1)
y_train_true = np.argmax(y_train_cat, axis=1)

st.markdown("## Model Comparison")

st.markdown("### Pilih Model yang Akan Dibandingkan")
model_option = st.radio("Pilih mode training:", ["plain", "weighted"])

if model_option == "plain":
    keras_model_path = MANUAL_MODEL_PATH
    manual_model_path = MANUAL_MODEL_PATH
else:
    keras_model_path = MANUAL_MODEL_WEIGHTED_PATH
    manual_model_path = MANUAL_MODEL_WEIGHTED_PATH

col1, col2, col3, col4= st.columns(4)
with col1:
    st.subheader(f"Manual TCN Model ({model_option}) Fitting")
    try:
        model_manual = tf.keras.models.load_model(manual_model_path)
        y_pred_manual = np.argmax(model_manual.predict(X_train), axis=1)
        acc_manual = np.mean(y_pred_manual == y_train_true)
        report_manual = classification_report(y_train_true, y_pred_manual, target_names=class_labels, output_dict=True)
        cm_manual = confusion_matrix(y_train_true, y_pred_manual)
        st.write(f"**Accuracy:** {acc_manual:.4f}")
        st.write("Classification Report")
        st.dataframe(pd.DataFrame(report_manual).transpose())
        st.write("Confusion Matrix")
        fig2, ax2 = plt.subplots()
        sns.heatmap(cm_manual, annot=True, fmt='d', cmap='Greens', xticklabels=class_labels, yticklabels=class_labels)
        st.pyplot(fig2)
    except Exception as e:
        st.error(f"Gagal load/model manual: {e}")


with col2:
    st.subheader(f"Manual TCN Model ({model_option}) Testing")
    try:
        model_manual = tf.keras.models.load_model(manual_model_path)
        y_pred_manual = np.argmax(model_manual.predict(X_test), axis=1)
        acc_manual = np.mean(y_pred_manual == y_test_true)
        report_manual = classification_report(y_test_true, y_pred_manual, target_names=class_labels, output_dict=True)
        cm_manual = confusion_matrix(y_test_true, y_pred_manual)
        st.write(f"**Accuracy:** {acc_manual:.4f}")
        st.write("Classification Report")
        st.dataframe(pd.DataFrame(report_manual).transpose())
        st.write("Confusion Matrix")
        fig2, ax2 = plt.subplots()
        sns.heatmap(cm_manual, annot=True, fmt='d', cmap='Greens', xticklabels=class_labels, yticklabels=class_labels)
        st.pyplot(fig2)
    except Exception as e:
        st.error(f"Gagal load/model manual: {e}")

# with col3:
#     st.subheader(f"Manual TCN Model ({model_option}) Fitting")
#     try:
#         model_manual = tf.keras.models.load_model(manual_model_path)
#         y_pred_manual = np.argmax(model_manual.predict(X_train), axis=1)
#         acc_manual = np.mean(y_pred_manual == y_train_true)
#         report_manual = classification_report(y_train_true, y_pred_manual, target_names=class_labels, output_dict=True)
#         cm_manual = confusion_matrix(y_train_true, y_pred_manual)
#         st.write(f"**Accuracy:** {acc_manual:.4f}")
#         st.write("Classification Report")
#         st.dataframe(pd.DataFrame(report_manual).transpose())
#         st.write("Confusion Matrix")
#         fig3, ax3 = plt.subplots()
#         sns.heatmap(cm_manual, annot=True, fmt='d', cmap='Greens', xticklabels=class_labels, yticklabels=class_labels)
#         st.pyplot(fig3)
#     except Exception as e:
#         st.error(f"Gagal load/model manual: {e}")

# with col4:
#     st.subheader(f"Manual TCN Model ({model_option}) Testing")
#     try:
#         model_manual = tf.keras.models.load_model(manual_model_path)
#         y_pred_manual = np.argmax(model_manual.predict(X_test), axis=1)
#         acc_manual = np.mean(y_pred_manual == y_test_true)
#         report_manual = classification_report(y_test_true, y_pred_manual, target_names=class_labels, output_dict=True)
#         cm_manual = confusion_matrix(y_test_true, y_pred_manual)
#         st.write(f"**Accuracy:** {acc_manual:.4f}")
#         st.write("Classification Report")
#         st.dataframe(pd.DataFrame(report_manual).transpose())
#         st.write("Confusion Matrix")
#         fig2, ax2 = plt.subplots()
#         sns.heatmap(cm_manual, annot=True, fmt='d', cmap='Greens', xticklabels=class_labels, yticklabels=class_labels)
#         st.pyplot(fig2)
#     except Exception as e:
#         st.error(f"Gagal load/model manual: {e}")

# --- Akurasi Comparison Chart ---
if 'acc_keras' in locals() and 'acc_manual' in locals():
    st.markdown("## 📊 Perbandingan Akurasi")
    compare_df = pd.DataFrame({
        "Keras-TCN": [acc_keras],
        "Manual TCN": [acc_manual]
    }, index=["Accuracy"])
    st.bar_chart(compare_df.T)

    

st.markdown("""
---
> **Note:**  
> Pastikan kedua model (`solarflare_model_*.keras` dan `solarflare_tcn_manual_*.h5`) sudah ditraining dengan dataset dan preprocessing yang sama!
> Model diambil dari folder `trained_model/`
""")
