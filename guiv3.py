import os
import pickle
import streamlit as st
import pandas as pd
import numpy as np

st.title("Solar Flare Model Summary (v5)")

folder = st.text_input("Enter history folder path:", value="trained_model")

if not os.path.isdir(folder):
    st.warning("Please enter a valid folder path above.")
    st.stop()

# Label mapping
label_map = {0: 'A', 1: 'B', 2: 'C', 3: 'M', 4: 'X'}
class_names = [label_map[i] for i in sorted(label_map.keys())]

# Gather model summaries
model_rows = []
results = []

for fname in sorted(os.listdir(folder)):
    if fname.startswith("train_history_manual_") and fname.endswith(".pkl"):
        with open(os.path.join(folder, fname), 'rb') as f:
            hist = pickle.load(f)
        model = fname.replace("train_history_manual_", "").replace(".pkl", "")
        record = {
            "Model": model,
            "Final Val Accuracy": hist.get('val_accuracy', [None])[-1],
            "Final Val Loss": hist.get('val_loss', [None])[-1],
            "Best Val Accuracy": max(hist.get('val_accuracy', [None])),
            "Epoch of Best Val Acc": hist.get('val_accuracy', [None]).index(max(hist.get('val_accuracy', [None]))),
            "Min Val Loss": min(hist.get('val_loss', [None])),
            "Epoch of Min Val Loss": hist.get('val_loss', [None]).index(min(hist.get('val_loss', [None])))
        }
        # Analyze predicted classes if available
        if 'y_pred' in hist:
            record['# Classes Predicted'] = len(set(hist['y_pred']))
        else:
            record['# Classes Predicted'] = "?"
        if 'y_test' in hist:
            record['# Classes True'] = len(set(hist['y_test']))
        else:
            record['# Classes True'] = "?"
        model_rows.append(record)
        
        # Per-class detailed comparison
        if 'y_test' in hist and 'y_pred' in hist:
            y_test = np.array(hist['y_test'])
            y_pred = np.array(hist['y_pred'])
            res_row = {"Model": model}
            for idx, cname in enumerate(class_names):
                n_true = int(np.sum(y_test == idx))
                n_correct = int(np.sum((y_test == idx) & (y_pred == idx)))
                recall = n_correct / n_true if n_true > 0 else 0
                res_row[f"Test {cname}"] = n_true
                res_row[f"Correct {cname}"] = n_correct
                res_row[f"Recall {cname}"] = recall
            results.append(res_row)

df = pd.DataFrame(model_rows)
df_results = pd.DataFrame(results)

# Main model metrics table
st.header("Model Metrics Summary")
st.dataframe(df, use_container_width=True)

# Download summary
csv = df.to_csv(index=False).encode()
st.download_button("Download model summary as CSV", csv, "model_summary.csv", "text/csv")

# Per-class performance
if not df_results.empty:
    st.header("Per-Class Test vs. Correct Predictions (Recall in each class)")
    st.dataframe(df_results, use_container_width=True)
    # Download per-class
    csv2 = df_results.to_csv(index=False).encode()
    st.download_button("Download per-class recall as CSV", csv2, "model_perclass_summary.csv", "text/csv")

    # --- Per-Class Correct Prediction Bar Charts ---
    st.header("Per-Class Correct Predictions (Bar Chart for Each Class)")
    for cname in class_names:
        st.subheader(f"Class {cname}: Correct Predictions by Model")
        chart_data = df_results[["Model", f"Correct {cname}"]].set_index("Model")
        st.bar_chart(chart_data)

    st.header("Per-Class Recall (Bar Chart for Each Class)")
    for cname in class_names:
        st.subheader(f"Class {cname}: Recall by Model")
        recall_data = df_results[["Model", f"Recall {cname}"]].set_index("Model")
        st.bar_chart(recall_data)
else:
    st.info("No y_test/y_pred found in history files for per-class analysis.")

# Visualize overall accuracy
st.subheader("Final Validation Accuracy (bar chart)")
if len(df) > 0:
    st.bar_chart(df.set_index("Model")["Final Val Accuracy"])
