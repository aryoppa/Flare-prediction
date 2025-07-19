# data/preprocessing.py
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.utils import to_categorical

def load_and_preprocess_data(split_ratio):
    print(split_ratio)
    base_dir = "data/datasets"
    image_features = np.load(os.path.join(base_dir, "sdo_image_features_new.npy"))
    valid_data = pd.read_csv(os.path.join(base_dir, "flare_image_metadata_with_label_new.csv"))

    # Filter out invalid features (NaNs)
    mask_valid = ~np.isnan(image_features).any(axis=1)
    valid_data = valid_data[mask_valid].reset_index(drop=True)
    image_features = image_features[mask_valid]

    # Drop rows with NaN in flare_class_idx
    valid_data = valid_data.dropna(subset=['flare_class_idx']).reset_index(drop=True)
    labels = valid_data['flare_class_idx'].astype(int).values

    # Normalize features
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(image_features)

    # Extract and sort by timestamp from image filename
    def extract_timestamp(filename):
        try:
            base = os.path.basename(filename).strip()
            date_str, time_str = base.split('_')[:2]
            return pd.to_datetime(date_str + time_str, format='%Y%m%d%H%M%S')
        except:
            return pd.NaT

    valid_data['timestamp'] = valid_data['image_name'].apply(extract_timestamp)
    valid_data = valid_data.dropna(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)

    scaled_features = scaled_features[valid_data.index]

    # Re-map flare labels (S -> A)
    data_fix = valid_data[['flare_label', 'flare_class_idx', 'timestamp', 'full_path', 'region']].copy()
    data_fix.loc[data_fix['flare_label'] == 'S', 'flare_label'] = 'A'

    label_map = {'A': 0, 'B': 1, 'C': 2, 'M': 3, 'X': 4}
    data_fix['flare_class_idx'] = data_fix['flare_label'].map(label_map)

    # Final label array
    labels = data_fix['flare_class_idx'].values

    # One-hot encode labels
    num_classes = len(np.unique(labels))
    labels_cat = to_categorical(labels, num_classes=num_classes)

    # Time-based split
    split_index = int(len(data_fix) * split_ratio / 100)
    print(f"Split index: {split_index}")
    X_train = scaled_features[:split_index]
    X_test = scaled_features[split_index:]
    y_train = labels[:split_index]
    y_test = labels[split_index:]
    y_train_cat = labels_cat[:split_index]
    y_test_cat = labels_cat[split_index:]

    # Reshape for TCN: (samples, timesteps, features)
    X_train_tcn = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test_tcn = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
    print(f"Train shape: {X_train_tcn.shape}, Test shape: {X_test_tcn.shape}")
    
    # Compute class weights
    from sklearn.utils.class_weight import compute_class_weight
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(zip(np.unique(y_train), class_weights))

    return X_train_tcn, X_test_tcn, y_train_cat, y_test_cat, y_train, y_test, split_index, data_fix, class_weight_dict
