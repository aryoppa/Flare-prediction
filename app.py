import argparse
import pickle
import os
from model.tcn_model import run_training
from model.evaluate import run_evaluation
from data.preprocessing import load_and_preprocess_data
from tcn import TCN

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Solar Flare Prediction using TCN")
    parser.add_argument("--mode", choices=["plain", "weighted"], default="plain",
                        help="Choose whether to train model with or without class weight")
    args = parser.parse_args()

    # --- Folder untuk model dan history
    MODEL_DIR = 'trained_model'
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Load data
    print("\nLoading and preprocessing data...")
    split_ratio = 70  # tambahkan jika belum didefinisikan
    X_train_tcn, X_test_tcn, y_train_cat, y_test_cat, y_train, y_test, split_index, valid_data, class_weight_dict = load_and_preprocess_data(split_ratio)

    # Train model
    print(f"\nTraining model ({args.mode})...")
    model, history = run_training(X_train_tcn, y_train_cat, X_test_tcn, y_test_cat,
                                  class_weight=class_weight_dict if args.mode == "weighted" else None)

    # Save model ke folder trained_model, pakai format .keras!

    model_path = os.path.join(MODEL_DIR, f'solarflare_model_{args.mode}.keras')
    model = tf.keras.models.load_model(model_path, custom_objects={'TCN': TCN})
    model.save(model_path)
    print(f"\nModel saved as {model_path}")

    # Save training history (optional) ke folder trained_model
    history_path = os.path.join(MODEL_DIR, f'train_history_{args.mode}.pkl')
    with open(history_path, 'wb') as f:
        pickle.dump(history.history, f)
    print(f"Training history saved as {history_path}")

    # Evaluate model
    print("\nEvaluating model...")
    run_evaluation(model, X_train_tcn, y_train_cat, X_test_tcn, y_test_cat,
                   split_index, valid_data, mode=args.mode)
