# app.py
import argparse
from model.tcn_model import run_training
from model.evaluate import run_evaluation
from data.preprocessing import load_and_preprocess_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Solar Flare Prediction using TCN")
    parser.add_argument("--mode", choices=["plain", "weighted"], default="plain",
                        help="Choose whether to train model with or without class weight")
    args = parser.parse_args()

    # Load data
    print("\nLoading and preprocessing data...")
    X_train_tcn, X_test_tcn, y_train_cat, y_test_cat, y_train, y_test, split_index, valid_data, class_weight_dict = load_and_preprocess_data()

    # Train model
    print(f"\nTraining model ({args.mode})...")
    model, history = run_training(X_train_tcn, y_train_cat, X_test_tcn, y_test_cat,
                                  class_weight=class_weight_dict if args.mode == "weighted" else None)

    # Evaluate model
    print("\nEvaluating model...")
    run_evaluation(model, X_train_tcn, y_train_cat, X_test_tcn, y_test_cat,
                   split_index, valid_data, mode=args.mode)
