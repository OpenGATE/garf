#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import click
import json
import joblib
import xgboost

# Assuming 'garf' is a module in your project that contains 'load_training_dataset'
import garf

# -----------------------------------------------------------------------------
CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])


@click.command(context_settings=CONTEXT_SETTINGS)
@click.argument("json_params_file")
@click.argument("data_file")
@click.argument("output_model_file")
def garf_train_xgboost(json_params_file, data_file, output_model_file):
    """
    \b
    Train an XGBoost model for ARF based on a training dataset.
    <JSON_PARAMS_FILE> : Training parameters (H, L, rr, etc) in json format
    <DATA_FILE> : Dataset in root format
    <OUTPUT_MODEL_FILE> : Filename for the output trained model (e.g., 'model.joblib')
    """

    # --- 1. Load Data and Parameters ---
    print(f"Loading data from '{data_file}'")
    data, theta, phi, E, w = garf.load_training_dataset(data_file)
    x_train = np.column_stack((theta, phi, E))
    y_train = w.astype(int)  # Labels must be integers
    # counts how many different unique windows?
    n_ene_win = np.max(y_train) + 1
    print(f"Number of ENE win: {n_ene_win}")

    print(f"Loading parameters from '{json_params_file}'")
    with open(json_params_file) as f:
        params = json.load(f)
    rr_factor = 500  # params["RR"] FIXME
    print(f"Russian Roulette factor found: {rr_factor}")

    # --- 2. Prepare Data and Metadata ---
    print("Preparing data and calculating weights...")

    # Normalize input features (good practice, though XGBoost is less sensitive)
    x_mean = np.mean(x_train, 0)
    x_std = np.std(x_train, 0)
    x_train_normalized = (x_train - x_mean) / x_std

    # Create model_data dictionary to save with the model
    model_data = {
        "x_mean": x_mean,
        "x_std": x_std,
        "rr": rr_factor,
        "N": len(x_train),
        "n_ene_win": n_ene_win,
        "model_type": "xgboost",  # Add model type for later use
    }

    # Calculate sample weights using the robust hybrid method
    # Note: XGBoost uses 'sample_weight' (one weight per training sample)
    class_counts = np.bincount(y_train)
    class_weights = 1.0 / (class_counts + 1e-9)
    class_weights = np.ones_like(class_counts)
    if rr_factor > 1 and len(class_weights) > 0:
        class_weights[0] *= rr_factor
    # class_weights = class_weights / np.mean(class_weights)

    # Create the final sample_weight array
    sample_weights = class_weights[y_train]
    print(f"Calculated class weights: {class_weights}")

    # --- 3. Define and Train XGBoost Model ---
    print("Training XGBoost model...")
    # These are starter parameters; you can tune them for better performance
    model = xgboost.XGBClassifier(
        objective="multi:softprob",  # Output probabilities for each class
        n_estimators=200,  # Number of boosting rounds (trees) was 200
        max_depth=8,  # Maximum depth of a tree
        learning_rate=0.01,  # Step size shrinkage
        # use_label_encoder=False,
        eval_metric="mlogloss",
        n_jobs=-1,  # Use all available CPU cores
    )

    # Train the model
    model.fit(x_train_normalized, y_train, sample_weight=sample_weights)
    print("Training complete.")

    # --- 4. Save the Model and Metadata ---
    # We save both the trained model and the metadata needed for inference
    output_data = {"model": model, "model_data": model_data}

    print(f"Saving trained model and metadata to '{output_model_file}'")
    joblib.dump(output_data, output_model_file)
    print("Done.")


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    garf_train_xgboost()
