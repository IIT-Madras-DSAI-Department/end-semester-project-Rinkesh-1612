# import os
# os.environ['OMP_NUM_THREADS'] = '1'
# os.environ['OPENBLAS_NUM_THREADS'] = '1'
# os.environ['MKL_NUM_THREADS'] = '1'
# os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
# os.environ['NUMEXPR_NUM_THREADS'] = '1'  # Uncomment for single-core timing

import pandas as pd
import numpy as np
import time
from algorithms import PCA, KNearestNeighbors, f1_score, accuracy_score

def load_and_prepare_data(train_path, val_path):
    train_df = pd.read_csv(train_path, header=None, low_memory=False)
    val_df = pd.read_csv(val_path, header=None, low_memory=False)
    if isinstance(train_df.iloc[0, 0], str):
        train_df = train_df.iloc[1:].reset_index(drop=True)
        val_df = val_df.iloc[1:].reset_index(drop=True)

    y_train = train_df.iloc[:, 0].values.astype(np.int64)
    X_train = train_df.iloc[:, 1:785].values.astype(np.float32)
    y_val = val_df.iloc[:, 0].values.astype(np.int64)
    X_val = val_df.iloc[:, 1:785].values.astype(np.float32)
    
    X_train /= 255.0
    X_val /= 255.0
    
    return X_train, y_train, X_val, y_val


if __name__ == "__main__":
    TRAIN_PATH = r'MNIST_train.csv'
    VALIDATION_PATH = r'MNIST_validation.csv'
    
    X_train_raw, y_train, X_val_raw, y_val = load_and_prepare_data(TRAIN_PATH, VALIDATION_PATH)
    
    # Optimal hyperparameters (from my tuning)
    n_components = 40
    k = 5

    # --- Train PCA ---
    pca_start = time.time()
    pca = PCA(n_components=n_components)
    pca.fit(X_train_raw)
    X_train_final = pca.transform(X_train_raw)
    X_val_final = pca.transform(X_val_raw)
    pca_end = time.time()

    # --- Train KNN ---
    knn_fit_start = time.time()
    model = KNearestNeighbors(k=k, weights='distance')
    model.fit(X_train_final, y_train)
    knn_fit_end = time.time()

    # --- Predict (this is the slow part!) ---
    predict_start = time.time()
    y_pred = model.predict(X_val_final)
    predict_end = time.time()

    # --- Compute metrics ---
    final_f1 = f1_score(y_val, y_pred)
    final_accuracy = accuracy_score(y_val, y_pred)

    # --- Total times ---
    pca_time = pca_end - pca_start
    knn_fit_time = knn_fit_end - knn_fit_start  # ~0
    predict_time = predict_end - predict_start
    total_training_time = pca_time + knn_fit_time  # Still ~0.17s
    total_system_time = pca_time + knn_fit_time + predict_time  # ~100s
    


    print("\n" + "="*60)
    print(" " * 15 + "FINAL SYSTEM PERFORMANCE REPORT")
    print("="*60)
    print(f"Algorithm:                   K-Nearest Neighbors")
    print(f"Preprocessing:               PCA")
    print(f"Hyperparameters:             n_components = {n_components}, k = {k}")
    print("-" * 60)
    print(f"Validation F1-Score:         {final_f1:.4f}")
    print(f"Validation Accuracy:         {final_accuracy:.4f}")
    print("-" * 60)
    print(f"PCA Training Time:           {pca_time:.2f} seconds")
    print(f"KNN Fit Time:                {knn_fit_time:.2f} seconds")
    print(f"Prediction Time (val):       {predict_time:.2f} seconds")
    print(f"Total System Time:           {total_system_time:.2f} seconds")
    print("="*60)

    if total_training_time > 300:
        print("WARNING: Training time exceeds 5-minute target.")
    else:
        print("SUCCESS: Training time is within 5-minute target.")


