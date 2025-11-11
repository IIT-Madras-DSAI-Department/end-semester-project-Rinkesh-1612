import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from algorithms import KNearestNeighbors, PCA, f1_score, accuracy_score

os.makedirs('report_figures', exist_ok=True)

def load_and_prepare_data(train_path, val_path):
    train_df = pd.read_csv(train_path, header=None)
    val_df = pd.read_csv(val_path, header=None)
    if isinstance(train_df.iloc[0, 0], str):
        train_df = train_df.iloc[1:].reset_index(drop=True)
        val_df = val_df.iloc[1:].reset_index(drop=True)
    X_train = train_df.iloc[:, 1:785].values.astype(np.float32)
    y_train = train_df.iloc[:, 0].values.astype(np.int64)
    X_val = val_df.iloc[:, 1:785].values.astype(np.float32)
    y_val = val_df.iloc[:, 0].values.astype(np.int64)
    X_train = X_train / 255.0
    X_val = X_val / 255.0
    return X_train, y_train, X_val, y_val


if __name__ == "__main__":
    X_train, y_train, X_val, y_val = load_and_prepare_data(
        r'C:\Users\Rinkesh\OneDrive\Desktop\Machine Learning\machine_learning_end_sem\data\MNIST_train.csv',
        r'C:\Users\Rinkesh\OneDrive\Desktop\Machine Learning\machine_learning_end_sem\data\MNIST_validation.csv'
    )

    pca_champion = PCA(n_components=40)
    pca_champion.fit(X_train)
    X_train_pca = pca_champion.transform(X_train)
    X_val_pca = pca_champion.transform(X_val)
    champion_model = KNearestNeighbors(k=5)  
    champion_model.fit(X_train_pca, y_train)
    y_pred = champion_model.predict(X_val_pca)
   
    overall_f1 = f1_score(y_val, y_pred)
    overall_acc = accuracy_score(y_val, y_pred)
    print(f"Overall Validation F1-Score: {overall_f1:.4f}")
    print(f"Overall Validation Accuracy: {overall_acc:.4f}")
    
    # Confusion Matrix
    labels = np.unique(y_val)
    confusion_matrix = np.zeros((len(labels), len(labels)), dtype=int)
    for true_label, pred_label in zip(y_val, y_pred):
        confusion_matrix[true_label, pred_label] += 1
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix for Champion Model (k-NN on 40 PCA Components)')
    plt.savefig('report_figures/confusion_matrix.png')
    plt.close()

    np.fill_diagonal(confusion_matrix, 0)  
    max_confusion = np.unravel_index(np.argmax(confusion_matrix), confusion_matrix.shape)
    
    # Per-Class Performance Bar Plot
    per_class_f1 = []
    for digit in labels:
        y_true_bin = (y_val == digit)
        y_pred_bin = (y_pred == digit)
        tp = np.sum(y_true_bin & y_pred_bin)
        fp = np.sum(~y_true_bin & y_pred_bin)
        fn = np.sum(y_true_bin & ~y_pred_bin)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        per_class_f1.append(f1)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=labels, y=per_class_f1, palette='viridis')
    plt.xlabel('Digit')
    plt.ylabel('F1-Score')
    plt.title('Per-Class F1-Scores for Champion Model')
    plt.ylim(0.8, 1.0)
    plt.savefig('report_figures/per_class_f1.png')
    plt.close()
    
    # PCA Cumulative Explained Variance
    X_centered = X_train - np.mean(X_train, axis=0)
    cov_matrix = np.cov(X_centered.T)
    eigenvalues, _ = np.linalg.eigh(cov_matrix)
    sorted_eigenvalues = np.sort(eigenvalues)[::-1]
    cumulative_variance = np.cumsum(sorted_eigenvalues) / np.sum(sorted_eigenvalues)
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, linestyle='-')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.title('Cumulative Explained Variance by PCA Components')
    plt.grid(True)
    plt.axhline(y=0.90, color='g', linestyle=':', label='90% Variance Threshold')
    plt.axvline(x=40, color='r', linestyle=':', label='Chosen n_components = 40')
    plt.legend()
    plt.xlim(0, 200)
    plt.ylim(0, 1.05)
    plt.savefig('report_figures/pca_variance_explained.png')
    plt.close()
    
    # 2D PCA Projection for Class Separation
    pca_2d = PCA(n_components=2)
    pca_2d.fit(X_train)
    X_val_2d = pca_2d.transform(X_val)
    df_2d = pd.DataFrame({'PC1': X_val_2d[:, 0], 'PC2': X_val_2d[:, 1], 'Label': y_val})
    
    plt.figure(figsize=(12, 10))
    sns.scatterplot(x='PC1', y='PC2', hue='Label', palette=sns.color_palette("hls", 10), data=df_2d, legend="full", s=20)
    plt.title('Validation Data Projected onto First Two Principal Components')
    plt.savefig('report_figures/class_separation_2d.png')
    plt.close()
    
    # Misclassified Examples Grid
    misclassified_indices = np.where(y_val != y_pred)[0]
    if len(misclassified_indices) > 0:
        num_examples = min(25, len(misclassified_indices))
        fig, axes = plt.subplots(5, 5, figsize=(12, 12))
        for i, ax in enumerate(axes.flatten()):
            if i < num_examples:
                idx = misclassified_indices[i]
                img = X_val[idx].reshape(28, 28)
                ax.imshow(img, cmap='gray')
                ax.set_title(f"True: {y_val[idx]} Pred: {y_pred[idx]}", fontsize=8)
                ax.axis('off')
            else:
                ax.axis('off')
        plt.tight_layout()
        plt.suptitle('Examples of Misclassified Digits', y=1.02)
        plt.savefig('report_figures/misclassified_examples.png')
        plt.close()