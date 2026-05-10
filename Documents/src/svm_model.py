import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def flatten_images(X):
    """Flatten image arrays for SVM input."""
    return X.reshape(X.shape[0], -1)

def train_svm(X_train, y_train, X_val, y_val):
    """Train SVM model with hyperparameter tuning."""
    # Flatten images
    X_train_flat = flatten_images(X_train)
    X_val_flat = flatten_images(X_val)

    # Use a subset for faster training (optional, for demo)
    subset_size = min(2000, len(X_train_flat))  # Limit to 2000 samples for speed
    indices = np.random.choice(len(X_train_flat), subset_size, replace=False)
    X_train_subset = X_train_flat[indices]
    y_train_subset = y_train[indices]

    # Simple SVM with RBF kernel
    svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
    svm_model.fit(X_train_subset, y_train_subset)

    # Validate on validation set
    val_predictions = svm_model.predict(X_val_flat)
    val_accuracy = np.mean(val_predictions == y_val)
    print(f"SVM Validation Accuracy: {val_accuracy:.4f}")

    return svm_model

def evaluate_svm(svm_model, X_test, y_test):
    """Evaluate SVM on test set."""
    X_test_flat = flatten_images(X_test)
    y_pred = svm_model.predict(X_test_flat)

    print("SVM Test Results:")
    print(f"Test Accuracy: {np.mean(y_pred == y_test):.4f}")

    print("\nSVM Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["NORMAL", "PNEUMONIA"]))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["NORMAL", "PNEUMONIA"])
    display.plot(cmap="Blues")
    plt.title("SVM Confusion Matrix")
    plt.show()