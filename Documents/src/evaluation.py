import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay


def evaluate_model(model, X_test, y_test):
    # evaluate performance on test set
    loss, accuracy = model.evaluate(X_test, y_test)

    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")

    # generate predictions
    y_prob = model.predict(X_test)
    y_pred = (y_prob > 0.5).astype(int).flatten()

    # classification metrics
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["NORMAL", "PNEUMONIA"]))

    # confusion matrix visualization
    cm = confusion_matrix(y_test, y_pred)
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["NORMAL", "PNEUMONIA"]
    )
    display.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.show()