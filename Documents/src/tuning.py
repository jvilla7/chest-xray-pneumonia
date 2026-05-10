from tensorflow.keras.optimizers import Adam, RMSprop
import numpy as np

BASELINE_CONFIG = {
    "epochs": 10,
    "batch_size": 32,
    "optimizer": Adam(learning_rate=0.001),
    "dropout_rate": 0.3
}

TUNED_CONFIGS = [
    {
        "epochs": 15,
        "batch_size": 32,
        "optimizer": Adam(learning_rate=0.001),
        "dropout_rate": 0.4
    },

    {
        "epochs": 20,
        "batch_size": 64,
        "optimizer": Adam(learning_rate=0.0005),
        "dropout_rate": 0.5
    },

    {
        "epochs": 15,
        "batch_size": 32,
        "optimizer": RMSprop(learning_rate=0.0005),
        "dropout_rate": 0.4
    }
]


def tune_hyperparameters(model, X_train, y_train, X_val, y_val):
    """Perform simple hyperparameter tuning by training on each config and selecting best validation accuracy."""
    print("Performing hyperparameter tuning...")

    best_config = None
    best_val_accuracy = 0.0

    for i, config in enumerate(TUNED_CONFIGS):
        print(f"Testing config {i+1}: {config}")

        # Temporarily modify model for this config
        model_copy = model.__class__.from_config(model.get_config())  # Create a fresh copy
        model_copy.compile(
            optimizer=config["optimizer"],
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )

        # Quick training on subset for tuning (to save time)
        subset_size = min(1000, len(X_train))  # Use subset for faster tuning
        indices = np.random.choice(len(X_train), subset_size, replace=False)
        X_train_subset = X_train[indices]
        y_train_subset = y_train[indices]

        history = model_copy.fit(
            X_train_subset,
            y_train_subset,
            validation_data=(X_val, y_val),
            epochs=config["epochs"] // 2,  # Shorter for tuning
            batch_size=config["batch_size"],
            verbose=0  # Silent
        )

        val_accuracy = max(history.history["val_accuracy"])
        print(f"  Validation Accuracy: {val_accuracy:.4f}")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_config = config

    print(f"Selected best config: {best_config} with val_acc {best_val_accuracy:.4f}")
    return best_config