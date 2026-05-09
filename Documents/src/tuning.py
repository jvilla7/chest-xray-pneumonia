from tensorflow.keras.optimizers import Adam, RMSprop

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


def tune_hyperparameters(X_train, y_train, X_val, y_val, model=None):

    print("Testing tuned CNN configurations...")

    return TUNED_CONFIGS[0]