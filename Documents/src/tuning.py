def tune_hyperparameters(model, X_train, y_train, X_val, y_val):
    # placeholder for future hyperparameter tuning

    print("Hyperparameter tuning currently set to baseline values.")

    # return default parameters
    best_params = {
        "epochs": 10,
        "batch_size": 32,
        "optimizer": "adam"
    }
    return best_params