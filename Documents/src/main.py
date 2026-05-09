from preprocessing import load_data, preprocess_data
from model import build_model
from evaluation import evaluate_model
from tuning import tune_hyperparameters


def main():
    # Load and preprocess the data
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()

    X_train, y_train = preprocess_data(X_train, y_train)
    X_val, y_val = preprocess_data(X_val, y_val)
    X_test, y_test = preprocess_data(X_test, y_test)

    # Build the model
    model = build_model()

    # Tune hyperparameters
    best_params = tune_hyperparameters(
        model,
        X_train,
        y_train,
        X_val,
        y_val
    )

    class_weight = {
    0: 2.0,  # NORMAL
    1: 1.0   # PNEUMONIA
    }

    history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=best_params["epochs"],
    batch_size=best_params["batch_size"],
    class_weight=class_weight
    )

    # Evaluate the model on the test set
    evaluate_model(model, X_test, y_test, history)


if __name__ == "__main__":
    main()