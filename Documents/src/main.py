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

    # Tune hyperparameters (optional)
    best_params = tune_hyperparameters(model, X_train, y_train, X_val, y_val)

    # Train the model with the best hyperparameters
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10)

    # Evaluate the model on the test set
    evaluate_model(model, X_test, y_test)
if __name__ == "__main__":
    main()