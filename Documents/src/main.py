from preprocessing import load_data, preprocess_data, create_data_generator
from model import build_model
from evaluation import evaluate_model
from tuning import tune_hyperparameters
from svm_model import train_svm, evaluate_svm


def main():
    # Load and preprocess the data
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()

    X_train, y_train = preprocess_data(X_train, y_train)
    X_val, y_val = preprocess_data(X_val, y_val)
    X_test, y_test = preprocess_data(X_test, y_test)

    # Train SVM model
    print("Training SVM model...")
    svm_model = train_svm(X_train, y_train, X_val, y_val)
    evaluate_svm(svm_model, X_test, y_test)

    # Build CNN model
    print("\nTraining CNN model...")
    cnn_model = build_model()

    # Tune hyperparameters
    best_params = tune_hyperparameters(
        cnn_model,
        X_train,
        y_train,
        X_val,
        y_val
    )

    class_weight = {
        0: 2.0,  # NORMAL
        1: 1.0   # PNEUMONIA
    }

    # Use data generator with augmentation for training
    train_generator = create_data_generator(X_train, y_train, batch_size=best_params["batch_size"])

    # Calculate steps per epoch
    steps_per_epoch = len(X_train) // best_params["batch_size"]

    history = cnn_model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        validation_data=(X_val, y_val),
        epochs=best_params["epochs"],
        class_weight=class_weight
    )

    # Evaluate the CNN model on the test set
    evaluate_model(cnn_model, X_test, y_test, history)


if __name__ == "__main__":
    main()