from tensorflow.keras import layers, models

IMG_SIZE = 128  # consistent image size


def build_model():
    # baseline CNN architecture
    model = models.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)),  # ADDED

        # convolutional layers for feature extraction
        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),

        # classification head
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),  # ADDED: reduce overfitting
        layers.Dense(1, activation="sigmoid")  # binary output
    ])

    # compile model
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model