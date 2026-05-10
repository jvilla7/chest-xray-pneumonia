import os
import numpy as np
import tensorflow as tf

IMG_SIZE = 128
# add datSET DIR as a global variable for easier access across modules
#DATASET_DIR = "chest_xray"
DATASET_DIR = r"C:\Users\jvilla24\Documents\GitHub\chest-xray-pneumonia\Documents\data\chest_xray\chest_xray"

def load_images(folder):
    X = []
    y = []

    labels = {
        "NORMAL": 0,
        "PNEUMONIA": 1
    }

    for label_name, label_value in labels.items():
        path = os.path.join(folder, label_name)

        if not os.path.exists(path):
            raise FileNotFoundError(f"Folder not found: {path}")

        for file in os.listdir(path):
            if not file.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            img_path = os.path.join(path, file)

            img = tf.keras.utils.load_img(
                img_path,
                color_mode="grayscale",
                target_size=(IMG_SIZE, IMG_SIZE)
            )

            img = tf.keras.utils.img_to_array(img)

            X.append(img)
            y.append(label_value)

    return np.array(X), np.array(y)


def load_data():
    X_train, y_train = load_images(os.path.join(DATASET_DIR, "train"))
    X_val, y_val = load_images(os.path.join(DATASET_DIR, "val"))
    X_test, y_test = load_images(os.path.join(DATASET_DIR, "test"))

    return X_train, y_train, X_val, y_val, X_test, y_test


def preprocess_data(X, y):
    X = X.astype("float32") / 255.0
    return X, y


def create_data_generator(X_train, y_train, batch_size=32):
    """Create data generator with augmentation for training."""
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Fit the generator on training data
    datagen.fit(X_train)

    return datagen.flow(X_train, y_train, batch_size=batch_size)