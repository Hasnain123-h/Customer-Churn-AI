# from pathlib import Path
# import pandas as pd
# import tensorflow as tf


# BASE_DIR = Path(__file__).resolve().parent.parent

# X_TRAIN_PATH = BASE_DIR / "data" / "processed" / "X_train_processed.csv"
# Y_TRAIN_PATH = BASE_DIR / "data" / "processed" / "y_train.csv"

# MODEL_PATH = BASE_DIR / "models" / "deep_learning_model.keras"


# def load_data():
#     X_train = pd.read_csv(X_TRAIN_PATH)
#     y_train = pd.read_csv(Y_TRAIN_PATH).squeeze()

#     return X_train, y_train


# def create_model(input_size):
#     model = tf.keras.Sequential([
#         tf.keras.layers.Dense(128, activation="relu", input_shape=(input_size,)),
#         tf.keras.layers.Dense(64, activation="relu"),
#         tf.keras.layers.Dense(1, activation="sigmoid")
#     ])

#     model.compile(
#         optimizer="adam",
#         loss="binary_crossentropy",
#         metrics=["accuracy"]
#     )

#     return model


# if __name__ == "__main__":
#     X_train, y_train = load_data()

#     model = create_model(X_train.shape[1])

#     print("Deep Learning model created successfully.")



from pathlib import Path
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
BASE_DIR = Path(__file__).resolve().parent.parent
X_TRAIN_PATH = BASE_DIR / "data" / "processed" / "X_train_processed.csv"
Y_TRAIN_PATH = BASE_DIR / "data" / "processed" / "y_train.csv"
MODEL_PATH = BASE_DIR / "models" / "deep_learning_model.keras"
def load_data():
    X_train = pd.read_csv(X_TRAIN_PATH)
    y_train = pd.read_csv(Y_TRAIN_PATH).squeeze()

    return X_train, y_train
def create_model(input_size):
    model = Sequential([
        Dense(
            128,
            activation="relu",
            input_shape=(input_size,)
        ),
        Dense(
            64,
            activation="relu"
        ),
        Dense(
            1,
            activation="sigmoid"
        )
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


def train_model(model, X_train, y_train):

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=10,
        mode="min",
        restore_best_weights=True
    )

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        mode="min",
        min_lr=1e-6
    )

    history = model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=100,
        batch_size=32,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )

    return history


def save_model(model):

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    model.save(MODEL_PATH)


if __name__ == "__main__":

    X_train, y_train = load_data()

    model = create_model(X_train.shape[1])

    train_model(model, X_train, y_train)

    save_model(model)

    print("Deep Learning model saved successfully.")