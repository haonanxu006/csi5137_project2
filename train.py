import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import json
import argparse

from myModel_RQ import (
    RQ_sel_2Input_DENSE2L_DENSE3L,
    RQ_sel_2Input_CNN2L_CNN1L_DENSE2L
)

def load_dataset(folder):
    print("Loading dataset...")
    x = np.load(folder + "/x.npy")
    x1 = np.load(folder + "/x1.npy")
    y = np.load(folder + "/y.npy")

    return x, x1, y

def build_model(type, dimx, dimy, hp):
    if type == "dnn":
        print("Buiding M2-DNN...")
        return RQ_sel_2Input_DENSE2L_DENSE3L(
            dimx=dimx,
            dimy=dimy,
            f1=hp["f1"],
            f2=hp["f2"],
            f3=hp["f3"],
            f4=hp["f4"],
            f5=hp["f5"]
        )
    elif type == "cnn":
        print("Building M2-CNN...")
        return RQ_sel_2Input_CNN2L_CNN1L_DENSE2L(
            dimx=dimx,
            dimy=dimy,
            f1=hp["f1"],
            f2=hp["f2"],
            f3=hp["f3"],
            f4=hp["f4"]
        )
    else:
        raise ValueError("Invalid model type:", type)

def wmape(y_true, y_pred):
    return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true) + 1e-8)

def train(config):
    folder = config["dataset_path"]
    type = config["model_type"]
    hp = config["hyperparameters"]

    batch_size = config["batch_size"]
    epochs = config["epochs"]
    patience = config["patience"]

    x, x1, y = load_dataset(folder)

    y = y.reshape(-1, 1)

    x_train, x_test, x1_train, x1_test, y_train, y_test = train_test_split(
        x, x1, y, test_size=0.2, random_state=42
    )

    # create validation set
    x_train, x_valid, x1_train, x1_valid, y_train, y_valid = train_test_split(
        x_train, x1_train, y_train, test_size=0.2, random_state=42
    )

    dimx, dimy = x.shape[1], x.shape[2]
    model = build_model(type, dimx, dimy, hp)

    model.compile(optimizer=Adam(1e-3), loss="mse")

    model([x_train[:1], x1_train[:1]])
    model.summary()

    callback = EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True
    )

    history = model.fit(
        [x_train, x1_train],
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[callback],
        validation_data=([x_valid, x1_valid], y_valid),
        verbose=1
    )

    y_pred = model.predict([x_test, x1_test])
    
    wmap = wmape(y_test, y_pred)
    
    print("\n=== Evaluation ===")
    print(f"WMAPE : {wmap}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True)
    args = parser.parse_args()

    with open(args.cfg, "r") as f:
        config = json.load(f)
    
    train(config)