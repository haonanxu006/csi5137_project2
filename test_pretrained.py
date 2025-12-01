import numpy as np
from run_model_all import load_and_test_model
from sklearn.model_selection import train_test_split
import json
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True)
    args = parser.parse_args()

    with open(args.cfg, "r") as f:
        config = json.load(f)

    x = np.load(config["x"])
    x1 = np.load(config["x1"])
    y = np.load(config["y"])
    y_ds = np.load(config["y_ds"])
    m = config["m"]

    x, x_test, x1, x1_test, y, y_test, y_ds, y_ds_test = train_test_split(
        x, x1, y, y_ds, 
        test_size=0.2, 
        random_state=42
    )

    load_and_test_model(m, x1_test, x_test, y_test, y_ds_test)