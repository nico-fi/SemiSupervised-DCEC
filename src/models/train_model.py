"""
This script trains the model and saves it.
"""

import yaml
import mlflow
import numpy as np
from pathlib import Path
from sdcec import SDCEC


mlflow.set_tracking_uri("https://dagshub.com/nico-fi/SemiSupervised-DCEC.mlflow")
mlflow.set_experiment("Train Model")
mlflow.start_run()

# Path of the parameters file
params_path = Path("params.yaml")

# Path of the prepared data folder
input_folder_path = Path("data/processed")

# Read training dataset
X = np.load(input_folder_path / "X.npy")
y_train = np.load(input_folder_path / "y_train.npy")

# Read training parameters
with open(params_path, "r") as params_file:
    try:
        params = yaml.safe_load(params_file)
        params = params["train"]
    except yaml.YAMLError as exc:
        print(exc)

# Log parameters
mlflow.log_params(
    {
        "batch_size": params["batch_size"],
        "epochs": params["epochs"],
        "max_iter": params["max_iter"],
        "tol": params["tol"],
    }
)

# Instantiate the model
model = SDCEC(input_shape=X.shape[1:], n_clusters=len(np.unique(y_train)) - 1)
model.compile()

# Train the model
model.fit(X, y_train, batch_size=params["batch_size"], epochs=params["epochs"], max_iter=params["max_iter"], tol=params["tol"])

# Save the model
Path("models").mkdir(exist_ok=True)
output_folder_path = Path("models")
model_file_path = output_folder_path / "model.tf"
model.save(model_file_path)

# Log the model
mlflow.log_artifacts(model_file_path)

mlflow.end_run()
