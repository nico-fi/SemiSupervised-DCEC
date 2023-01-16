"""
This script evaluates the model on the test data.
The metric used for the evaluation is the accuracy.
"""

import mlflow
import numpy as np
from pathlib import Path
from keras.models import load_model
from sklearn.metrics import accuracy_score


mlflow.set_tracking_uri("https://dagshub.com/nico-fi/SemiSupervised-DCEC.mlflow")
mlflow.set_experiment("Evaluate")
mlflow.start_run()

# Path to the prepared data folder
input_folder_path = Path("data/processed")

# Path to the models folder
model_folder_path = Path("models")

# Read the test data
X_test = np.load(input_folder_path / "X_test.npy")
y_test = np.load(input_folder_path / "y_test.npy")

# Load the model
model = load_model(model_folder_path / "model.tf")

# Compute predictions using the model
predictions = model.predict(X_test)

# Compute the accuracy value for the model
accuracy = accuracy_score(y_test, predictions.argmax(axis=1))
print(f"Accuracy: {accuracy}")

# Log the accuracy value
mlflow.log_metric("accuracy", accuracy)

mlflow.end_run()
