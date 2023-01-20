"""
This script evaluates the model on the test data.
The metric used for the evaluation is the accuracy.
"""

from pathlib import Path
import json
import numpy as np
from keras.models import load_model
from sklearn.metrics import accuracy_score
import mlflow


def main():
    """
    Evaluates the model on the test data.
    """
    mlflow.set_tracking_uri("https://dagshub.com/nico-fi/SemiSupervised-DCEC.mlflow")
    mlflow.set_experiment("Evaluate Model")
    mlflow.start_run()

    input_folder_path = Path("data/processed")
    model_folder_path = Path("models")

    # Read test data
    x_test = np.load(input_folder_path / "x_test.npy")
    y_test = np.load(input_folder_path / "y_test.npy")

    # Load the model and compute predictions
    model = load_model(model_folder_path / "model.tf")
    predictions = model.predict(x_test)

    # Compute and log the accuracy value
    accuracy = accuracy_score(y_test, predictions.argmax(axis=1))
    print(f"Accuracy: {accuracy}")
    with open(model_folder_path / "score.json", "w", encoding="utf-8") as score_file:
        json.dump({"accuracy": accuracy}, score_file, indent=4)
    mlflow.log_metric("accuracy", accuracy)

    mlflow.end_run()


if __name__ == "__main__":
    main()
