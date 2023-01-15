"""
This script evaluates the model on the test data.
The metric used for the evaluation is the accuracy.
"""

import pickle
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score


# Path to the prepared data folder
input_folder_path = Path("data/processed")

# Path to the models folder
model_folder_path = Path("models")

# Read the test data
X_test = np.load(input_folder_path / "X_test.npy")
y_test = np.load(input_folder_path / "y_test.npy")

# Load the model
with open(model_folder_path / "model.pkl", "rb") as pickled_model:
    model = pickle.load(pickled_model)

# Compute predictions using the model
predictions = model.predict(X_test)

# Compute the accuracy value for the model
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy}")
