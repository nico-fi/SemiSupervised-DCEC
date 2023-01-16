"""
This script turns raw data into cleaned data ready to be analyzed.
"""

import yaml
import glob
import mlflow
import numpy as np
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split


mlflow.set_tracking_uri("https://dagshub.com/nico-fi/SemiSupervised-DCEC.mlflow")
mlflow.set_experiment("Prepare Data")
mlflow.start_run()

# Path of the parameters file
params_path = Path("params.yaml")

# Path of the input data folder
input_folder_path = Path("data/raw/fashion_mnist")

# Read images and labels
print("Processing images...")
X, y = [], []
for image_path in glob.glob("data/raw/fashion_mnist/*.png"):
     image = Image.open(image_path)
     X.append(np.array(image))
     y.append(int(image_path.split('/')[-1].split('_')[1].split('.')[0]))

# Convert to numpy array
X = np.expand_dims(X, axis=-1)
y = np.array(y)

# Normalize pixel values to be between 0 and 1
X = X / 255.0

# Read data preparation parameters
with open(params_path, "r") as params_file:
    try:
        params = yaml.safe_load(params_file)
        params = params["prepare"]
    except yaml.YAMLError as exc:
        print(exc)

# Log parameters
mlflow.log_param("random_state", params["random_state"])
mlflow.log_param("supervision", params["supervision"])

# Split supervised data from unsupervised data
i_train, i_test = train_test_split(range(len(X)), train_size=params["supervision"], random_state=params["random_state"], stratify=y)
y_train = np.full(len(X), -1, dtype=int)
y_train[i_train] = y[i_train]
X_test = X[i_test]
y_test = y[i_test]

# Path of the output data folder
Path("data/processed").mkdir(exist_ok=True)
prepared_folder_path = Path("data/processed")

# Save data
np.save(prepared_folder_path / "X.npy", X)
np.save(prepared_folder_path / "y_train.npy", y_train)
np.save(prepared_folder_path / "X_test.npy", X_test)
np.save(prepared_folder_path / "y_test.npy", y_test)

mlflow.end_run()
