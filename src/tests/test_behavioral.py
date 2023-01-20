"""
This script tests the behavior of the model on sample images.
"""

import glob
import random
from pathlib import Path
import numpy as np
from PIL import Image
from keras.models import load_model
import pytest


samples_folder_path = Path("src/tests/samples")
model_path = Path("models/model.tf")

@pytest.fixture(scope="module")
def create_data():
    """
    Creates some corrupted images and saves them to the samples folder.
    Returns the original, rotated, and noisy images with their class id.
    """
    x_original, x_rotated, x_noisy, y_original  = [], [], [], []
    random.seed(42)

    for image_path in glob.glob(str(samples_folder_path / "[0-9].png")):
        image = Image.open(image_path)
        class_id = int(image_path.split('/')[-1][0])

        rotated = image.rotate(20)
        noisy = np.reshape(image, -1)
        noise_amount = int(0.05 * len(noisy))
        noise_coords = random.sample(range(len(noisy)), noise_amount)
        noisy[noise_coords] = random.choices([0, 255], k=noise_amount)
        noisy = Image.fromarray(noisy.reshape(image.size))

        rotated.save(samples_folder_path / Path(str(class_id) + "_rotated.png"))
        noisy.save(samples_folder_path / Path(str(class_id) + "_noisy.png"))

        x_original.append(np.array(image))
        x_rotated.append(np.array(rotated))
        x_noisy.append(np.array(noisy))
        y_original.append(class_id)

    x_original = np.array(x_original) / 255.0
    x_rotated = np.array(x_rotated) / 255.0
    x_noisy = np.array(x_noisy) / 255.0
    y_original = np.array(y_original)
    return x_original, x_rotated, x_noisy, y_original

@pytest.fixture(scope="module")
def get_predictions(create_data):
    """
    Loads the trained model and returns the predictions for the original,
    rotated, and noisy images with their true class id.
    """
    x_original = create_data[0]
    x_rotated = create_data[1]
    x_noisy = create_data[2]
    y_original = create_data[3]
    model = load_model(model_path)
    pred_original = model.predict(x_original).argmax(axis=1)
    pred_rotated = model.predict(x_rotated).argmax(axis=1)
    pred_noisy = model.predict(x_noisy).argmax(axis=1)
    return pred_original, pred_rotated, pred_noisy, y_original

def test_invariance(get_predictions):
    """
    Tests that the model is invariant to rotation and noise.
    """
    pred_original = get_predictions[0]
    pred_rotated = get_predictions[1]
    pred_noisy = get_predictions[2]
    assert (pred_original == pred_rotated).all()
    assert (pred_original == pred_noisy).all()

def test_directional(get_predictions):
    """
    Tests that the model provides different predictions for samples
    belonging to different classes.
    """
    pred_original = get_predictions[0]
    assert len(pred_original) == len(set(pred_original))

def test_minimum_functionality(get_predictions):
    """
    Tests that the model provides the correct predictions for some input samples.
    """
    pred_original = get_predictions[0]
    y_original = get_predictions[-1]
    assert (pred_original == y_original).all()
