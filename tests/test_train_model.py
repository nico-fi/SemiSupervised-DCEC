"""
Tests for the model training script.
"""

from pathlib import Path
import yaml
from keras.models import load_model, Model
from src.models import train_model


params_path = Path("params.yaml")
model_file_path = Path("models/model.tf")


def test_training_parameters():
    """
    Tests that the parameter values are valid.
    """
    with open(params_path, "r", encoding="utf-8") as params_file:
        params = yaml.safe_load(params_file)["train"]
    assert isinstance(params["batch_size"], int)
    assert isinstance(params["epochs"], int)
    assert isinstance(params["max_iter"], int)
    assert isinstance(params["tol"], float)
    assert params["batch_size"] in [64, 128, 256, 512]
    assert params["epochs"] > 3
    assert params["max_iter"] > 1000
    assert 0.0001 < params["tol"] < 1


def test_train_model():
    """
    Tests that the model training script creates the expected files.
    """
    train_model.main()
    assert model_file_path.exists()
    loaded_model = load_model(model_file_path)
    assert isinstance(loaded_model, Model)
