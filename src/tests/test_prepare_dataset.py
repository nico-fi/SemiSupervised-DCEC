"""
Tests for the data preparation script.
"""

from pathlib import Path
import numpy as np
import yaml
import pytest
from ..data import prepare_dataset


params_path = Path("params.yaml")
prepared_folder_path = Path("data/processed")

def test_preparation_parameters():
    """
    Tests that the parameter values are valid.
    """
    with open(params_path, "r", encoding="utf-8") as params_file:
        params = yaml.safe_load(params_file)["prepare"]
    assert isinstance(params["supervision"], float)
    assert isinstance(params["random_state"], int)
    assert 0 < params["supervision"] < 1

def test_prepare_dataset():
    """
    Tests that the data preparation script creates the expected files.
    """
    prepare_dataset.main()
    assert (prepared_folder_path / "x.npy").is_file()
    assert (prepared_folder_path / "y_train.npy").is_file()
    assert (prepared_folder_path / "x_test.npy").is_file()
    assert (prepared_folder_path / "y_test.npy").is_file()
    x_data = np.load(prepared_folder_path / "x.npy")
    y_train = np.load(prepared_folder_path / "y_train.npy")
    x_test = np.load(prepared_folder_path / "x_test.npy")
    y_test = np.load(prepared_folder_path / "y_test.npy")
    assert len(x_data) == len(y_train)
    assert len(x_test) == len(y_test)
    assert len(x_test) < len(x_data)
    assert ((x_data >= 0) & (x_data <= 1)).all()
    assert ((x_test >= 0) & (x_test <= 1)).all()
    assert (y_train >= -1).all()
    assert (y_test >= 0).all()
    for class_id in np.unique(y_test):
        train_percentage = np.count_nonzero(y_train == class_id) / np.count_nonzero(y_train != -1)
        test_percentage = np.count_nonzero(y_test == class_id) / len(y_test)
        assert train_percentage == pytest.approx(test_percentage, abs=0.1)
