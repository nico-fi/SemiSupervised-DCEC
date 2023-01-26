"""
Tests for the model evaluation script.
"""

from pathlib import Path
import json
from src.models import evaluate_model


model_folder_path = Path("models")


def test_evaluate_model():
    """
    Tests that the model evaluation script creates the expected files and
    that the accuracy value is better than the baseline.
    """
    evaluate_model.main()
    assert (model_folder_path / "score.json").is_file()
    with open(model_folder_path / "score.json", "r", encoding="utf-8") as score_file:
        accuracy = json.load(score_file)["accuracy"]
    assert isinstance(accuracy, float)
    assert 0.6 < accuracy <= 1
