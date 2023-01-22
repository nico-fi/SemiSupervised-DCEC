"""
Tests for the API.
"""

from pathlib import Path
from PIL import Image
from fastapi.testclient import TestClient
import pytest
from app.api import app


samples_folder_path = Path("src/tests/samples")


@pytest.fixture(scope="module")
def client():
    """
    Create a test client.
    """
    with TestClient(app) as cli:
        yield cli


def test_index(client): # pylint: disable=redefined-outer-name
    """
    Test that the index page returns a 200 response.
    """
    response = client.get("/")
    assert response.status_code == 200
    assert response.request.method == "GET"


def test_get_parameters(client): # pylint: disable=redefined-outer-name
    """
    Test that the parameters page returns the required values.
    """
    response = client.get("/model/parameters")
    assert response.status_code == 200
    assert response.request.method == "GET"
    assert len(response.json()["data"]) > 0


def test_get_metrics(client): # pylint: disable=redefined-outer-name
    """
    Test that the metrics page returns the required values.
    """
    response = client.get("/model/metrics")
    assert response.status_code == 200
    assert response.request.method == "GET"
    assert len(response.json()["data"]) > 0


def test_predict_item(client): # pylint: disable=redefined-outer-name
    """
    Test that the API returns a valid response after a prediction request.
    """
    with open(samples_folder_path / "0.png", "rb") as image:
        response = client.post("/model", files={"file": image})
    assert response.status_code == 200
    assert response.request.method == "POST"
    data = response.json()["data"]
    assert isinstance(data["predicted_class"], int)
    assert isinstance(data["predicted_type"], str)
    assert isinstance(data["confidence"], float)
    assert data["predicted_class"] >= 0
    assert 0 < data["confidence"] <= 1
