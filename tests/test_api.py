"""
Tests for the API.
"""

from io import BytesIO
from PIL import Image
from fastapi.testclient import TestClient
import pytest
from app.api import app


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


def test_predict_valid_item(client): # pylint: disable=redefined-outer-name
    """
    Test that the API returns a response after a valid prediction request.
    """
    stream = BytesIO()
    Image.new("L", (28, 28)).save(stream, format="png")
    response = client.post("/model", files={"file": stream.getvalue()})
    assert response.status_code == 200
    assert response.request.method == "POST"
    prediction = response.json()["data"]["prediction"]
    assert isinstance(prediction, list)
    assert all(isinstance(value, float) for value in prediction)
    assert sum(prediction) == pytest.approx(1)


def test_predict_invalid_item(client): # pylint: disable=redefined-outer-name
    """
    Test that the API returns an error after an invalid prediction request.
    """
    stream = BytesIO()
    Image.new("L", (30, 30)).save(stream, format="png")
    response = client.post("/model", files={"file": stream.getvalue()})
    assert response.status_code == 400
    assert response.request.method == "POST"
