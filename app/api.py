"""
This module contains the FastAPI application and all of its endpoints.
"""

from datetime import datetime
from http import HTTPStatus
import io
from pathlib import Path
import json
import yaml
import numpy as np
from keras.models import load_model
from fastapi import FastAPI, File
from PIL import Image, UnidentifiedImageError


model_folder_path = Path("models")
artifacts = {}
article_type = {
    0:'T-shirt/top',
    1:'Trouser',
    2:'Pullover',
    3:'Dress',
    4:'Coat',
    5:'Sandal',
    6:'Shirt',
    7:'Sneaker',
    8:'Bag',
    9:'Ankle boot'
}


app = FastAPI(
    title="Fashion MNIST Classifier",
    description="This API lets you make predictions on the Fashion MNIST dataset using SDCEC.",
    version="1.0",
)


@app.on_event("startup")
def _load_artifacts():
    """
    Load the model, the training parameters and the evaluation metrics.
    """
    artifacts["model"] = load_model(model_folder_path / "model.tf")
    with open("params.yaml", "r", encoding="utf-8") as params_file:
        artifacts["params"] = yaml.safe_load(params_file)
    with open(model_folder_path / "score.json", "r", encoding="utf-8") as score_file:
        artifacts["metrics"] = json.load(score_file)


@app.get("/")
def _index():
    """
    Health check.
    """
    return {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "timestamp": datetime.now().isoformat(),
        "data": {"message": "Welcome to the Fashion MNIST Classifier API! "
                            "Please, read the `/docs`."},
    }


@app.get("/model/parameters")
def _get_parameters():
    """
    Get the parameters used for data preparation and model training.
    """
    return {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "timestamp": datetime.now().isoformat(),
        "data": artifacts["params"],
    }


@app.get("/model/metrics")
def _get_metrics():
    """
    Get the metrics obtained from the model evaluation.
    """
    return {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "timestamp": datetime.now().isoformat(),
        "data": artifacts["metrics"],
    }


@app.post("/model")
def _predict(file: bytes = File(...)):
    """
    Classify a data point.
    """
    try:
        image = Image.open(io.BytesIO(file))
        assert image.size == (28, 28)
    except (UnidentifiedImageError, AssertionError):
        return {
            "message": HTTPStatus.BAD_REQUEST.phrase,
            "status-code": HTTPStatus.BAD_REQUEST,
            "timestamp": datetime.now().isoformat(),
            "data": {"message": "The image could not be identified. "
                                "Please, try again with a different image."}
        }
    image = np.expand_dims(image, axis=0) / 255.0
    prediction = artifacts["model"].predict(image)
    predicted_class = prediction.argmax().item()
    predicted_type = article_type[predicted_class]
    confidence = prediction.max().item()
    return {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "timestamp": datetime.now().isoformat(),
        "data": {
            "predicted_class": predicted_class,
            "predicted_type": predicted_type,
            "confidence": confidence,
        }
    }
