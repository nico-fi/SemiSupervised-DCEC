"""
This module contains the FastAPI application and all of its endpoints.
"""

from datetime import datetime
from http import HTTPStatus
from pathlib import Path
from enum import Enum
import json
import yaml
import numpy as np
from keras.models import load_model
from fastapi import FastAPI, UploadFile
from PIL import Image, UnidentifiedImageError


class ArticleType(Enum):
    """
    Enum class for the article types.
    """
    T_SHIRT = 0
    TROUSER = 1
    PULLOVER = 2
    DRESS = 3
    COAT = 4
    SANDAL = 5
    SHIRT = 6
    SNEAKER = 7
    BAG = 8
    ANKLE_BOOT = 9


model_folder_path = Path("models")
artifacts = {}


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
def _predict(image: UploadFile):
    """
    Classify a new data point.
    """
    try:
        image = Image.open(image.file).convert("L").resize((28, 28))
    except UnidentifiedImageError:
        return {
            "message": HTTPStatus.BAD_REQUEST.phrase,
            "status-code": HTTPStatus.BAD_REQUEST,
            "data": {"message": "The image could not be identified. "
                                "Please, try again with a different image."}
        }
    image = np.expand_dims(image, axis=0) / 255.0
    prediction = artifacts["model"].predict(image)
    predicted_class = prediction.argmax().item()
    predicted_type = ArticleType(predicted_class).name
    confidence = prediction.max().item()
    return {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {
            "predicted_class": predicted_class,
            "predicted_type": predicted_type,
            "confidence": confidence,
        }
    }
