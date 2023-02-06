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
from fastapi import FastAPI, File, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError
from .monitoring import instrumentator


model_folder_path = Path("models")
artifacts = {}
app = FastAPI(
    title="Fashion MNIST Classifier",
    description="This API lets you make predictions on the Fashion MNIST dataset using SDCEC.",
    version="1.0",
)
instrumentator.instrument(app).expose(app, include_in_schema=False, should_gzip=True)


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
    except (UnidentifiedImageError, AssertionError) as exc:
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST,
            detail="The image could not be identified. "
            "Please, try again with a different image.") from exc

    image = np.expand_dims(image, axis=0) / 255.0
    prediction = artifacts["model"].predict(image)[0]
    content = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "timestamp": datetime.now().isoformat(),
        "data": {"prediction": prediction.tolist()}
    }
    headers={"model-prediction": str(prediction.argmax())}
    return JSONResponse(content=content, headers=headers)
