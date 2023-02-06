"""
This module contains the FastAPI application and all of its endpoints.
"""

from datetime import datetime
from http import HTTPStatus
from typing import Callable
from pathlib import Path
import io
import os
import json
import yaml
import numpy as np
from keras.models import load_model
from fastapi import FastAPI, File, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError
from prometheus_client import Counter
from prometheus_fastapi_instrumentator import Instrumentator, metrics
from prometheus_fastapi_instrumentator.metrics import Info


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


def class_prediction(
    metric_name: str = "class_prediction",
    metric_doc: str = "Class predicted by the model",
    metric_namespace: str = "",
    metric_subsystem: str = "") -> Callable[[Info], None]:
    """
    This function creates a metric that tracks the class predicted by the model.
    """
    METRIC = Counter( # pylint: disable=invalid-name
        metric_name,
        metric_doc,
        labelnames=["class"],
        namespace=metric_namespace,
        subsystem=metric_subsystem,
    )
    def instrumentation(info: Info) -> None:
        if info.modified_handler == "/model":
            if info.response.status_code == 200:
                predicted_class = info.response.headers.get("model-prediction")
                METRIC.labels(predicted_class).inc()
    return instrumentation


NAMESPACE = os.environ.get("METRICS_NAMESPACE", "api")
SUBSYSTEM = os.environ.get("METRICS_SUBSYSTEM", "model")

instrumentator = Instrumentator(
    should_ignore_untemplated=True,
    should_instrument_requests_inprogress=True,
    inprogress_name="api_inprogress",
    inprogress_labels=True,
)

instrumentator.add(metrics.request_size(metric_namespace=NAMESPACE, metric_subsystem=SUBSYSTEM))
instrumentator.add(metrics.response_size(metric_namespace=NAMESPACE, metric_subsystem=SUBSYSTEM))
instrumentator.add(metrics.latency(metric_namespace=NAMESPACE, metric_subsystem=SUBSYSTEM))
instrumentator.add(metrics.requests(metric_namespace=NAMESPACE, metric_subsystem=SUBSYSTEM))
instrumentator.add(class_prediction(metric_namespace=NAMESPACE, metric_subsystem=SUBSYSTEM))
instrumentator.instrument(app).expose(app, include_in_schema=False, should_gzip=True)
