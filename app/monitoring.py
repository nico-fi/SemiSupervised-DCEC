"""
This module specifies the metrics collected by Prometheus.
"""

import os
from typing import Callable
from prometheus_client import Counter
from prometheus_fastapi_instrumentator import Instrumentator, metrics
from prometheus_fastapi_instrumentator.metrics import Info


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
                predicted_class = info.response.headers.get("X-model-prediction")
                METRIC.labels(predicted_class).inc()

    return instrumentation


instrumentator.add(class_prediction(metric_namespace=NAMESPACE, metric_subsystem=SUBSYSTEM))
