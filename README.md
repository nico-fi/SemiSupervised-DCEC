---
[![Quality Assurance](https://github.com/nico-fi/SemiSupervised-DCEC/actions/workflows/quality_assurance.yml/badge.svg)](https://github.com/nico-fi/SemiSupervised-DCEC/actions/workflows/quality_assurance.yml)
[![Deployment](https://github.com/nico-fi/SemiSupervised-DCEC/actions/workflows/deployment.yml/badge.svg)](https://github.com/nico-fi/SemiSupervised-DCEC/actions/workflows/deployment.yml)
[![Better Uptime Badge](https://betteruptime.com/status-badges/v1/monitor/lu7r.svg)](https://sdcec.betteruptime.com)
---

SemiSupervised DCEC
==============================

Semi-supervised image clustering with convolutional autoencoder.

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── app                <- API built with FastAPI.
    │   ├── api.py
    │   ├── Dockerfile
    │   ├── monitoring.py
    │   └── requirements.txt
    │
    ├── data
    │   ├── ge             <- Metadata for Great Expectations.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   ├── raw            <- The original, immutable data dump.
    │   └── samples        <- Sample images.
    │
    ├── great_expectations <- Great Expectations directory.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries.
    │
    ├── monitoring         <- Monitoring tools.
    │   ├── grafana
    │   ├── prometheus
    │   └── alibi_detect.ipynb
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── src                <- Source code for use in this project.
    │   ├── data           <- Scripts to download or generate data.
    │   │   └── prepare_dataset.py
    │   │
    │   └── models         <- Scripts to train models and make predictions.
    │       ├── evaluate_model.py
    │       └── train_model.py
    │
    ├── tests              <- Tests for ML pipeline, model behavior and API.
    │   ├── test_api.py
    │   ├── test_behavioral.py
    │   ├── test_evaluate_model.py
    │   ├── test_prepare_dataset.py
    │   └── test_train_model.py
    │
    ├── web_app            <- Web interface for the API.
    │   ├── Dockerfile
    │   ├── requirements.txt
    │   └── web_app.py
    │
    ├── docker-compose.yaml<- Compose file defining services of Docker application.
    ├── dvc.lock           <- File recording the state of DVC pipeline.
    ├── dvc.yaml           <- File recording DVC pipeline stages.
    ├── locustfile.py      <- File for performing load tests.
    ├── params.yaml        <- Parameters for building the model.
    └── requirements.txt   <- The requirements file for reproducing the analysis environment.


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
