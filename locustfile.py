"""
Locust file for load testing the API.
"""

import random
import glob
from pathlib import Path
from io import BytesIO
from PIL import Image
from locust import HttpUser, task, between


data_folder_path = Path("data/raw/fashion_mnist")


class ApiUser(HttpUser):
    """
    This class defines a user which will be spawned by Locust when load testing.
    """

    # Default host
    host = "https://api-nico-fi.cloud.okteto.net"

    # Delay after each task execution
    wait_time = between(1, 5)


    @task(1)
    def health_check(self):
        """
        Check the availability of the API.
        """
        self.client.get("/")


    @task(10)
    def predict_valid_item(self):
        """
        Make a prediction on a valid item.
        """
        sample = random.choice(glob.glob(str(data_folder_path / "*.png")))
        with open(sample, "rb") as image:
            self.client.post("/model", files={"file": image})


    @task(2)
    def predict_invalid_item(self):
        """
        Make a prediction on an invalid item.
        """
        stream = BytesIO()
        size = random.randint(40, 300)
        Image.new("L", (size, size)).save(stream, format="png")
        with self.client.post("/model", files={"file": stream.getvalue()},
            catch_response=True) as response:
            if response.status_code == 400:
                response.success()


    @task(2)
    def get_parameters(self):
        """
        Get the model parameters.
        """
        self.client.get("/model/parameters")


    @task(2)
    def get_metrics(self):
        """
        Get the model evaluation metrics.
        """
        self.client.get("/model/metrics")
