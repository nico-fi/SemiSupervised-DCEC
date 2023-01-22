"""
This module defines the Gradio interface.
"""

import glob
from pathlib import Path
import requests
import gradio as gr


samples_folder_path = Path("src/tests/samples")


def classify_image(inp):
    """
    Classify an image using the API.
    """
    url = "http://localhost:5000/model"
    with open(inp, "rb") as image:
        response = requests.request("POST", url, files={"file": image}, timeout=5)
    prediction = response.json()["data"]["predicted_type"]
    confidence = response.json()["data"]["confidence"]
    return {prediction: confidence}


gr.Interface(
    title="Fashion-MNIST with SemiSupervised DCEC",
    allow_flagging="never",
    fn=classify_image,
    inputs=gr.Image(image_mode="L", type="filepath", show_label=False),
    outputs=gr.Label(label="Prediction"),
    examples=glob.glob(str(samples_folder_path / "[0-9].png"))
).launch()
