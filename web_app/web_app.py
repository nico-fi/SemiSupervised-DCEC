"""
This module defines the Gradio interface.
"""

import glob
from pathlib import Path
import requests
import gradio as gr


samples_folder_path = Path("data/samples")
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


def classify_image(inp):
    """
    Classify an image using the API.
    """
    url = "http://api:5000/model"
    with open(inp, "rb") as image:
        response = requests.request("POST", url, files={"file": image}, timeout=5)
    prediction = response.json()["data"]["prediction"]
    return {article_type[i]: prediction[i] for i in range(len(prediction))}


gr.Interface(
    title="Fashion-MNIST with SemiSupervised DCEC",
    allow_flagging="never",
    fn=classify_image,
    inputs=gr.Image(image_mode="L", type="filepath", show_label=False),
    outputs=gr.Label(label="Prediction", num_top_classes=3),
    examples=glob.glob(str(samples_folder_path / "[0-9].png"))
).launch(server_name="0.0.0.0", server_port=5001, show_error=True)
