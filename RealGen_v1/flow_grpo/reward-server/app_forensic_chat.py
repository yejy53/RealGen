from PIL import Image
from io import BytesIO
import pickle
import traceback
from reward_server.forensic_chat_scorer import QwenVLScorer
import os
import torch

from flask import Flask, request, Blueprint

root = Blueprint("root", __name__)

def create_app():
    global INFERENCE_FN
    INFERENCE_FN = QwenVLScorer(
        device="cuda",
        dtype=torch.bfloat16
    )

    app = Flask(__name__)
    app.register_blueprint(root)
    return app

@root.route("/", methods=["POST"]) 
def inference():
    print(f"received POST request from {request.remote_addr}")
    data = request.get_data()

    try:
        images = pickle.loads(data)

        response = INFERENCE_FN(images)
        # print(response)

        response = pickle.dumps(response)

        returncode = 200
    except Exception as e:
        response = traceback.format_exc()
        # print(response)
        response = response.encode("utf-8")
        returncode = 500

    return response, returncode


HOST = "127.0.0.1"
# HOST = "10.140.28.89"
PORT = 18085

if __name__ == "__main__":
    create_app().run(HOST, PORT)