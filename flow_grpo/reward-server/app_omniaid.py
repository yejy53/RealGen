from PIL import Image
from io import BytesIO
import pickle
import traceback
from reward_server.omniaid_scorer import OmniAIDScorer
import os
import torch

from flask import Flask, request, Blueprint

root = Blueprint("root", __name__)

INFERENCE_FN = None

def create_app():
    app = Flask(__name__)
    app.register_blueprint(root)
    return app

@root.route("/", methods=["POST"])
def inference():
    global INFERENCE_FN
    if INFERENCE_FN is None:
        print(f"Worker (PID: {os.getpid()}) is loading the model...")
        INFERENCE_FN = OmniAIDScorer(
            device="cuda"
        )
        print(f"Worker (PID: {os.getpid()}) model loaded.")

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
PORT = 18087

if __name__ == "__main__":
    create_app().run(HOST, PORT)