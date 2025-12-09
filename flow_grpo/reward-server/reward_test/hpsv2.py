import requests
from PIL import Image
import io
import pickle
import glob
import tqdm
from concurrent.futures import ThreadPoolExecutor
import os


def f(_):

    image_paths = [
        "xxx",
        "xxx"
    ]
    prompts = [
        "portrait of a man",
        "A realistic portrait of a young woman, clear facial features, smooth black hair, natural makeup, wearing casual clothes, photographed in soft natural light."
    ]
    image_paths = [Image.open(img) for img in image_paths]
    data = {
        "images": image_paths,
        "prompts": prompts,
    }
    data_bytes = pickle.dumps(data)

    # Send the JPEG data in an HTTP POST request to the server
    url = "http://127.0.0.1:18086"
    response = requests.post(url, data=data_bytes)
    # Print the response from the server
    response_data = pickle.loads(response.content)
    print(response_data)

# with ThreadPoolExecutor(max_workers=8) as executor:
#     for _ in executor.map(f, range(8)):
#         pass
f(1)