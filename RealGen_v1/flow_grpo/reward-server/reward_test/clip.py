import requests
from PIL import Image
import io
import pickle
import glob
import tqdm
from concurrent.futures import ThreadPoolExecutor
import os
import numpy as np
import torch


def f(_):

    images=[
        "xxx",
        "xxx"
    ]
    prompts=[
        'Pensive boy with glasses resting his head on a wooden chair',
        'Pensive boy with glasses resting his head on a wooden chair'
    ]
    image_paths = [Image.open(img) for img in images]
    images = [np.array(img) for img in image_paths]
    images = np.array(images)
    images = images.transpose(0, 3, 1, 2)  # NHWC -> NCHW
    image_paths = torch.tensor(images, dtype=torch.uint8)/255.0
    data = {
        "images": image_paths,
        "prompts": prompts,
    }
    data_bytes = pickle.dumps(data)

    # Send the JPEG data in an HTTP POST request to the server
    url = "http://127.0.0.1:18088"
    response = requests.post(url, data=data_bytes)
    # Print the response from the server
    response_data = pickle.loads(response.content)
    print(response_data)

# with ThreadPoolExecutor(max_workers=8) as executor:
#     for _ in executor.map(f, range(8)):
#         pass
f(1)