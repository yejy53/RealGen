import requests
from PIL import Image
import io
import pickle
import glob
import tqdm
from concurrent.futures import ThreadPoolExecutor
import os


def f(_):

    images=[  
        'xxx',
    ]
    pil_images = [Image.open(img) for img in images]
    data_bytes = pickle.dumps(pil_images)

    # Send the JPEG data in an HTTP POST request to the server
    url = "http://127.0.0.1:18085"
    response = requests.post(url, data=data_bytes)
    # print(f"服务器响应状态码: {response.status_code}")
    # print("服务器返回的原始内容:")
    # Print the response from the server
    response_data = pickle.loads(response.content)
    print(response_data)

# with ThreadPoolExecutor(max_workers=8) as executor:
#     for _ in executor.map(f, range(8)):
#         pass
f(1)