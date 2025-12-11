from PIL import Image
import io
import random
import numpy as np
import torch
from collections import defaultdict
import requests
from requests.adapters import HTTPAdapter, Retry
from io import BytesIO
import pickle

def qwen_fake_score(device):

    url = "http://127.0.0.1:18085"
    sess = requests.Session()
    retries = Retry(
        total=1000, backoff_factor=1, status_forcelist=[500], allowed_methods=False
    )
    sess.mount("http://", HTTPAdapter(max_retries=retries))

    def _fn(images):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
            images = [Image.fromarray(image) for image in images]
        data_bytes = pickle.dumps(images)
        response = sess.post(url, data=data_bytes, timeout=120)
        scores = pickle.loads(response.content)
        return scores, {}

    return _fn

def effort_score(device):

    url = "http://127.0.0.1:18087"
    sess = requests.Session()
    retries = Retry(
        total=1000, backoff_factor=1, status_forcelist=[500], allowed_methods=False
    )
    sess.mount("http://", HTTPAdapter(max_retries=retries))

    def _fn(images):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
            images = [Image.fromarray(image) for image in images]
        data_bytes = pickle.dumps(images)
        response = sess.post(url, data=data_bytes, timeout=600)
        scores = pickle.loads(response.content)
        return scores, {}

    return _fn

def hpsv2_score(device):

    url = "http://127.0.0.1:18086"
    sess = requests.Session()
    retries = Retry(
        total=1000, backoff_factor=1, status_forcelist=[500], allowed_methods=False
    )
    sess.mount("http://", HTTPAdapter(max_retries=retries))

    def _fn(images, prompts):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
            images = [Image.fromarray(image) for image in images]
        data = {
            "images": images,
            "prompts": prompts,
        }
        data_bytes = pickle.dumps(data)
        response = sess.post(url, data=data_bytes, timeout=120)
        scores = pickle.loads(response.content)
        return scores, {}

    return _fn

def clip_score(device):
    url = "http://127.0.0.1:18088"
    sess = requests.Session()
    retries = Retry(
        total=1000, backoff_factor=1, status_forcelist=[500], allowed_methods=False
    )
    sess.mount("http://", HTTPAdapter(max_retries=retries))

    def _fn(images, prompts):
        if not isinstance(images, torch.Tensor):
            images = images.transpose(0, 3, 1, 2)  # NHWC -> NCHW
            pixels = torch.tensor(pixels, dtype=torch.uint8)/255.0
        images = images.cpu()
        data = {
            "images": images,
            "prompts": prompts,
        }
        data_bytes = pickle.dumps(data)
        response = sess.post(url, data=data_bytes, timeout=120)
        scores = pickle.loads(response.content)
        return scores.cpu().tolist(), {}

    return _fn

# def clip_score(device):
#     from flow_grpo.clip_scorer import ClipScorer

#     scorer = ClipScorer(device=device, dtype=torch.float32)

#     def _fn(images, prompts):
#         if not isinstance(images, torch.Tensor):
#             images = images.transpose(0, 3, 1, 2)  # NHWC -> NCHW
#             images = torch.tensor(images, dtype=torch.uint8)/255.0
#         scores = scorer(images, prompts)
#         return scores.cpu().tolist(), {}

#     return _fn

def longclip_score(device):
    url = "http://127.0.0.1:18089"
    sess = requests.Session()
    retries = Retry(
        total=1000, backoff_factor=1, status_forcelist=[500], allowed_methods=False
    )
    sess.mount("http://", HTTPAdapter(max_retries=retries))

    def _fn(images, prompts):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
            images = [Image.fromarray(image) for image in images]
        
        data = {
            "images": images,
            "prompts": prompts,
        }
        data_bytes = pickle.dumps(data)
        response = sess.post(url, data=data_bytes, timeout=120)
        scores = pickle.loads(response.content)
        return scores, {}

    return _fn

def hpsv3_score(device):

    url = "http://10.140.28.37:18090"
    sess = requests.Session()
    retries = Retry(
        total=1000, backoff_factor=1, status_forcelist=[500], allowed_methods=False
    )
    sess.mount("http://", HTTPAdapter(max_retries=retries))

    def _fn(images, prompts):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
            images = [Image.fromarray(image) for image in images]
        data = {
            "images": images,
            "prompts": prompts,
        }
        data_bytes = pickle.dumps(data)
        response = sess.post(url, data=data_bytes, timeout=120)
        scores = pickle.loads(response.content)
        return scores, {}

    return _fn

def pickscore_score(device):

    url = "http://127.0.0.1:18091"
    sess = requests.Session()
    retries = Retry(
        total=1000, backoff_factor=1, status_forcelist=[500], allowed_methods=False
    )
    sess.mount("http://", HTTPAdapter(max_retries=retries))

    def _fn(images, prompts):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
            images = [Image.fromarray(image) for image in images]
        data = {
            "images": images,
            "prompts": prompts,
        }
        data_bytes = pickle.dumps(data)
        response = sess.post(url, data=data_bytes, timeout=120)
        scores = pickle.loads(response.content)
        return scores, {}

    return _fn

def multi_score(device, score_dict):
    score_functions = {
        "forensic_chat": qwen_fake_score,
        "omniaid": effort_score,
        "hpsv2": hpsv2_score,
        "hpsv3": hpsv3_score,
        "clipscore": clip_score,
        "longclip": longclip_score,
        "pickscore": pickscore_score
    }
    score_fns={}
    for score_name, weight in score_dict.items():
        score_fns[score_name] = score_functions[score_name](device) if 'device' in score_functions[score_name].__code__.co_varnames else score_functions[score_name]()

    # only_strict is only for geneval. During training, only the strict reward is needed, and non-strict rewards don't need to be computed, reducing reward calculation time.
    def _fn(images, prompts, metadata, ref_images=None, only_strict=True):
        total_scores = []
        score_details = {}
        
        for score_name, weight in score_dict.items():
            if score_name == "qwenvlfake":
                scores, rewards = score_fns[score_name](images)
            elif score_name == "effort":
                scores, rewards = score_fns[score_name](images)
            elif score_name == "hpsv2":
                scores, rewards = score_fns[score_name](images, prompts)
            elif score_name == "hpsv3":
                scores, rewards = score_fns[score_name](images, prompts)
            elif score_name == "clipscore":
                scores, rewards = score_fns[score_name](images, prompts)
            elif score_name == "longclip":
                scores, rewards = score_fns[score_name](images, prompts)
            elif score_name == "pickscore":
                scores, rewards = score_fns[score_name](images, prompts)
            else:
                scores, rewards = score_fns[score_name](images, prompts, metadata)
            score_details[score_name] = scores
            weighted_scores = [weight * score for score in scores]
            
            if not total_scores: 
                total_scores = weighted_scores
            else:
                total_scores = [total + weighted for total, weighted in zip(total_scores, weighted_scores)]
        
        score_details['avg'] = total_scores
        return score_details, {}

    return _fn

def main():
    import torchvision.transforms as transforms

    image_paths = [
        "nasa.jpg",
    ]

    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert to tensor
    ])

    images = torch.stack([transform(Image.open(image_path).convert('RGB')) for image_path in image_paths])
    prompts=[
        'A astronaut’s glove floating in zero-g with "NASA 2049" on the wrist',
    ]
    metadata = {}  # Example metadata
    score_dict = {
        "unifiedreward": 1.0
    }
    # Initialize the multi_score function with a device and score_dict
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scoring_fn = multi_score(device, score_dict)
    # Get the scores
    scores, _ = scoring_fn(images, prompts, metadata)
    # Print the scores
    print("Scores:", scores)


if __name__ == "__main__":
    main()
